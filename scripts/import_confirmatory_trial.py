"""Import a completed trial into a larger confirmatory experiment plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


VOLATILE_CONTRACT_FIELDS = {
    "contract_id",
    "created_at",
    "fingerprint",
    "contract_fingerprint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed records from an exploratory plan into "
            "a compatible confirmatory plan."
        )
    )

    parser.add_argument(
        "--source-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-results",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--target-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments"),
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=1,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on line {line_number} of {path}"
            ) from error

    return records


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def cell_key(cell: dict[str, Any]) -> str:
    required_fields = {
        "phase",
        "model",
        "task_id",
        "attack",
        "injection_task_id",
        "condition",
        "trial",
    }

    missing = required_fields - set(cell)

    if missing:
        raise ValueError(
            f"Cell is missing required fields: {sorted(missing)}"
        )

    normalized = {
        field: cell[field]
        for field in sorted(required_fields)
    }

    return canonical_json(normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def remove_volatile_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_volatile_fields(item)
            for key, item in value.items()
            if key not in VOLATILE_CONTRACT_FIELDS
        }

    if isinstance(value, list):
        return [
            remove_volatile_fields(item)
            for item in value
        ]

    return value


def normalized_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(configuration)

    # Trial count is expected to differ between the exploratory and
    # confirmatory plans.
    normalized.pop("trials", None)

    return normalized


def compare_protocols(
    source_plan: dict[str, Any],
    target_plan: dict[str, Any],
) -> None:
    source_configuration = normalized_configuration(
        source_plan["configuration"]
    )

    target_configuration = normalized_configuration(
        target_plan["configuration"]
    )

    if source_configuration != target_configuration:
        print("SOURCE CONFIGURATION:")
        print(json.dumps(source_configuration, indent=2))

        print("TARGET CONFIGURATION:")
        print(json.dumps(target_configuration, indent=2))

        raise RuntimeError(
            "Source and target experiment configurations differ."
        )

    source_contracts = remove_volatile_fields(
        source_plan["contracts"]
    )

    target_contracts = remove_volatile_fields(
        target_plan["contracts"]
    )

    if source_contracts != target_contracts:
        print("SOURCE CONTRACTS:")
        print(json.dumps(source_contracts, indent=2))

        print("TARGET CONTRACTS:")
        print(json.dumps(target_contracts, indent=2))

        raise RuntimeError(
            "Source and target contract semantics differ."
        )

    source_rejected = remove_volatile_fields(
        source_plan.get("rejected_tasks", {})
    )

    target_rejected = remove_volatile_fields(
        target_plan.get("rejected_tasks", {})
    )

    if source_rejected != target_rejected:
        raise RuntimeError(
            "Source and target rejected-task manifests differ."
        )


def validate_source_records(
    records: list[dict[str, Any]],
    source_plan_id: str,
    trial: int,
) -> None:
    if not records:
        raise RuntimeError("Source results are empty.")

    incomplete = [
        record.get("cell_id")
        for record in records
        if record.get("status") != "completed"
    ]

    if incomplete:
        raise RuntimeError(
            "Source results contain incomplete records: "
            f"{incomplete}"
        )

    incorrect_plan_ids = {
        record.get("plan_id")
        for record in records
        if record.get("plan_id") != source_plan_id
    }

    if incorrect_plan_ids:
        raise RuntimeError(
            "Source result plan IDs do not match the source plan: "
            f"{sorted(str(value) for value in incorrect_plan_ids)}"
        )

    incorrect_trials = [
        record.get("cell_id")
        for record in records
        if record["cell"].get("trial") != trial
    ]

    if incorrect_trials:
        raise RuntimeError(
            "Source results contain records outside the requested "
            f"trial: {incorrect_trials}"
        )

    cell_ids = [
        record["cell_id"]
        for record in records
    ]

    duplicated_ids = [
        cell_id
        for cell_id, count in Counter(cell_ids).items()
        if count > 1
    ]

    if duplicated_ids:
        raise RuntimeError(
            f"Duplicate source cell IDs: {duplicated_ids}"
        )

    keys = [
        cell_key(record["cell"])
        for record in records
    ]

    duplicated_cells = [
        key
        for key, count in Counter(keys).items()
        if count > 1
    ]

    if duplicated_cells:
        raise RuntimeError(
            "Source results contain duplicate cell definitions."
        )


def main() -> int:
    args = parse_args()

    source_plan_path = args.source_plan.resolve()
    source_results_path = args.source_results.resolve()
    target_plan_path = args.target_plan.resolve()

    source_plan = load_json(source_plan_path)
    target_plan = load_json(target_plan_path)
    source_records = load_jsonl(source_results_path)

    source_plan_id = source_plan["plan_id"]
    target_plan_id = target_plan["plan_id"]

    if source_plan_id == target_plan_id:
        raise RuntimeError(
            "Source and target plans must be different."
        )

    compare_protocols(
        source_plan,
        target_plan,
    )

    validate_source_records(
        source_records,
        source_plan_id,
        args.trial,
    )

    source_plan_cells = {
        cell_key(cell): cell
        for cell in source_plan["cells"]
        if cell.get("trial") == args.trial
    }

    target_trial_cells = [
        cell
        for cell in target_plan["cells"]
        if cell.get("trial") == args.trial
    ]

    target_trial_keys = {
        cell_key(cell)
        for cell in target_trial_cells
    }

    source_record_map = {
        cell_key(record["cell"]): record
        for record in source_records
    }

    source_plan_keys = set(source_plan_cells)
    source_result_keys = set(source_record_map)

    if source_plan_keys != source_result_keys:
        missing_results = sorted(
            source_plan_keys - source_result_keys
        )

        unexpected_results = sorted(
            source_result_keys - source_plan_keys
        )

        raise RuntimeError(
            "Source results do not exactly cover the source trial. "
            f"Missing={len(missing_results)}, "
            f"unexpected={len(unexpected_results)}"
        )

    if source_result_keys != target_trial_keys:
        missing_in_source = sorted(
            target_trial_keys - source_result_keys
        )

        unexpected_in_source = sorted(
            source_result_keys - target_trial_keys
        )

        raise RuntimeError(
            "Source trial cells do not exactly match target trial "
            f"cells. Missing={len(missing_in_source)}, "
            f"unexpected={len(unexpected_in_source)}"
        )

    target_directory = (
        args.output_root.resolve()
        / target_plan_id
    )

    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_results_path = (
        target_directory / "results.jsonl"
    )

    if (
        target_results_path.exists()
        and target_results_path.stat().st_size > 0
    ):
        raise RuntimeError(
            f"Target results already exist: {target_results_path}. "
            "The importer refuses to overwrite them."
        )

    source_runs_root = (
        source_results_path.parent / "runs"
    )

    target_runs_root = (
        target_directory / "runs"
    )

    # Validate every source run before copying anything.
    missing_source_runs = []

    for record in source_records:
        source_result_path = (
            source_runs_root
            / record["cell_id"]
            / "result.json"
        )

        if not source_result_path.exists():
            missing_source_runs.append(
                str(source_result_path)
            )

    if missing_source_runs:
        raise RuntimeError(
            "Source per-run result files are missing: "
            + ", ".join(missing_source_runs)
        )

    target_runs_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    imported_records = []

    for target_cell in target_trial_cells:
        key = cell_key(target_cell)

        imported = dict(source_record_map[key])

        # Preserve original execution evidence while associating the
        # imported record with the confirmatory plan.
        imported["plan_id"] = target_plan_id
        imported["imported_from"] = {
            "source_plan_id": source_plan_id,
            "source_cell_id": imported["cell_id"],
            "source_trial": args.trial,
        }

        cell_id = imported["cell_id"]

        source_run_directory = (
            source_runs_root / cell_id
        )

        target_run_directory = (
            target_runs_root / cell_id
        )

        if target_run_directory.exists():
            raise RuntimeError(
                "Target run directory already exists: "
                f"{target_run_directory}"
            )

        shutil.copytree(
            source_run_directory,
            target_run_directory,
        )

        # Rewrite only the copied result metadata. Original source
        # evidence and source experiment files remain unchanged.
        (
            target_run_directory / "result.json"
        ).write_text(
            json.dumps(
                imported,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        imported_records.append(imported)

    temporary_results_path = (
        target_directory / "results.jsonl.tmp"
    )

    with temporary_results_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in imported_records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_results_path.replace(
        target_results_path
    )

    provenance = {
        "schema_version": "0.1",
        "target_plan_id": target_plan_id,
        "source_plan_id": source_plan_id,
        "imported_trial": args.trial,
        "imported_records": len(imported_records),
        "source_plan": str(source_plan_path),
        "source_results": str(source_results_path),
        "target_plan": str(target_plan_path),
        "source_plan_sha256": sha256_file(
            source_plan_path
        ),
        "source_results_sha256": sha256_file(
            source_results_path
        ),
        "target_plan_sha256": sha256_file(
            target_plan_path
        ),
        "protocol_configuration_match": True,
        "contract_semantics_match": True,
        "cell_coverage_match": True,
    }

    provenance_path = (
        target_directory
        / "trial_import_provenance.json"
    )

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    benign_count = sum(
        record["cell"]["phase"] == "benign"
        for record in imported_records
    )

    attack_count = sum(
        record["cell"]["phase"] == "attack"
        for record in imported_records
    )

    print("=== CONFIRMATORY TRIAL IMPORT ===")
    print(f"Source plan: {source_plan_id}")
    print(f"Target plan: {target_plan_id}")
    print(f"Imported trial: {args.trial}")
    print(f"Imported records: {len(imported_records)}")
    print(f"Benign records: {benign_count}")
    print(f"Attack records: {attack_count}")
    print("Configuration match: True")
    print("Contract semantics match: True")
    print("Cell coverage match: True")
    print(f"Results: {target_results_path}")
    print(f"Provenance: {provenance_path}")
    print(
        "PASS: Existing trial imported without making API calls."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
