#!/usr/bin/env python3
"""Build leakage-resistant leave-one-task-out SentryGraph folds.

Each of the six tasks is held out once for testing.  Whole tasks are also
assigned to validation and calibration, leaving the other tasks for training.
The builder writes both complete and feature-deduplicated partitions and fails
if tasks, contract fingerprints, groups, example IDs, or raw feature payloads
cross partition boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROLES = ("train", "validation", "calibration", "test")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def digest(value: Any, length: int | None = None) -> str:
    result = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return result if length is None else result[:length]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise RuntimeError(f"No examples found in {path}")
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def feature_hash(record: dict[str, Any]) -> str:
    return digest(record["features"])


def validate_source(records: list[dict[str, Any]]) -> list[str]:
    required = {
        "example_id", "features", "group_id", "labels", "source",
    }
    example_ids: set[str] = set()
    label_by_feature: dict[str, tuple[Any, Any]] = {}
    tasks: set[str] = set()

    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise RuntimeError(f"Record {index} missing {sorted(missing)}")
        source = record["source"]
        for key in ("task_id", "contract_fingerprint"):
            if not source.get(key):
                raise RuntimeError(f"Record {index} lacks source.{key}")
        example_id = record["example_id"]
        if example_id in example_ids:
            raise RuntimeError(f"Duplicate example_id: {example_id}")
        example_ids.add(example_id)
        tasks.add(source["task_id"])

        labels = (
            record["labels"].get("action_permitted"),
            record["labels"].get("trajectory_permitted"),
        )
        fingerprint = feature_hash(record)
        previous = label_by_feature.setdefault(fingerprint, labels)
        if previous != labels:
            raise RuntimeError(
                "Identical feature payload has conflicting labels: "
                f"{fingerprint}"
            )

    tasks = sorted(tasks)
    if len(tasks) < 6:
        raise RuntimeError(
            "At least six tasks are required for separate train, validation, "
            f"calibration, and test task sets; found {len(tasks)}."
        )
    return tasks


def has_binary_action_support(records: list[dict[str, Any]]) -> bool:
    labels = {
        bool(record["labels"]["action_permitted"])
        for record in records
    }
    return labels == {False, True}


def fold_assignment(
    tasks: list[str],
    test_index: int,
    binary_tasks: set[str],
) -> dict[str, list[str]]:
    """Assign binary-capable tuning tasks and reserve one task for testing."""
    count = len(tasks)
    test = tasks[test_index]
    ordered_candidates = [
        tasks[(test_index + offset) % count]
        for offset in range(1, count)
        if tasks[(test_index + offset) % count] in binary_tasks
    ]
    if len(ordered_candidates) < 2:
        raise RuntimeError(
            "At least two non-test tasks with both action classes are "
            "required for validation and calibration."
        )
    calibration = ordered_candidates[0]
    validation = ordered_candidates[1]
    reserved = {test, calibration, validation}
    train = [task for task in tasks if task not in reserved]
    return {
        "train": train,
        "validation": [validation],
        "calibration": [calibration],
        "test": [test],
    }


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item["example_id"]):
        selected.setdefault(feature_hash(record), record)
    unique = sorted(selected.values(), key=lambda item: item["example_id"])
    return unique, len(records) - len(unique)


def partition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    action = Counter(
        "allow" if row["labels"]["action_permitted"] else "deny"
        for row in records
    )
    trajectory = Counter(
        "allow" if row["labels"]["trajectory_permitted"] else "deny"
        for row in records
    )
    return {
        "examples": len(records),
        "tasks": sorted({row["source"]["task_id"] for row in records}),
        "groups": len({row["group_id"] for row in records}),
        "contracts": len({
            row["source"]["contract_fingerprint"] for row in records
        }),
        "unique_feature_payloads": len({feature_hash(row) for row in records}),
        "action": dict(sorted(action.items())),
        "trajectory": dict(sorted(trajectory.items())),
    }


def pairwise_overlap(
    partitions: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, int]]:
    extractors = {
        "tasks": lambda r: r["source"]["task_id"],
        "contracts": lambda r: r["source"]["contract_fingerprint"],
        "groups": lambda r: r["group_id"],
        "example_ids": lambda r: r["example_id"],
        "feature_payloads": feature_hash,
    }
    audit: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1:]:
            name = f"{left}__{right}"
            audit[name] = {}
            for field, extractor in extractors.items():
                left_values = {extractor(row) for row in partitions[left]}
                right_values = {extractor(row) for row in partitions[right]}
                audit[name][field] = len(left_values & right_values)
    return audit


def assert_zero_overlap(audit: dict[str, dict[str, int]]) -> None:
    findings = {
        pair: values
        for pair, values in audit.items()
        if any(values.values())
    }
    if findings:
        raise RuntimeError(
            "Cross-partition leakage detected:\n" +
            json.dumps(findings, indent=2, sort_keys=True)
        )


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build(source: Path, output: Path) -> dict[str, Any]:
    records = load_jsonl(source)
    tasks = validate_source(records)
    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_task[record["source"]["task_id"]].append(record)

    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))

    try:
        fold_manifests = []
        binary_tasks = {
            task
            for task in tasks
            if has_binary_action_support(records_by_task[task])
        }
        for test_index, test_task in enumerate(tasks):
            assignment = fold_assignment(tasks, test_index, binary_tasks)
            fold_name = f"fold_{test_task}"
            fold_dir = temporary / "folds" / fold_name
            full_dir = fold_dir / "full"
            unique_dir = fold_dir / "deduplicated"
            full_dir.mkdir(parents=True)
            unique_dir.mkdir(parents=True)

            partitions: dict[str, list[dict[str, Any]]] = {}
            unique_partitions: dict[str, list[dict[str, Any]]] = {}
            dropped: dict[str, int] = {}

            for role in ROLES:
                rows = [
                    record
                    for task in assignment[role]
                    for record in records_by_task[task]
                ]
                rows.sort(key=lambda item: item["example_id"])
                partitions[role] = rows
                unique_rows, removed = deduplicate(rows)
                unique_partitions[role] = unique_rows
                dropped[role] = removed
                write_jsonl(full_dir / f"{role}.jsonl", rows)
                write_jsonl(unique_dir / f"{role}.jsonl", unique_rows)

            full_overlap = pairwise_overlap(partitions)
            unique_overlap = pairwise_overlap(unique_partitions)
            assert_zero_overlap(full_overlap)
            assert_zero_overlap(unique_overlap)

            fold_manifest = {
                "fold": fold_name,
                "assignment": assignment,
                "test_evaluation": (
                    "binary_generalization"
                    if test_task in binary_tasks
                    else "allow_only_negative_control"
                ),
                "full": {
                    role: partition_summary(partitions[role]) for role in ROLES
                },
                "deduplicated": {
                    role: partition_summary(unique_partitions[role])
                    for role in ROLES
                },
                "duplicates_removed_within_partition": dropped,
                "cross_partition_overlap_full": full_overlap,
                "cross_partition_overlap_deduplicated": unique_overlap,
            }
            write_json(fold_dir / "manifest.json", fold_manifest)
            fold_manifests.append(fold_manifest)

        protocol = {
            "schema_version": "0.2",
            "protocol": "leave-one-task-out-with-task-held-out-tuning",
            "source": str(source),
            "source_sha256": file_sha256(source),
            "tasks": tasks,
            "binary_action_tasks": sorted(binary_tasks),
            "allow_only_tasks": sorted(set(tasks) - binary_tasks),
            "folds": fold_manifests,
            "guarantees": {
                "whole_task_assignment": True,
                "zero_task_overlap": True,
                "zero_contract_fingerprint_overlap": True,
                "zero_group_overlap": True,
                "zero_example_id_overlap": True,
                "zero_raw_feature_payload_overlap": True,
                "deduplicated_variant_available": True,
                "test_excluded_from_model_selection": True,
                "calibration_excluded_from_model_selection": True,
                "validation_has_both_action_classes": True,
                "calibration_has_both_action_classes": True,
            },
        }
        protocol["protocol_id"] = digest(protocol, 20)
        write_json(temporary / "manifest.json", protocol)

        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        checksum_lines = [
            f"{file_sha256(path)}  {path.relative_to(temporary)}"
            for path in files
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}. Choose a new directory."
            )
        os.replace(temporary, output)
        return protocol
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task-held-out graph-conformance folds."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.dataset_root / "all_examples.jsonl"
    protocol = build(source, args.output_dir)
    print("=== PHASE 1F GENERALIZATION FOLDS ===")
    print("Protocol ID:", protocol["protocol_id"])
    print("Output:", args.output_dir)
    print("Tasks:", len(protocol["tasks"]))
    print("Folds:", len(protocol["folds"]))
    for fold in protocol["folds"]:
        test_task = fold["assignment"]["test"][0]
        full_test = fold["full"]["test"]["examples"]
        unique_test = fold["deduplicated"]["test"]["examples"]
        print(
            f"  {test_task}: test={full_test}, "
            f"deduplicated_test={unique_test}, "
            f"evaluation={fold['test_evaluation']}, "
            "cross_partition_overlap=0"
        )
    print("PASS: All folds have disjoint tasks, contracts, groups, examples, and raw features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
