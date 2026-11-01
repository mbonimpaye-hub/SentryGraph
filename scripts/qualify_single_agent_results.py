"""Create a qualification manifest from benign experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records_by_cell = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        records_by_cell[record["cell_id"]] = record

    return list(records_by_cell.values())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify model-task-trial groups for attack evaluation."
        )
    )

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    plan = json.loads(
        args.plan.read_text()
    )

    plan_id = plan["plan_id"]

    experiment_root = (
        args.output_root
        / plan_id
    )

    results_path = (
        experiment_root
        / "results.jsonl"
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found: {results_path}"
        )

    records = load_jsonl(results_path)

    benign = {
        (
            record["cell"]["model"],
            record["cell"]["task_id"],
            record["cell"]["trial"],
            record["cell"]["condition"],
        ): record
        for record in records
        if record["cell"]["phase"] == "benign"
        and record.get("status") == "completed"
    }

    qualified = []
    excluded = []

    configuration = plan["configuration"]

    for model in configuration["models"]:
        for task_id in configuration["task_ids"]:
            for trial in range(
                1,
                configuration["trials"] + 1,
            ):
                baseline = benign.get(
                    (
                        model,
                        task_id,
                        trial,
                        "baseline",
                    )
                )

                sentrygraph = benign.get(
                    (
                        model,
                        task_id,
                        trial,
                        "sentrygraph",
                    )
                )

                reasons = []

                if baseline is None:
                    reasons.append(
                        "missing_baseline_result"
                    )

                elif baseline.get("utility") is not True:
                    reasons.append(
                        "baseline_utility_failed"
                    )

                if sentrygraph is None:
                    reasons.append(
                        "missing_sentrygraph_result"
                    )

                else:
                    if (
                        sentrygraph.get("utility")
                        is not True
                    ):
                        reasons.append(
                            "sentrygraph_utility_failed"
                        )

                    if (
                        sentrygraph.get("false_denial")
                        is True
                    ):
                        reasons.append(
                            "sentrygraph_false_denial"
                        )

                    if (
                        sentrygraph.get(
                            "balanced_lifecycle"
                        )
                        is not True
                    ):
                        reasons.append(
                            "unbalanced_lifecycle"
                        )

                    if (
                        sentrygraph.get(
                            "contains_governs"
                        )
                        is not True
                    ):
                        reasons.append(
                            "missing_governs_relation"
                        )

                entry = {
                    "model": model,
                    "task_id": task_id,
                    "trial": trial,
                    "baseline_utility": (
                        baseline.get("utility")
                        if baseline
                        else None
                    ),
                    "sentrygraph_utility": (
                        sentrygraph.get("utility")
                        if sentrygraph
                        else None
                    ),
                    "sentrygraph_false_denial": (
                        sentrygraph.get(
                            "false_denial"
                        )
                        if sentrygraph
                        else None
                    ),
                    "eligible": not reasons,
                    "exclusion_reasons": reasons,
                }

                if reasons:
                    excluded.append(entry)
                else:
                    qualified.append(entry)

    qualified_keys = {
        (
            entry["model"],
            entry["task_id"],
            entry["trial"],
        )
        for entry in qualified
    }

    eligible_attack_cells = [
        cell
        for cell in plan["cells"]
        if cell["phase"] == "attack"
        and (
            cell["model"],
            cell["task_id"],
            cell["trial"],
        )
        in qualified_keys
    ]

    manifest = {
        "schema_version": "0.2",
        "plan_id": plan_id,
        "qualification_rule": {
            "baseline_utility": True,
            "sentrygraph_utility": True,
            "sentrygraph_false_denial": False,
            "balanced_lifecycle": True,
            "contains_governs": True,
        },
        "qualified": qualified,
        "excluded": excluded,
        "counts": {
            "qualified_groups": len(qualified),
            "excluded_groups": len(excluded),
            "eligible_attack_cells": len(
                eligible_attack_cells
            ),
        },
    }

    output_path = (
        experiment_root
        / "qualification.json"
    )

    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("=== BENIGN QUALIFICATION ===")
    print(f"Plan ID: {plan_id}")
    print(
        "Qualified groups:",
        len(qualified),
    )
    print(
        "Excluded groups:",
        len(excluded),
    )
    print(
        "Eligible attack cells:",
        len(eligible_attack_cells),
    )

    for entry in qualified:
        print(
            "QUALIFIED:",
            entry["model"],
            entry["task_id"],
            f"trial={entry['trial']}",
        )

    for entry in excluded:
        print(
            "EXCLUDED:",
            entry["model"],
            entry["task_id"],
            entry["exclusion_reasons"],
        )

    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
