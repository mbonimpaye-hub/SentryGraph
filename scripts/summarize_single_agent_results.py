"""Aggregate matched single-agent experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records_by_cell: dict[str, dict[str, Any]] = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        records_by_cell[record["cell_id"]] = record

    return list(records_by_cell.values())


def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def paired_outcome(
    baseline_attack_success: bool,
    sentry_attack_success: bool,
) -> str:
    if baseline_attack_success and not sentry_attack_success:
        return "prevented"

    if baseline_attack_success and sentry_attack_success:
        return "both_succeeded"

    if (
        not baseline_attack_success
        and not sentry_attack_success
    ):
        return "both_resisted"

    return "regression"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate benign and matched attack results."
        )
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    results_path = (
        args.experiment_root
        / "results.jsonl"
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found: {results_path}"
        )

    records = load_records(results_path)

    completed = [
        record
        for record in records
        if record.get("status") == "completed"
    ]

    errors = [
        record
        for record in records
        if record.get("status") == "error"
    ]

    benign = [
        record
        for record in completed
        if record["cell"]["phase"] == "benign"
    ]

    attacks = [
        record
        for record in completed
        if record["cell"]["phase"] == "attack"
    ]

    attack_pairs: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for record in attacks:
        pair_id = record["pair_id"]
        condition = record["condition"]

        attack_pairs.setdefault(
            pair_id,
            {},
        )[condition] = record

    complete_pairs = {
        pair_id: conditions
        for pair_id, conditions in attack_pairs.items()
        if set(conditions) == {
            "baseline",
            "sentrygraph",
        }
    }

    incomplete_pairs = {
        pair_id: sorted(conditions)
        for pair_id, conditions in attack_pairs.items()
        if set(conditions) != {
            "baseline",
            "sentrygraph",
        }
    }

    pair_rows = []

    for pair_id, conditions in complete_pairs.items():
        baseline = conditions["baseline"]
        sentry = conditions["sentrygraph"]

        baseline_attack_success = bool(
            baseline["attack_success"]
        )

        sentry_attack_success = bool(
            sentry["attack_success"]
        )

        outcome = paired_outcome(
            baseline_attack_success,
            sentry_attack_success,
        )

        pair_rows.append(
            {
                "pair_id": pair_id,
                "model": baseline["model"],
                "task_id": baseline["task_id"],
                "attack": baseline["attack"],
                "injection_task_id": baseline[
                    "injection_task_id"
                ],
                "trial": baseline["cell"]["trial"],
                "injection_fingerprint": baseline[
                    "injection_fingerprint"
                ],
                "baseline_utility": baseline["utility"],
                "sentrygraph_utility": sentry["utility"],
                "baseline_attack_success": (
                    baseline_attack_success
                ),
                "sentrygraph_attack_success": (
                    sentry_attack_success
                ),
                "sentrygraph_denied_actions": sentry.get(
                    "denied_actions",
                    0,
                ),
                "sentrygraph_denied_tools": sentry.get(
                    "denied_tools",
                    [],
                ),
                "sentrygraph_contains_blocks": sentry.get(
                    "contains_blocks",
                    False,
                ),
                "sentrygraph_contains_governs": sentry.get(
                    "contains_governs",
                    False,
                ),
                "baseline_duration_seconds": baseline.get(
                    "duration_seconds"
                ),
                "sentrygraph_duration_seconds": sentry.get(
                    "duration_seconds"
                ),
                "sentrygraph_tool_latency_p50_ms": (
                    sentry.get("tool_latency_p50_ms")
                ),
                "sentrygraph_tool_latency_p95_ms": (
                    sentry.get("tool_latency_p95_ms")
                ),
                "outcome": outcome,
            }
        )

    pair_rows.sort(
        key=lambda row: (
            row["model"],
            row["task_id"],
            row["attack"],
            row["injection_task_id"],
            row["trial"],
        )
    )

    outcome_counts = Counter(
        row["outcome"]
        for row in pair_rows
    )

    baseline_successes = sum(
        row["baseline_attack_success"]
        for row in pair_rows
    )

    sentry_successes = sum(
        row["sentrygraph_attack_success"]
        for row in pair_rows
    )

    prevented = outcome_counts["prevented"]

    benign_baseline = [
        record
        for record in benign
        if record["condition"] == "baseline"
    ]

    benign_sentry = [
        record
        for record in benign
        if record["condition"] == "sentrygraph"
    ]

    benign_false_denials = sum(
        record.get("false_denial") is True
        for record in benign_sentry
    )

    sentry_p95_values = [
        row["sentrygraph_tool_latency_p95_ms"]
        for row in pair_rows
        if row["sentrygraph_tool_latency_p95_ms"]
        is not None
    ]

    duration_differences = [
        row["sentrygraph_duration_seconds"]
        - row["baseline_duration_seconds"]
        for row in pair_rows
        if row["sentrygraph_duration_seconds"]
        is not None
        and row["baseline_duration_seconds"]
        is not None
    ]

    summary = {
        "schema_version": "0.1",
        "experiment_root": str(
            args.experiment_root
        ),
        "records": {
            "unique_records": len(records),
            "completed": len(completed),
            "errors": len(errors),
            "benign_records": len(benign),
            "attack_records": len(attacks),
        },
        "qualification": {
            "benign_baseline_runs": len(
                benign_baseline
            ),
            "benign_sentrygraph_runs": len(
                benign_sentry
            ),
            "baseline_utility_passes": sum(
                record.get("utility") is True
                for record in benign_baseline
            ),
            "sentrygraph_utility_passes": sum(
                record.get("utility") is True
                for record in benign_sentry
            ),
            "sentrygraph_false_denials": (
                benign_false_denials
            ),
        },
        "matched_attacks": {
            "complete_pairs": len(pair_rows),
            "incomplete_pairs": incomplete_pairs,
            "baseline_attack_successes": (
                baseline_successes
            ),
            "sentrygraph_attack_successes": (
                sentry_successes
            ),
            "baseline_attack_success_rate": safe_rate(
                baseline_successes,
                len(pair_rows),
            ),
            "sentrygraph_attack_success_rate": safe_rate(
                sentry_successes,
                len(pair_rows),
            ),
            "prevented_attacks": prevented,
            "conditional_prevention_rate": safe_rate(
                prevented,
                baseline_successes,
            ),
            "outcomes": dict(outcome_counts),
            "sentrygraph_pairs_with_denial": sum(
                row["sentrygraph_denied_actions"] > 0
                for row in pair_rows
            ),
            "sentrygraph_pairs_with_blocks": sum(
                row["sentrygraph_contains_blocks"]
                is True
                for row in pair_rows
            ),
        },
        "performance": {
            "median_sentrygraph_run_p95_tool_latency_ms": (
                median(sentry_p95_values)
                if sentry_p95_values
                else None
            ),
            "mean_sentrygraph_run_p95_tool_latency_ms": (
                mean(sentry_p95_values)
                if sentry_p95_values
                else None
            ),
            "median_matched_duration_difference_seconds": (
                median(duration_differences)
                if duration_differences
                else None
            ),
        },
    }

    summary_path = (
        args.experiment_root
        / "summary.json"
    )

    pairs_path = (
        args.experiment_root
        / "paired_attack_results.csv"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    if pair_rows:
        with pairs_path.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    pair_rows[0].keys()
                ),
            )

            writer.writeheader()

            for row in pair_rows:
                serializable = dict(row)
                serializable[
                    "sentrygraph_denied_tools"
                ] = json.dumps(
                    row["sentrygraph_denied_tools"]
                )

                writer.writerow(serializable)

    print("=== SINGLE-AGENT RESULT SUMMARY ===")
    print(f"Unique records: {len(records)}")
    print(f"Completed: {len(completed)}")
    print(f"Errors: {len(errors)}")
    print(f"Complete attack pairs: {len(pair_rows)}")
    print(
        "Baseline attack successes:",
        baseline_successes,
    )
    print(
        "SentryGraph attack successes:",
        sentry_successes,
    )
    print(
        "Prevented attacks:",
        prevented,
    )
    print(
        "Outcome counts:",
        dict(outcome_counts),
    )
    print(
        "Benign false denials:",
        benign_false_denials,
    )
    print(f"Summary: {summary_path}")
    print(f"Paired results: {pairs_path}")


if __name__ == "__main__":
    main()
