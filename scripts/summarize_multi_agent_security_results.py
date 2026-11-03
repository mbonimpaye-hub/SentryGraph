"""Summarize matched Phase 2F multi-agent security outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.experiment_root / "results.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique = {record["cell"]["cell_id"]: record for record in records}
    completed = [r for r in unique.values() if r["status"] == "completed"]
    errors = [r for r in unique.values() if r["status"] == "error"]
    pairs: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for record in completed:
        cell = record["cell"]
        pairs[(cell["scenario_id"], cell["trial"])][
            cell["condition"]
        ] = record

    complete_pairs = [value for value in pairs.values() if len(value) == 2]
    attack_pairs = [
        pair
        for pair in complete_pairs
        if pair["baseline"]["scenario"]["category"] == "attack"
    ]
    prevented = sum(
        pair["baseline"]["attack_success"]
        and not pair["sentrygraph"]["attack_success"]
        for pair in attack_pairs
    )
    regressions = sum(
        not pair["baseline"]["attack_success"]
        and pair["sentrygraph"]["attack_success"]
        for pair in attack_pairs
    )
    outcome_mismatches = sum(not r.get("outcome_match") for r in completed)
    by_family = Counter(
        (
            pair["baseline"]["scenario"]["family"],
            "prevented"
            if pair["baseline"]["attack_success"]
            and not pair["sentrygraph"]["attack_success"]
            else "other",
        )
        for pair in attack_pairs
    )
    summary = {
        "schema_version": "0.1",
        "unique_records": len(unique),
        "completed": len(completed),
        "errors": len(errors),
        "complete_pairs": len(complete_pairs),
        "attack_pairs": len(attack_pairs),
        "baseline_attack_successes": sum(
            pair["baseline"]["attack_success"] for pair in attack_pairs
        ),
        "sentrygraph_attack_successes": sum(
            pair["sentrygraph"]["attack_success"] for pair in attack_pairs
        ),
        "prevented_attacks": prevented,
        "security_regressions": regressions,
        "outcome_mismatches": outcome_mismatches,
        "by_family": {
            f"{family}:{outcome}": count
            for (family, outcome), count in sorted(by_family.items())
        },
    }
    output = args.experiment_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("=== MULTI-AGENT SECURITY SUMMARY ===")
    for key, value in summary.items():
        if key != "by_family":
            print(f"{key}: {value}")
    print("Summary:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
