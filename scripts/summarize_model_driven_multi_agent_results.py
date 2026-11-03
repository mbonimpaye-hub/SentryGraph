"""Summarize complete three-arm Phase 3 model-driven attack pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = {
    "baseline",
    "sentrygraph_deterministic",
    "sentrygraph_hybrid",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted((args.experiment_root / "runs").glob("*/result.json")):
        item = json.loads(path.read_text())
        if item.get("cell", {}).get("phase") == "attack":
            records.append(item)
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    errors = 0
    for item in records:
        errors += int(item.get("status") == "error")
        groups[item.get("pair_id", "missing")][item["cell"]["condition"]] = item
    complete = [arms for arms in groups.values() if set(arms) == CONDITIONS]
    summary = {
        "schema_version": "0.1",
        "unique_attack_records": len(records),
        "completed_attack_records": sum(r.get("status") == "completed" for r in records),
        "errors": errors,
        "complete_three_arm_pairs": len(complete),
        "baseline_attack_successes": sum(
            bool(a["baseline"].get("attack_success")) for a in complete
        ),
        "deterministic_attack_successes": sum(
            bool(a["sentrygraph_deterministic"].get("attack_success"))
            for a in complete
        ),
        "hybrid_attack_successes": sum(
            bool(a["sentrygraph_hybrid"].get("attack_success")) for a in complete
        ),
        "hybrid_utility_preserved": sum(
            bool(a["sentrygraph_hybrid"].get("utility")) for a in complete
        ),
    }
    path = args.experiment_root / "model_driven_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("=== PHASE 3G MODEL-DRIVEN SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Summary:", path)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
