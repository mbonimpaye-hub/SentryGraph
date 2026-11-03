"""Qualify matched Phase 3 groups only when all benign arms preserve utility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = {"baseline", "sentrygraph_deterministic", "sentrygraph_hybrid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/model_driven_multi_agent_experiments"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.output_root / plan["plan_id"]
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell in plan["cells"]:
        if cell["phase"] != "benign":
            continue
        result_path = root / "runs" / cell["cell_id"] / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            key = (cell["model"], cell["task_id"], cell["topology"], cell["trial"])
            groups[key][cell["condition"]] = result

    qualified: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, arms in sorted(groups.items()):
        complete = set(arms) == CONDITIONS
        utilities = {
            condition: bool(arms.get(condition, {}).get("utility", False))
            for condition in sorted(CONDITIONS)
        }
        entry = {
            "model": key[0],
            "task_id": key[1],
            "topology": key[2],
            "trial": key[3],
            "complete": complete,
            "utilities": utilities,
            "hybrid_denied_actions": arms.get("sentrygraph_hybrid", {}).get("denied_actions"),
            "hybrid_learned_decisions": arms.get("sentrygraph_hybrid", {}).get("learned_decisions", []),
        }
        (qualified if complete and all(utilities.values()) else excluded).append(entry)

    eligible_attack_cells = 0
    qualified_keys = {
        (item["model"], item["task_id"], item["topology"], item["trial"])
        for item in qualified
    }
    for cell in plan["cells"]:
        if cell["phase"] != "attack":
            continue
        key = (cell["model"], cell["task_id"], cell["topology"], cell["trial"])
        eligible_attack_cells += int(key in qualified_keys)

    manifest = {
        "schema_version": "0.1",
        "plan_id": plan["plan_id"],
        "qualification_rule": "all three matched benign arms must preserve utility",
        "qualified_groups": qualified,
        "excluded_groups": excluded,
        "eligible_attack_cells": eligible_attack_cells,
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "qualification.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("=== PHASE 3C BENIGN QUALIFICATION ===")
    print("Plan ID:", plan["plan_id"])
    print("Qualified groups:", len(qualified))
    print("Excluded groups:", len(excluded))
    print("Eligible attack cells:", eligible_attack_cells)
    for item in qualified:
        print("QUALIFIED:", item)
    for item in excluded:
        print("EXCLUDED:", item)
    print("Manifest:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

