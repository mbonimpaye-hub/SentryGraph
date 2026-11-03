"""Offline audit of Phase 3G qualification and matched attack cells."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    qualification = json.loads(args.qualification.read_text())
    assert qualification["plan_id"] == plan["plan_id"]
    keys = {
        (x["model"], x["task_id"], x["topology"], x["trial"])
        for x in qualification["qualified_groups"]
    }
    cells = [
        c
        for c in plan["cells"]
        if c["phase"] == "attack"
        and (c["model"], c["task_id"], c["topology"], c["trial"]) in keys
    ]
    assert len(cells) == qualification["eligible_attack_cells"]
    counts = Counter(c["condition"] for c in cells)
    assert counts == {
        "baseline": len(cells) // 3,
        "sentrygraph_deterministic": len(cells) // 3,
        "sentrygraph_hybrid": len(cells) // 3,
    }
    assert len({c["cell_id"] for c in cells}) == len(cells)
    print("=== PHASE 3G QUALIFICATION AUDIT ===")
    print("Plan ID:", plan["plan_id"])
    print("Qualified groups:", len(keys))
    print("Eligible attack cells:", len(cells))
    print("Conditions:", dict(sorted(counts.items())))
    print("PASS: Only benign-qualified, exactly matched attack cells are eligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
