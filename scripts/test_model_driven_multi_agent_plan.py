"""Audit Phase 3A plan identity, pairing, and source binding."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cells = plan["cells"]
    assert len({cell["cell_id"] for cell in cells}) == len(cells)

    pair_counts: Counter[tuple[tuple[str, object], ...]] = Counter()
    for cell in cells:
        key = tuple(
            sorted((name, value) for name, value in cell.items()
                   if name not in {"cell_id", "condition"})
        )
        pair_counts[key] += 1
    assert pair_counts and set(pair_counts.values()) == {3}

    for key in pair_counts:
        matching = [
            cell for cell in cells
            if tuple(sorted((name, value) for name, value in cell.items()
                            if name not in {"cell_id", "condition"})) == key
        ]
        assert {cell["condition"] for cell in matching} == {
            "baseline",
            "sentrygraph_deterministic",
            "sentrygraph_hybrid",
        }

    identity = plan["identity"]
    assert identity["implementation_fingerprints"]
    assert all(len(value) == 64 for value in
               identity["implementation_fingerprints"].values())
    assert identity["protocol_fingerprint"] == (
        plan["protocol"]["protocol_fingerprint"]
    )
    benign = sum(cell["phase"] == "benign" for cell in cells)
    attacks = sum(cell["phase"] == "attack" for cell in cells)
    print("=== PHASE 3A PLAN AUDIT ===")
    print("Plan ID:", plan["plan_id"])
    print("Unique cells:", len(cells))
    print("Matched pairs:", len(pair_counts))
    print("Benign cells:", benign)
    print("Attack cells:", attacks)
    print("Bound implementation files:", len(
        identity["implementation_fingerprints"]
    ))
    print("PASS: Every cell is unique and exactly matched across three arms.")
    print("PASS: Protocol and implementation fingerprints are frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
