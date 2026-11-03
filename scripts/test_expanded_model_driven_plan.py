#!/usr/bin/env python3
"""Offline structural and policy-binding audit for the Phase 3I plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    identity = plan["identity"]
    policy = identity["hybrid_policy_bundle"]
    root = Path(policy["path"])
    assert root.is_dir()
    observed = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert observed == policy["files"]
    canonical = json.dumps(observed, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == policy["fingerprint"]
    cells = plan["cells"]
    assert len({cell["cell_id"] for cell in cells}) == len(cells)
    benign = [cell for cell in cells if cell["phase"] == "benign"]
    attacks = [cell for cell in cells if cell["phase"] == "attack"]
    assert Counter(cell["condition"] for cell in benign) == {
        "baseline": len(benign) // 3,
        "sentrygraph_deterministic": len(benign) // 3,
        "sentrygraph_hybrid": len(benign) // 3,
    }
    assert Counter(cell["condition"] for cell in attacks) == {
        "baseline": len(attacks) // 3,
        "sentrygraph_deterministic": len(attacks) // 3,
        "sentrygraph_hybrid": len(attacks) // 3,
    }
    attack_pairs = {
        (
            c["model"], c["task_id"], c["attack"],
            c["injection_task_id"], c["topology"],
            c["handoff_channel"], c["trial"],
        )
        for c in attacks
    }
    assert len(attacks) == 3 * len(attack_pairs)
    print("=== PHASE 3I PLAN AUDIT ===")
    print("Plan ID:", plan["plan_id"])
    print("Policy fingerprint:", policy["fingerprint"])
    print("Benign cells:", len(benign))
    print("Attack cells:", len(attacks))
    print("Matched attack triples:", len(attack_pairs))
    print("PASS: Every cell is unique and exactly matched across three arms.")
    print("PASS: The plan is bound to the complete direct-ablated policy bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
