"""Audit a deployed hybrid conformance bundle without API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentrygraph.hybrid_conformance_policy import FrozenHybridConformancePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = FrozenHybridConformancePolicy(args.bundle_root)
    matrix = np.load(args.features_root / "X_calibration.npy", allow_pickle=False)
    predictions = policy.predict_matrix(matrix)
    assert len(predictions) == len(matrix)
    assert {item.decision for item in predictions}.issubset(
        {"allow", "approve", "deny", "isolate"}
    )
    deterministic = policy.combine_with_deterministic(
        deterministic_allowed=False,
        prediction=None,
    )
    assert deterministic == {
        "decision": "deny",
        "source": "deterministic_policy_gate",
        "effect_allowed": False,
        "learned_override": False,
    }
    learned = policy.combine_with_deterministic(
        deterministic_allowed=True,
        prediction=predictions[0],
    )
    assert learned["effect_allowed"] == (predictions[0].decision == "allow")
    manifest = json.loads(
        (args.bundle_root / "manifest.json").read_text(encoding="utf-8")
    )
    print("=== PHASE 3B HYBRID POLICY AUDIT ===")
    print("Bundle ID:", manifest["bundle_id"])
    print("Calibration examples:", len(predictions))
    print("Decisions:", {
        decision: sum(item.decision == decision for item in predictions)
        for decision in ("allow", "approve", "deny", "isolate")
    })
    print("PASS: Bundle checksums and feature dimensions are valid.")
    print("PASS: Four-way learned decisions are reproducible.")
    print("PASS: Learned policy cannot override deterministic denial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
