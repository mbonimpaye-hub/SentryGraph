#!/usr/bin/env python3
"""Verify exact isolation of the Phase 3H direct-invariant ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROLES = ("train", "validation", "calibration", "test")
FEATURE = "candidate.invariant_contract_conformant"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.ablation_root / "manifest.json").read_text())
    assert manifest["removed_feature"] == FEATURE
    print("=== PHASE 3H ISOLATION AUDIT ===")
    print("Ablation encoding:", manifest["encoding_id"])
    for fold in manifest["folds"]:
        name = fold["fold"]
        source = args.source_features_root / "folds" / name / "invariant_relational"
        ablated = args.ablation_root / "folds" / name / "direct_invariant_ablated"
        source_names = json.loads((source / "feature_names.json").read_text())
        ablated_names = json.loads((ablated / "feature_names.json").read_text())
        assert source_names.count(FEATURE) == 1
        index = source_names.index(FEATURE)
        assert ablated_names == source_names[:index] + source_names[index + 1 :]
        for role in ROLES:
            original = np.load(source / f"X_{role}.npy", allow_pickle=False)
            expected = np.delete(original, index, axis=1)
            observed = np.load(ablated / f"X_{role}.npy", allow_pickle=False)
            assert np.array_equal(expected, observed)
            for label in ("action", "trajectory"):
                left = np.load(source / f"y_{label}_{role}.npy", allow_pickle=False)
                right = np.load(ablated / f"y_{label}_{role}.npy", allow_pickle=False)
                assert np.array_equal(left, right)
            assert (source / f"metadata_{role}.jsonl").read_bytes() == (
                ablated / f"metadata_{role}.jsonl"
            ).read_bytes()
        print(f"  {name}: PASS ({len(source_names)} -> {len(ablated_names)} features)")
    print("PASS: Every ablated matrix is an exact one-column projection.")
    print("PASS: Labels, metadata, folds, and example order are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
