#!/usr/bin/env python3
"""Verify exact isolation of the structure/provenance-only ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROLES = ("train", "validation", "calibration", "test")
EXACT_RETAINED = frozenset({
    "candidate.argument_count",
    "candidate.null_argument_count",
})
RETAINED_PREFIXES = ("contract.node_type::", "contract.relation::", "prefix.")
REQUIRED_REMOVALS = frozenset({
    "candidate.constraint_satisfied_count",
    "candidate.constraint_violated_count",
    "candidate.unexpected_non_null_argument_count",
    "candidate.invariant_contract_conformant",
})


def retained(name: str) -> bool:
    return name in EXACT_RETAINED or name.startswith(RETAINED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.ablation_root / "manifest.json").read_text())
    assert manifest["views"] == ["invariant_relational", "structure_provenance_only"]
    print("=== PHASE 4B STRUCTURE-PROVENANCE ISOLATION AUDIT ===")
    print("Ablation encoding:", manifest["encoding_id"])
    for fold in manifest["folds"]:
        name = fold["fold"]
        source = args.source_features_root / "folds" / name / "invariant_relational"
        ablated = args.ablation_root / "folds" / name / "structure_provenance_only"
        source_names = json.loads((source / "feature_names.json").read_text())
        ablated_names = json.loads((ablated / "feature_names.json").read_text())
        indices = [index for index, feature in enumerate(source_names) if retained(feature)]
        expected_names = [source_names[index] for index in indices]
        assert ablated_names == expected_names
        assert REQUIRED_REMOVALS <= set(source_names)
        assert REQUIRED_REMOVALS.isdisjoint(ablated_names)
        assert all(retained(feature) for feature in ablated_names)
        for role in ROLES:
            original = np.load(source / f"X_{role}.npy", allow_pickle=False)
            expected = original[:, indices]
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
    print("PASS: Every ablated matrix is an exact structure/provenance projection.")
    print("PASS: Direct authorization, constraint, and conformance summaries are absent.")
    print("PASS: Labels, metadata, folds, and example order are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
