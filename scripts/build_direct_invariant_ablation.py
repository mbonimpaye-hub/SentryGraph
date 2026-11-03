#!/usr/bin/env python3
"""Create an exact one-feature ablation of the invariant-relational view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROLES = ("train", "validation", "calibration", "test")
SOURCE_VIEW = "invariant_relational"
ABLATION_VIEW = "direct_invariant_ablated"
REMOVED_FEATURE = "candidate.invariant_contract_conformant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def overlap(matrices: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left_index, left in enumerate(ROLES):
        left_map: dict[str, set[int]] = {}
        for row, label in zip(matrices[left], labels[left], strict=True):
            left_map.setdefault(row_digest(row), set()).add(int(label))
        for right in ROLES[left_index + 1:]:
            right_map: dict[str, set[int]] = {}
            for row, label in zip(matrices[right], labels[right], strict=True):
                right_map.setdefault(row_digest(row), set()).add(int(label))
            shared = set(left_map) & set(right_map)
            result[f"{left}__{right}"] = {
                "shared_unique_vectors": len(shared),
                "conflicting_label_vectors": sum(
                    len(left_map[item] | right_map[item]) > 1 for item in shared
                ),
                "left_rows_matching_right": sum(
                    row_digest(row) in right_map for row in matrices[left]
                ),
                "right_rows_matching_left": sum(
                    row_digest(row) in left_map for row in matrices[right]
                ),
            }
    return result


def copy_source_view(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def build_ablation(source: Path, destination: Path) -> dict[str, Any]:
    names = read_json(source / "feature_names.json")
    if names.count(REMOVED_FEATURE) != 1:
        raise RuntimeError(
            f"Expected exactly one {REMOVED_FEATURE!r}; found {names.count(REMOVED_FEATURE)}"
        )
    removed_index = names.index(REMOVED_FEATURE)
    retained_names = names[:removed_index] + names[removed_index + 1 :]
    destination.mkdir(parents=True)
    write_json(destination / "feature_names.json", retained_names)
    write_json(
        destination / "vocabulary.json",
        {name: index for index, name in enumerate(retained_names)},
    )

    matrices: dict[str, np.ndarray] = {}
    action_labels: dict[str, np.ndarray] = {}
    split_summary: dict[str, Any] = {}
    for role in ROLES:
        original = np.load(source / f"X_{role}.npy", allow_pickle=False)
        reduced = np.delete(original, removed_index, axis=1).astype(
            np.float32, copy=False
        )
        matrices[role] = reduced
        np.save(destination / f"X_{role}.npy", reduced, allow_pickle=False)
        for label_name in ("action", "trajectory"):
            filename = f"y_{label_name}_{role}.npy"
            shutil.copy2(source / filename, destination / filename)
        action = np.load(
            source / f"y_action_{role}.npy", allow_pickle=False
        )
        action_labels[role] = action
        shutil.copy2(
            source / f"metadata_{role}.jsonl",
            destination / f"metadata_{role}.jsonl",
        )
        nonzero = int(np.count_nonzero(reduced))
        split_summary[role] = {
            "examples": int(reduced.shape[0]),
            "features": int(reduced.shape[1]),
            "nonzero_values": nonzero,
            "density": float(nonzero / reduced.size),
            "action_allow": int(action.sum()),
            "action_deny": int((1 - action).sum()),
        }

    manifest = {
        "view": ABLATION_VIEW,
        "source_view": SOURCE_VIEW,
        "feature_count": len(retained_names),
        "fit_split": "train inherited from source encoding",
        "ablation_type": "exact_column_removal",
        "removed_features": [REMOVED_FEATURE],
        "labels_observed_for_ablation": False,
        "matrices_refitted": False,
        "split_summary": split_summary,
        "encoded_vector_overlap": overlap(matrices, action_labels),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    source_manifest = read_json(args.features_root / "manifest.json")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent
        )
    )
    fold_results = []
    try:
        for fold in source_manifest["folds"]:
            fold_name = fold["fold"]
            source = args.features_root / "folds" / fold_name / SOURCE_VIEW
            if not source.is_dir():
                raise FileNotFoundError(source)
            source_destination = temporary / "folds" / fold_name / SOURCE_VIEW
            ablation_destination = temporary / "folds" / fold_name / ABLATION_VIEW
            copy_source_view(source, source_destination)
            ablation = build_ablation(source, ablation_destination)
            source_view_manifest = read_json(source / "manifest.json")
            fold_manifest = {
                "fold": fold_name,
                "assignment": fold["assignment"],
                "test_evaluation": fold["test_evaluation"],
                "views": [source_view_manifest, ablation],
            }
            write_json(
                temporary / "folds" / fold_name / "manifest.json",
                fold_manifest,
            )
            fold_results.append(fold_manifest)

        manifest = {
            **{
                key: value
                for key, value in source_manifest.items()
                if key not in {"encoding_id", "encoder", "views", "folds"}
            },
            "schema_version": "0.1",
            "encoder": "DirectInvariantAblation/0.1",
            "source_encoding_id": source_manifest["encoding_id"],
            "source_features_root": str(args.features_root),
            "views": [SOURCE_VIEW, ABLATION_VIEW],
            "removed_feature": REMOVED_FEATURE,
            "isolation_rule": (
                "The ablated matrices equal the source invariant matrices "
                "with exactly one named column removed."
            ),
            "folds": fold_results,
        }
        manifest["encoding_id"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        write_json(temporary / "manifest.json", manifest)
        payloads = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary)}"
                for path in payloads
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 3H DIRECT-INVARIANT ABLATION ===")
    print("Source encoding:", source_manifest["encoding_id"])
    print("Ablation encoding:", manifest["encoding_id"])
    print("Removed feature:", REMOVED_FEATURE)
    print("Output:", args.output_dir)
    for fold in fold_results:
        ablation = fold["views"][1]
        collision = ablation["encoded_vector_overlap"]["train__test"]
        print(
            f"  {fold['fold']}: features={ablation['feature_count']}, "
            f"shared={collision['shared_unique_vectors']}, "
            f"conflicting={collision['conflicting_label_vectors']}"
        )
    print("PASS: Exactly one direct conformance feature was removed.")
    print("PASS: Labels and test outcomes were not used to construct the ablation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
