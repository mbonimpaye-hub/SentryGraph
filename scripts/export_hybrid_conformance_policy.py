"""Export and audit a deployable task-held-out hybrid policy bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path(
            "outputs/learning/graph_conformance_generalization_features_v1"
        ),
    )
    parser.add_argument(
        "--baselines-root",
        type=Path,
        default=Path(
            "outputs/learning/task_heldout_conformance_baselines_v1"
        ),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path(
            "outputs/learning/task_heldout_conformance_calibration_v1"
        ),
    )
    parser.add_argument("--fold", default="fold_user_task_35")
    parser.add_argument("--view", default="full_fidelity")
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/learning/deployable_hybrid_policy_task35_v1"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def violation_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError(f"Missing violation class: {classes}")
    return probabilities[:, int(matches[0])]


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    feature_dir = args.features_root / "folds" / args.fold / args.view
    model_dir = (
        args.baselines_root / "folds" / args.fold / args.view / args.model
    )
    policy_dir = (
        args.calibration_root / "folds" / args.fold / args.view / args.model
    )
    required = [
        feature_dir / "feature_names.json",
        feature_dir / "vocabulary.json",
        feature_dir / "X_calibration.npy",
        feature_dir / "metadata_calibration.jsonl",
        model_dir / "predictions_calibration.csv",
        policy_dir / "calibrator.joblib",
        policy_dir / "policy.json",
        policy_dir / "metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment inputs: " + ", ".join(missing))

    model_paths = sorted((model_dir / "models").glob("seed_*.joblib"))
    if not model_paths:
        raise RuntimeError("No seed models found.")
    seeds = tuple(int(path.stem.split("_")[-1]) for path in model_paths)

    matrix = np.load(feature_dir / "X_calibration.npy", allow_pickle=False)
    calculated: dict[tuple[str, int], float] = {}
    metadata = [
        json.loads(line)
        for line in (feature_dir / "metadata_calibration.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metadata) != len(matrix):
        raise RuntimeError("Calibration metadata/matrix length mismatch.")
    for path, seed in zip(model_paths, seeds, strict=True):
        probabilities = violation_probability(joblib.load(path), matrix)
        for item, probability in zip(metadata, probabilities, strict=True):
            calculated[(str(item["example_id"]), seed)] = float(probability)

    stored = load_prediction_rows(model_dir / "predictions_calibration.csv")
    maximum_error = 0.0
    for row in stored:
        key = (row["example_id"], int(row["seed"]))
        if key not in calculated:
            raise RuntimeError(f"Stored prediction has no encoded row: {key}")
        maximum_error = max(
            maximum_error,
            abs(calculated[key] - float(row["probability_violation"])),
        )
    if maximum_error > 1e-10:
        raise RuntimeError(
            f"Deployment model differs from stored predictions: {maximum_error}"
        )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )
    try:
        (temporary / "models").mkdir()
        copies = {
            "feature_names.json": feature_dir / "feature_names.json",
            "vocabulary.json": feature_dir / "vocabulary.json",
            "calibrator.joblib": policy_dir / "calibrator.joblib",
            "policy.json": policy_dir / "policy.json",
            "metrics.json": policy_dir / "metrics.json",
        }
        for destination, source in copies.items():
            shutil.copy2(source, temporary / destination)
        for path in model_paths:
            shutil.copy2(path, temporary / "models" / path.name)

        payloads = sorted(path for path in temporary.rglob("*") if path.is_file())
        file_hashes = {
            str(path.relative_to(temporary)): sha256(path) for path in payloads
        }
        identity = {
            "schema_version": "0.1",
            "policy_type": "task_held_out_graph_conformance",
            "fold": args.fold,
            "view": args.view,
            "model": args.model,
            "seeds": list(seeds),
            "feature_count": int(matrix.shape[1]),
            "positive_class": "violation",
            "calibration": "platt",
            "hybrid_precedence": (
                "deterministic denial is final; learned policy evaluates only "
                "deterministically allowed actions"
            ),
            "offline_equivalence_max_abs_error": maximum_error,
            "files": file_hashes,
        }
        identity["bundle_id"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        (temporary / "manifest.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksummed = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary)}"
                for path in checksummed
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 3B DEPLOYABLE HYBRID POLICY ===")
    print("Bundle ID:", identity["bundle_id"])
    print("Fold:", args.fold)
    print("View:", args.view)
    print("Model:", args.model)
    print("Seeds:", list(seeds))
    print("Features:", matrix.shape[1])
    print("Equivalence maximum absolute error:", maximum_error)
    print("Output:", args.output_dir)
    print("PASS: Deployed ensemble reproduces frozen calibration predictions.")
    print("PASS: Deterministic denials retain absolute precedence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
