"""Train leakage-controlled baselines for graph-conformance prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


MODEL_ORDER = ("dummy", "logistic_regression", "random_forest", "mlp")
DEFAULT_SEEDS = (13, 42, 73, 101, 137)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not seeds:
        raise ValueError("At least one random seed is required.")
    return seeds


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def metric_bundle(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "examples": int(len(y_true)),
        "violations": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision_violation": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall_violation": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_violation": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": {
            "true_allow_pred_allow": int(tn),
            "true_allow_pred_violation": int(fp),
            "true_violation_pred_allow": int(fn),
            "true_violation_pred_violation": int(tp),
        },
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
        result["average_precision"] = float(
            average_precision_score(y_true, probability)
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def model_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "dummy":
        return [{"strategy": "prior"}]
    if model_name == "logistic_regression":
        return [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    if model_name == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (None, 4, 8)
            for leaf in (1, 3)
        ]
    if model_name == "mlp":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden in ((32,), (64, 32))
            for alpha in (0.0001, 0.01)
        ]
    raise ValueError(f"Unknown model: {model_name}")


def build_model(
    model_name: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> BaseEstimator:
    if model_name == "dummy":
        return DummyClassifier(strategy=str(parameters["strategy"]))

    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )

    if model_name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
                        activation="relu",
                        solver="lbfgs",
                        alpha=float(parameters["alpha"]),
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model_name: str,
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    if model_name == "mlp":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)


def violation_probability(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError(f"Model classes do not contain violation class 1: {classes}")
    return probabilities[:, int(matches[0])]


def selection_score(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["f1_violation"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["average_precision"] or 0.0),
    )


def select_configuration(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []

    for parameters in model_grid(model_name):
        seed_metrics: list[dict[str, Any]] = []
        for seed in seeds:
            model = build_model(model_name, parameters, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fit_model(model_name, model, X_train, y_train)
            probability = violation_probability(model, X_validation)
            seed_metrics.append(metric_bundle(y_validation, probability))

        mean_metrics = {
            name: float(np.mean([item[name] for item in seed_metrics]))
            for name in (
                "accuracy",
                "balanced_accuracy",
                "precision_violation",
                "recall_violation",
                "f1_violation",
                "specificity",
                "roc_auc",
                "average_precision",
            )
        }
        candidates.append(
            {
                "parameters": parameters,
                "mean_validation_metrics": mean_metrics,
                "seed_validation_metrics": seed_metrics,
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            selection_score(item["mean_validation_metrics"]),
            canonical_json(item["parameters"]),
        ),
    )
    return dict(best["parameters"]), candidates


def load_arrays(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "calibration", "test"):
        arrays[f"X_{split}"] = np.load(root / f"X_{split}.npy", allow_pickle=False)
        permitted = np.load(
            root / f"y_action_{split}.npy", allow_pickle=False
        ).astype(np.int64)
        arrays[f"y_{split}"] = 1 - permitted
    return arrays


def write_predictions(
    path: Path,
    metadata: list[Mapping[str, Any]],
    y_true: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    fieldnames = [
        "example_id",
        "group_id",
        "task_id",
        "attack",
        "injection_task_id",
        "trial",
        "step",
        "true_violation",
    ] + [f"{model}_violation_probability" for model in MODEL_ORDER]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(metadata):
            source = item["source"]
            row = {
                "example_id": item["example_id"],
                "group_id": item["group_id"],
                "task_id": source.get("task_id"),
                "attack": source.get("attack"),
                "injection_task_id": source.get("injection_task_id"),
                "trial": source.get("trial"),
                "step": source.get("step"),
                "true_violation": int(y_true[index]),
            }
            for model_name in MODEL_ORDER:
                row[f"{model_name}_violation_probability"] = float(
                    probabilities[model_name][index]
                )
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    feature_manifest = json.loads(
        (args.features_root / "manifest.json").read_text(encoding="utf-8")
    )
    arrays = load_arrays(args.features_root)
    X_train = arrays["X_train"]
    y_train = arrays["y_train"]
    X_validation = arrays["X_validation"]
    y_validation = arrays["y_validation"]
    X_refit = np.concatenate([X_train, X_validation], axis=0)
    y_refit = np.concatenate([y_train, y_validation], axis=0)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )
    models_directory = temporary / "models"
    models_directory.mkdir()

    selection: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    probabilities_by_split: dict[str, dict[str, np.ndarray]] = {
        "calibration": {},
        "test": {},
    }

    try:
        for model_name in MODEL_ORDER:
            selected_parameters, candidates = select_configuration(
                model_name,
                X_train,
                y_train,
                X_validation,
                y_validation,
                seeds,
            )
            selection[model_name] = {
                "selection_split": "validation",
                "selection_metric_order": [
                    "f1_violation",
                    "balanced_accuracy",
                    "average_precision",
                ],
                "selected_parameters": selected_parameters,
                "candidates": candidates,
            }

            seed_probabilities: dict[str, list[np.ndarray]] = {
                "calibration": [],
                "test": [],
            }
            for seed in seeds:
                model = build_model(model_name, selected_parameters, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    fit_model(model_name, model, X_refit, y_refit)
                joblib.dump(
                    model,
                    models_directory / f"{model_name}_seed{seed}.joblib",
                    compress=3,
                )
                for split in ("calibration", "test"):
                    seed_probabilities[split].append(
                        violation_probability(model, arrays[f"X_{split}"])
                    )

            final_metrics[model_name] = {}
            for split in ("calibration", "test"):
                stacked = np.stack(seed_probabilities[split], axis=0)
                ensemble_probability = stacked.mean(axis=0)
                probabilities_by_split[split][model_name] = ensemble_probability
                final_metrics[model_name][split] = {
                    "ensemble": metric_bundle(
                        arrays[f"y_{split}"], ensemble_probability
                    ),
                    "per_seed": [
                        metric_bundle(arrays[f"y_{split}"], probability)
                        for probability in seed_probabilities[split]
                    ],
                }

        for split in ("calibration", "test"):
            metadata = read_jsonl(
                args.features_root / f"metadata_{split}.jsonl"
            )
            write_predictions(
                temporary / f"predictions_{split}.csv",
                metadata,
                arrays[f"y_{split}"],
                probabilities_by_split[split],
            )

        results = {
            "schema_version": "0.1",
            "task": "action_violation_detection",
            "positive_class": "violation",
            "source_dataset_id": feature_manifest["source_dataset_id"],
            "feature_encoder": feature_manifest["encoder"],
            "feature_count": feature_manifest["feature_count"],
            "seeds": list(seeds),
            "protocol": {
                "hyperparameter_selection": "train -> validation",
                "final_refit": "train + validation",
                "calibration_used_for_selection": False,
                "test_used_for_selection": False,
                "ensemble": "mean probability across fixed seeds",
                "decision_threshold": 0.5,
            },
            "versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "selection": selection,
            "final_metrics": final_metrics,
        }
        write_json(temporary / "results.json", results)

        summary = {
            model_name: {
                split: final_metrics[model_name][split]["ensemble"]
                for split in ("calibration", "test")
            }
            for model_name in MODEL_ORDER
        }
        write_json(temporary / "summary.json", summary)

        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary)}" for path in files
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 1E LEARNED BASELINES ===")
    print("Output:", args.output_dir)
    print("Seeds:", list(seeds))
    for model_name in MODEL_ORDER:
        selected = selection[model_name]["selected_parameters"]
        metrics = final_metrics[model_name]["test"]["ensemble"]
        print()
        print(model_name)
        print("  selected:", selected)
        print("  test balanced accuracy:", f"{metrics['balanced_accuracy']:.4f}")
        print("  test violation F1:", f"{metrics['f1_violation']:.4f}")
        print("  test violation recall:", f"{metrics['recall_violation']:.4f}")
        print("  test ROC-AUC:", f"{metrics['roc_auc']:.4f}")
        print("  test average precision:", f"{metrics['average_precision']:.4f}")
        print("  confusion:", metrics["confusion_matrix"])
    print()
    print("PASS: Test and calibration splits were excluded from model selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
