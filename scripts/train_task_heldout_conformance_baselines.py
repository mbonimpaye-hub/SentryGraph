#!/usr/bin/env python3
"""Train classical baselines on leakage-resistant SentryGraph task folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROLES = ("train", "validation", "calibration", "test")
VIEWS = (
    "full_fidelity",
    "structural_generalization",
    "semantic_ablated",
    "invariant_relational",
)
MODELS = ("dummy", "logistic_regression", "random_forest", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train task-held-out graph conformance baselines."
    )
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="13,42,73,101,137")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    seeds = list(dict.fromkeys(int(item.strip()) for item in text.split(",")))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def violation_labels(allow_labels: np.ndarray) -> np.ndarray:
    return 1 - np.asarray(allow_labels, dtype=np.int64)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(np.int64)
    labels_present = sorted(np.unique(y_true).tolist())
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "examples": int(len(y_true)),
        "classes_present": labels_present,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predicted))
            if len(labels_present) == 2 else None
        ),
        "precision_violation": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall_violation": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1_violation": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity_allow": float(tn / (tn + fp)) if (tn + fp) else None,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else None,
        "roc_auc": (
            float(roc_auc_score(y_true, probability))
            if len(labels_present) == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, probability))
            if len(labels_present) == 2 else None
        ),
        "confusion_matrix": {
            "true_allow_pred_allow": int(tn),
            "true_allow_pred_violation": int(fp),
            "true_violation_pred_allow": int(fn),
            "true_violation_pred_violation": int(tp),
        },
    }
    return result


def probability_violation(model: ClassifierMixin, matrix: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    index = classes.index(1)
    return np.asarray(model.predict_proba(matrix)[:, index], dtype=float)


def parameter_grid(model: str) -> list[dict[str, Any]]:
    if model == "dummy":
        return [{"strategy": "prior"}]
    if model == "logistic_regression":
        return [{"C": value} for value in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if model == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (None, 4, 8)
            for leaf in (1, 3, 5)
        ]
    if model == "mlp":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden in ((16,), (32,), (64,))
            for alpha in (0.0001, 0.001)
        ]
    raise ValueError(model)


def build_model(model: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    if model == "dummy":
        return DummyClassifier(strategy=params["strategy"], random_state=seed)
    if model == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            ),
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if model == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                alpha=float(params["alpha"]),
                max_iter=2000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    raise ValueError(model)


def selection_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(row["mean_balanced_accuracy"]),
        float(row["mean_f1_violation"]),
        canonical_json(row["parameters"]),
    )


def select_parameters(
    model_name: str,
    seeds: list[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search_rows = []
    for params in parameter_grid(model_name):
        balanced = []
        f1_values = []
        for seed in seeds:
            model = build_model(model_name, params, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_train, y_train)
            score = probability_violation(model, X_validation)
            result = metrics(y_validation, score)
            balanced.append(float(result["balanced_accuracy"]))
            f1_values.append(float(result["f1_violation"]))
        search_rows.append(
            {
                "parameters": params,
                "mean_balanced_accuracy": float(np.mean(balanced)),
                "std_balanced_accuracy": float(np.std(balanced, ddof=0)),
                "mean_f1_violation": float(np.mean(f1_values)),
                "std_f1_violation": float(np.std(f1_values, ddof=0)),
            }
        )
    selected = max(search_rows, key=selection_key)
    return dict(selected["parameters"]), search_rows


def prediction_rows(
    metadata: list[dict[str, Any]],
    y_true: np.ndarray,
    probabilities: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for seed, values in probabilities.items():
        for meta, truth, probability in zip(metadata, y_true, values, strict=True):
            source = meta["source"]
            rows.append(
                {
                    "seed": seed,
                    "example_id": meta["example_id"],
                    "group_id": meta["group_id"],
                    "task_id": source["task_id"],
                    "action_violation": int(truth),
                    "probability_violation": float(probability),
                    "predicted_violation": int(probability >= 0.5),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    source_manifest = json.loads(
        (args.features_root / "manifest.json").read_text(encoding="utf-8")
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )
    fold_results: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    negative_controls: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    try:
        for fold in source_manifest["folds"]:
            fold_name = fold["fold"]
            fold_result = {
                "fold": fold_name,
                "assignment": fold["assignment"],
                "test_evaluation": fold["test_evaluation"],
                "views": [],
            }
            for view in VIEWS:
                root = args.features_root / "folds" / fold_name / view
                X = {
                    role: np.load(root / f"X_{role}.npy", allow_pickle=False)
                    for role in ROLES
                }
                y = {
                    role: violation_labels(
                        np.load(root / f"y_action_{role}.npy", allow_pickle=False)
                    )
                    for role in ROLES
                }
                metadata = {
                    role: load_metadata(root / f"metadata_{role}.jsonl")
                    for role in ("calibration", "test")
                }
                X_refit = np.concatenate([X["train"], X["validation"]], axis=0)
                y_refit = np.concatenate([y["train"], y["validation"]], axis=0)
                view_result = {"view": view, "models": []}

                for model_name in MODELS:
                    selected, search = select_parameters(
                        model_name,
                        seeds,
                        X["train"],
                        y["train"],
                        X["validation"],
                        y["validation"],
                    )
                    destination = temporary / "folds" / fold_name / view / model_name
                    model_dir = destination / "models"
                    model_dir.mkdir(parents=True)
                    write_json(destination / "selected_parameters.json", selected)
                    write_csv(
                        destination / "validation_search.csv",
                        [
                            {
                                "parameters": canonical_json(row["parameters"]),
                                "mean_balanced_accuracy": row["mean_balanced_accuracy"],
                                "std_balanced_accuracy": row["std_balanced_accuracy"],
                                "mean_f1_violation": row["mean_f1_violation"],
                                "std_f1_violation": row["std_f1_violation"],
                            }
                            for row in search
                        ],
                    )

                    calibration_probabilities: dict[int, np.ndarray] = {}
                    test_probabilities: dict[int, np.ndarray] = {}
                    seed_metrics = []
                    for seed in seeds:
                        model = build_model(model_name, selected, seed)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            model.fit(X_refit, y_refit)
                        joblib.dump(model, model_dir / f"seed_{seed}.joblib")
                        calibration_probability = probability_violation(
                            model, X["calibration"]
                        )
                        test_probability = probability_violation(model, X["test"])
                        calibration_probabilities[seed] = calibration_probability
                        test_probabilities[seed] = test_probability
                        seed_metrics.append(
                            {
                                "seed": seed,
                                "calibration": metrics(
                                    y["calibration"], calibration_probability
                                ),
                                "test": metrics(y["test"], test_probability),
                            }
                        )

                    write_csv(
                        destination / "predictions_calibration.csv",
                        prediction_rows(
                            metadata["calibration"],
                            y["calibration"],
                            calibration_probabilities,
                        ),
                    )
                    write_csv(
                        destination / "predictions_test.csv",
                        prediction_rows(
                            metadata["test"], y["test"], test_probabilities
                        ),
                    )
                    ensemble_calibration = np.mean(
                        list(calibration_probabilities.values()), axis=0
                    )
                    ensemble_test = np.mean(list(test_probabilities.values()), axis=0)
                    ensemble_metrics = {
                        "calibration": metrics(y["calibration"], ensemble_calibration),
                        "test": metrics(y["test"], ensemble_test),
                    }
                    write_json(destination / "seed_metrics.json", seed_metrics)
                    write_json(destination / "ensemble_metrics.json", ensemble_metrics)

                    target = (
                        pooled
                        if fold["test_evaluation"] == "binary_generalization"
                        else negative_controls
                    )
                    target[(view, model_name)].append((y["test"], ensemble_test))
                    view_result["models"].append(
                        {
                            "model": model_name,
                            "selected_parameters": selected,
                            "ensemble_metrics": ensemble_metrics,
                        }
                    )
                fold_result["views"].append(view_result)
            fold_results.append(fold_result)

        summary: dict[str, Any] = {}
        for view in VIEWS:
            summary[view] = {}
            for model_name in MODELS:
                binary_parts = pooled[(view, model_name)]
                control_parts = negative_controls[(view, model_name)]
                y_binary = np.concatenate([part[0] for part in binary_parts])
                p_binary = np.concatenate([part[1] for part in binary_parts])
                y_control = np.concatenate([part[0] for part in control_parts])
                p_control = np.concatenate([part[1] for part in control_parts])
                summary[view][model_name] = {
                    "binary_generalization_pooled": metrics(y_binary, p_binary),
                    "allow_only_negative_control": metrics(y_control, p_control),
                }

        manifest = {
            "schema_version": "0.1",
            "experiment": "task-held-out-classical-conformance-baselines",
            "source_encoding_id": source_manifest["encoding_id"],
            "features_root": str(args.features_root),
            "seeds": seeds,
            "models": list(MODELS),
            "views": list(VIEWS),
            "positive_class": "action_violation",
            "selection_split": "validation",
            "refit_splits": ["train", "validation"],
            "calibration_role": "prediction_export_only; no selection or fitting",
            "test_role": "single held-out evaluation after selection",
            "folds": fold_results,
            "summary": summary,
        }
        manifest["experiment_id"] = hashlib.sha256(
            canonical_json({
                key: value for key, value in manifest.items() if key != "folds"
            }).encode("utf-8")
        ).hexdigest()[:20]
        write_json(temporary / "results.json", manifest)
        write_json(temporary / "summary.json", summary)

        payloads = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary)}"
                for path in payloads
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 1H TASK-HELD-OUT LEARNED BASELINES ===")
    print("Experiment:", manifest["experiment_id"])
    print("Encoding:", source_manifest["encoding_id"])
    print("Seeds:", seeds)
    print("Output:", args.output_dir)
    for view in VIEWS:
        print(f"\n{view}")
        for model_name in MODELS:
            binary = summary[view][model_name]["binary_generalization_pooled"]
            control = summary[view][model_name]["allow_only_negative_control"]
            print(
                f"  {model_name}: "
                f"BA={binary['balanced_accuracy']:.4f}, "
                f"F1={binary['f1_violation']:.4f}, "
                f"recall={binary['recall_violation']:.4f}, "
                f"specificity={binary['specificity_allow']:.4f}, "
                f"ROC-AUC={binary['roc_auc']:.4f}, "
                f"control-FPR={control['false_positive_rate']:.4f}"
            )
    print("PASS: Validation alone selected hyperparameters.")
    print("PASS: Calibration and test were excluded from model selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
