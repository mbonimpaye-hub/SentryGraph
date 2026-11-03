"""Calibrate task-held-out conformance models and derive abstaining policies.

Phase 1I consumes the frozen predictions produced by Phase 1H.  Calibration
examples are used only for probability calibration and policy construction;
test labels are evaluated exactly once after the policy is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit calibration-only probability mappings and abstaining "
            "conformance policies to Phase 1H predictions."
        )
    )
    parser.add_argument(
        "--baselines-root",
        type=Path,
        required=True,
        help="Phase 1H task-held-out baseline result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Atomic output directory for calibrated policies.",
    )
    parser.add_argument(
        "--risk-budget",
        type=float,
        default=0.10,
        help=(
            "Maximum empirical calibration escape rate and false-denial "
            "rate used when selecting automatic-decision thresholds."
        ),
    )
    parser.add_argument(
        "--isolation-quantile",
        type=float,
        default=0.95,
        help=(
            "Calibration quantile of cross-seed disagreement above which "
            "an example is routed to isolation."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of equal-width bins for expected calibration error.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    """Convert NumPy scalar values into JSON-native values."""

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def load_prediction_ensemble(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        example_id = row["example_id"]
        truth = int(row["action_violation"])
        seed = int(row["seed"])

        if example_id not in grouped:
            grouped[example_id] = {
                "example_id": example_id,
                "group_id": row["group_id"],
                "task_id": row["task_id"],
                "truth": truth,
                "probabilities": {},
            }

        record = grouped[example_id]

        if (
            record["truth"] != truth
            or record["group_id"] != row["group_id"]
            or record["task_id"] != row["task_id"]
        ):
            raise RuntimeError(
                f"Inconsistent metadata for example {example_id}."
            )

        if seed in record["probabilities"]:
            raise RuntimeError(
                f"Duplicate seed {seed} for example {example_id}."
            )

        record["probabilities"][seed] = float(
            row["probability_violation"]
        )

    records: list[dict[str, Any]] = []
    expected_seeds: tuple[int, ...] | None = None

    for example_id in sorted(grouped):
        record = grouped[example_id]
        seeds = tuple(sorted(record["probabilities"]))

        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise RuntimeError(
                f"Seed mismatch for example {example_id}: {seeds} "
                f"!= {expected_seeds}."
            )

        probabilities = np.asarray(
            [record["probabilities"][seed] for seed in seeds],
            dtype=np.float64,
        )

        records.append(
            {
                "example_id": example_id,
                "group_id": record["group_id"],
                "task_id": record["task_id"],
                "truth": record["truth"],
                "seeds": list(seeds),
                "raw_probability": float(probabilities.mean()),
                "seed_std": float(probabilities.std(ddof=0)),
            }
        )

    if not records:
        raise RuntimeError(f"No predictions found in {path}.")

    return records


def logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_platt_calibrator(
    records: list[dict[str, Any]],
) -> LogisticRegression:
    scores = np.asarray(
        [record["raw_probability"] for record in records],
        dtype=np.float64,
    )
    labels = np.asarray(
        [record["truth"] for record in records],
        dtype=np.int64,
    )

    if len(np.unique(labels)) != 2:
        raise RuntimeError(
            "Calibration split must contain both action classes."
        )

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    calibrator.fit(logits(scores), labels)
    return calibrator


def apply_calibrator(
    calibrator: LogisticRegression,
    records: list[dict[str, Any]],
) -> None:
    scores = np.asarray(
        [record["raw_probability"] for record in records],
        dtype=np.float64,
    )
    calibrated = calibrator.predict_proba(logits(scores))[:, 1]

    for record, probability in zip(records, calibrated, strict=True):
        record["calibrated_probability"] = float(probability)


def empirical_rate(values: Iterable[bool]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def choose_allow_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    risk_budget: float,
) -> float:
    candidates = sorted(
        {0.0, 1.0, *map(float, probabilities.tolist())}
    )
    violation_scores = probabilities[labels == 1]
    feasible = [
        threshold
        for threshold in candidates
        if empirical_rate(violation_scores < threshold) <= risk_budget
    ]
    return max(feasible)


def choose_deny_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    risk_budget: float,
) -> float:
    candidates = sorted(
        {0.0, 1.0, *map(float, probabilities.tolist())}
    )
    allow_scores = probabilities[labels == 0]
    feasible = [
        threshold
        for threshold in candidates
        if empirical_rate(allow_scores > threshold) <= risk_budget
    ]
    return min(feasible)


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(
            np.quantile(values, quantile, interpolation="higher")
        )


def build_policy(
    records: list[dict[str, Any]],
    risk_budget: float,
    isolation_quantile: float,
) -> dict[str, Any]:
    probabilities = np.asarray(
        [record["calibrated_probability"] for record in records],
        dtype=np.float64,
    )
    labels = np.asarray(
        [record["truth"] for record in records],
        dtype=np.int64,
    )
    disagreements = np.asarray(
        [record["seed_std"] for record in records],
        dtype=np.float64,
    )

    raw_allow = choose_allow_threshold(
        probabilities,
        labels,
        risk_budget,
    )
    raw_deny = choose_deny_threshold(
        probabilities,
        labels,
        risk_budget,
    )

    thresholds_crossed = raw_allow > raw_deny
    if thresholds_crossed:
        midpoint = (raw_allow + raw_deny) / 2.0
        allow_threshold = midpoint
        deny_threshold = midpoint
    else:
        allow_threshold = raw_allow
        deny_threshold = raw_deny

    return {
        "schema_version": "0.1",
        "risk_budget": risk_budget,
        "isolation_quantile": isolation_quantile,
        "allow_threshold": allow_threshold,
        "deny_threshold": deny_threshold,
        "isolation_seed_std_threshold": quantile_higher(
            disagreements,
            isolation_quantile,
        ),
        "raw_budget_feasible_allow_threshold": raw_allow,
        "raw_budget_feasible_deny_threshold": raw_deny,
        "thresholds_crossed_before_reconciliation": thresholds_crossed,
        "decision_order": ["isolate", "allow", "deny", "approve"],
        "threshold_semantics": {
            "isolate": "seed_std > isolation_seed_std_threshold",
            "allow": "calibrated_probability < allow_threshold",
            "deny": "calibrated_probability > deny_threshold",
            "approve": "otherwise",
        },
    }


def assign_decisions(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
) -> None:
    for record in records:
        probability = record["calibrated_probability"]
        disagreement = record["seed_std"]

        if disagreement > policy["isolation_seed_std_threshold"]:
            decision = "isolate"
        elif probability < policy["allow_threshold"]:
            decision = "allow"
        elif probability > policy["deny_threshold"]:
            decision = "deny"
        else:
            decision = "approve"

        record["decision"] = decision


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0

    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        if not np.any(mask):
            continue

        confidence = float(probabilities[mask].mean())
        frequency = float(labels[mask].mean())
        value += (int(mask.sum()) / total) * abs(confidence - frequency)

    return value


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def decision_metrics(
    records: list[dict[str, Any]],
    ece_bins: int,
) -> dict[str, Any]:
    labels = np.asarray(
        [record["truth"] for record in records],
        dtype=np.int64,
    )
    raw = np.asarray(
        [record["raw_probability"] for record in records],
        dtype=np.float64,
    )
    calibrated = np.asarray(
        [record["calibrated_probability"] for record in records],
        dtype=np.float64,
    )
    decisions = [record["decision"] for record in records]
    counts = Counter(decisions)

    allow_total = int((labels == 0).sum())
    violation_total = int((labels == 1).sum())
    automatic = counts["allow"] + counts["deny"]
    automatic_correct = sum(
        (decision == "allow" and truth == 0)
        or (decision == "deny" and truth == 1)
        for decision, truth in zip(decisions, labels, strict=True)
    )
    violation_escapes = sum(
        decision == "allow" and truth == 1
        for decision, truth in zip(decisions, labels, strict=True)
    )
    false_denials = sum(
        decision == "deny" and truth == 0
        for decision, truth in zip(decisions, labels, strict=True)
    )
    benign_escalations = sum(
        decision in {"approve", "isolate"} and truth == 0
        for decision, truth in zip(decisions, labels, strict=True)
    )
    violation_interventions = sum(
        decision in {"deny", "approve", "isolate"} and truth == 1
        for decision, truth in zip(decisions, labels, strict=True)
    )

    binary = len(np.unique(labels)) == 2

    return {
        "examples": len(records),
        "allow_examples": allow_total,
        "violation_examples": violation_total,
        "decision_counts": dict(sorted(counts.items())),
        "automatic_coverage": safe_ratio(automatic, len(records)),
        "approval_rate": safe_ratio(counts["approve"], len(records)),
        "isolation_rate": safe_ratio(counts["isolate"], len(records)),
        "selective_accuracy": safe_ratio(automatic_correct, automatic),
        "violation_escape_rate": safe_ratio(
            violation_escapes,
            violation_total,
        ),
        "violation_intervention_rate": safe_ratio(
            violation_interventions,
            violation_total,
        ),
        "false_denial_rate": safe_ratio(false_denials, allow_total),
        "benign_escalation_rate": safe_ratio(
            benign_escalations,
            allow_total,
        ),
        "unsafe_automatic_error_rate": safe_ratio(
            violation_escapes + false_denials,
            len(records),
        ),
        "brier_raw": float(brier_score_loss(labels, raw)),
        "brier_calibrated": float(
            brier_score_loss(labels, calibrated)
        ),
        "ece_raw": expected_calibration_error(labels, raw, ece_bins),
        "ece_calibrated": expected_calibration_error(
            labels,
            calibrated,
            ece_bins,
        ),
        "roc_auc_raw": (
            float(roc_auc_score(labels, raw)) if binary else None
        ),
        "roc_auc_calibrated": (
            float(roc_auc_score(labels, calibrated)) if binary else None
        ),
    }


def write_decisions(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    columns = [
        "example_id",
        "group_id",
        "task_id",
        "truth",
        "raw_probability",
        "seed_std",
        "calibrated_probability",
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_model_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    for calibration in root.glob(
        "folds/*/*/*/predictions_calibration.csv"
    ):
        model_directory = calibration.parent
        if not (model_directory / "predictions_test.csv").exists():
            raise RuntimeError(
                f"Missing test predictions beside {calibration}."
            )
        directories.append(model_directory)
    return sorted(directories)


def pooled_metrics(
    records: list[dict[str, Any]],
    ece_bins: int,
) -> dict[str, Any] | None:
    if not records:
        return None
    return decision_metrics(records, ece_bins)


def format_metric(
    value: float | None,
) -> str:
    """Format optional metrics safely for terminal output."""

    if value is None:
        return "NA"

    return f"{value:.4f}"


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.risk_budget < 0.5:
        raise ValueError("--risk-budget must be in [0, 0.5).")
    if not 0.0 < args.isolation_quantile <= 1.0:
        raise ValueError("--isolation-quantile must be in (0, 1].")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be at least two.")

    source_results = read_json(args.baselines_root / "results.json")
    model_directories = discover_model_directories(args.baselines_root)
    if not model_directories:
        raise RuntimeError("No Phase 1H prediction bundles were found.")

    temporary_parent = Path(
        tempfile.mkdtemp(
            prefix="phase1i-",
            dir=args.output_dir.parent,
        )
    )
    temporary_output = temporary_parent / args.output_dir.name
    temporary_output.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    pooled_binary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pooled_control: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    try:
        for model_directory in model_directories:
            relative = model_directory.relative_to(args.baselines_root)
            _, fold, view, model = relative.parts
            output_directory = temporary_output / relative
            output_directory.mkdir(parents=True, exist_ok=True)

            calibration = load_prediction_ensemble(
                model_directory / "predictions_calibration.csv"
            )
            test = load_prediction_ensemble(
                model_directory / "predictions_test.csv"
            )

            calibrator = fit_platt_calibrator(calibration)
            apply_calibrator(calibrator, calibration)
            apply_calibrator(calibrator, test)
            policy = build_policy(
                calibration,
                args.risk_budget,
                args.isolation_quantile,
            )
            assign_decisions(calibration, policy)
            assign_decisions(test, policy)

            calibration_metrics = decision_metrics(
                calibration,
                args.ece_bins,
            )
            test_metrics = decision_metrics(test, args.ece_bins)

            policy["calibration_budget_audit"] = {
                "violation_escape_rate": calibration_metrics[
                    "violation_escape_rate"
                ],
                "false_denial_rate": calibration_metrics[
                    "false_denial_rate"
                ],
                "escape_budget_satisfied": (
                    calibration_metrics["violation_escape_rate"]
                    is not None
                    and calibration_metrics["violation_escape_rate"]
                    <= args.risk_budget + 1e-12
                ),
                "false_denial_budget_satisfied": (
                    calibration_metrics["false_denial_rate"]
                    is not None
                    and calibration_metrics["false_denial_rate"]
                    <= args.risk_budget + 1e-12
                ),
            }

            joblib.dump(calibrator, output_directory / "calibrator.joblib")
            write_json(output_directory / "policy.json", policy)
            write_decisions(
                output_directory / "decisions_calibration.csv",
                calibration,
            )
            write_decisions(
                output_directory / "decisions_test.csv",
                test,
            )
            metrics = {
                "fold": fold,
                "view": view,
                "model": model,
                "calibration": calibration_metrics,
                "test": test_metrics,
            }
            write_json(output_directory / "metrics.json", metrics)

            evaluation = next(
                item["test_evaluation"]
                for item in source_results["folds"]
                if item["fold"] == fold
            )
            result = {
                "fold": fold,
                "view": view,
                "model": model,
                "test_evaluation": evaluation,
                "policy": policy,
                "metrics": metrics,
            }
            results.append(result)

            target = (
                pooled_binary
                if evaluation == "binary_generalization"
                else pooled_control
            )
            target[(view, model)].extend(test)

        summary: dict[str, Any] = {}
        keys = sorted(set(pooled_binary) | set(pooled_control))
        for view, model in keys:
            summary.setdefault(view, {})[model] = {
                "binary_generalization": pooled_metrics(
                    pooled_binary[(view, model)],
                    args.ece_bins,
                ),
                "allow_only_negative_control": pooled_metrics(
                    pooled_control[(view, model)],
                    args.ece_bins,
                ),
            }

        configuration = {
            "source_experiment_id": source_results.get("experiment_id"),
            "risk_budget": args.risk_budget,
            "isolation_quantile": args.isolation_quantile,
            "ece_bins": args.ece_bins,
            "calibration_source": "calibration predictions only",
            "test_labels_used_for_policy_construction": False,
        }
        experiment_id = hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]

        root_results = {
            "schema_version": "0.1",
            "experiment_id": experiment_id,
            "configuration": configuration,
            "results": results,
            "summary": summary,
            "limitations": [
                "Calibration uses one held-out task per fold and remains small.",
                "The allow-only control contains one unique deduplicated example.",
                "Approval and isolation are offline routing outcomes, not yet live human workflows.",
            ],
        }
        write_json(temporary_output / "results.json", root_results)
        write_json(temporary_output / "summary.json", summary)

        checksum_paths = sorted(
            path
            for path in temporary_output.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(temporary_output)}"
            for path in checksum_paths
        ]
        (temporary_output / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        if args.output_dir.exists():
            raise FileExistsError(
                f"Output already exists: {args.output_dir}"
            )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.replace(args.output_dir)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    print("=== PHASE 1I CALIBRATION AND ABSTENTION ===")
    print("Experiment:", experiment_id)
    print("Output:", args.output_dir)
    print("Risk budget:", args.risk_budget)
    print("Isolation quantile:", args.isolation_quantile)
    print("Policies:", len(results))

    for view in sorted(summary):
        print()
        print(view)
        for model in sorted(summary[view]):
            metrics = summary[view][model]["binary_generalization"]
            print(
                f"  {model}: "
                f"coverage={format_metric(metrics['automatic_coverage'])}, "
                f"selective_accuracy={format_metric(metrics['selective_accuracy'])}, "
                f"escape_rate={format_metric(metrics['violation_escape_rate'])}, "
                f"false_denial_rate={format_metric(metrics['false_denial_rate'])}, "
                f"approval_rate={format_metric(metrics['approval_rate'])}, "
                f"isolation_rate={format_metric(metrics['isolation_rate'])}"
            )

    print()
    print("PASS: Calibration and policy construction used no test labels.")
    print("PASS: Each policy exposes allow, deny, approval, and isolation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
