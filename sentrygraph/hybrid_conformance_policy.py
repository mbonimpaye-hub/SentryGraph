"""Deploy a frozen task-held-out graph-conformance policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from sentrygraph.graph_features import example_to_feature_dict


EPSILON = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def violation_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError(
            f"Model classes do not contain violation class 1: {classes}"
        )
    return probabilities[:, int(matches[0])]


@dataclass(frozen=True)
class HybridPrediction:
    decision: str
    raw_probability: float
    calibrated_probability: float
    seed_std: float
    seed_probabilities: tuple[float, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrozenHybridConformancePolicy:
    """Five-seed ensemble plus frozen calibrator and abstention policy."""

    def __init__(self, bundle_root: str | Path) -> None:
        self.root = Path(bundle_root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.policy = json.loads(
            (self.root / "policy.json").read_text(encoding="utf-8")
        )
        self.feature_names = json.loads(
            (self.root / "feature_names.json").read_text(encoding="utf-8")
        )
        self.vocabulary = json.loads(
            (self.root / "vocabulary.json").read_text(encoding="utf-8")
        )
        if len(self.feature_names) != len(self.vocabulary):
            raise RuntimeError("Feature-name and vocabulary sizes differ.")
        for index, name in enumerate(self.feature_names):
            if self.vocabulary.get(name) != index:
                raise RuntimeError("Vocabulary order does not match features.")

        self.seeds = tuple(int(value) for value in self.manifest["seeds"])
        self.models = tuple(
            joblib.load(self.root / "models" / f"seed_{seed}.joblib")
            for seed in self.seeds
        )
        self.calibrator = joblib.load(self.root / "calibrator.joblib")
        self._verify_checksums()

    def _verify_checksums(self) -> None:
        expected = self.manifest.get("files", {})
        for relative, digest in expected.items():
            path = self.root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != digest:
                raise RuntimeError(f"Checksum mismatch: {relative}")

    def encode_feature_dict(self, values: Mapping[str, Any]) -> np.ndarray:
        row = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        for key, value in values.items():
            if isinstance(value, str):
                feature = f"{key}={value}"
                encoded = 1.0
            else:
                feature = key
                try:
                    encoded = float(value)
                except (TypeError, ValueError):
                    continue
            index = self.vocabulary.get(feature)
            if index is not None:
                row[0, int(index)] = encoded
        return row

    def encode_example(self, example: Mapping[str, Any]) -> np.ndarray:
        return self.encode_feature_dict(example_to_feature_dict(example))

    def predict_matrix(self, matrix: np.ndarray) -> list[HybridPrediction]:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected (*, {len(self.feature_names)}) matrix, "
                f"received {matrix.shape}."
            )
        per_seed = np.stack(
            [violation_probability(model, matrix) for model in self.models],
            axis=0,
        )
        raw = per_seed.mean(axis=0)
        disagreement = per_seed.std(axis=0, ddof=0)
        calibrated = self.calibrator.predict_proba(logits(raw))[:, 1]
        output: list[HybridPrediction] = []
        for column in range(matrix.shape[0]):
            probability = float(calibrated[column])
            seed_std = float(disagreement[column])
            if seed_std > self.policy["isolation_seed_std_threshold"]:
                decision = "isolate"
            elif probability < self.policy["allow_threshold"]:
                decision = "allow"
            elif probability > self.policy["deny_threshold"]:
                decision = "deny"
            else:
                decision = "approve"
            output.append(
                HybridPrediction(
                    decision=decision,
                    raw_probability=float(raw[column]),
                    calibrated_probability=probability,
                    seed_std=seed_std,
                    seed_probabilities=tuple(
                        float(value) for value in per_seed[:, column]
                    ),
                    seeds=self.seeds,
                )
            )
        return output

    def predict_example(self, example: Mapping[str, Any]) -> HybridPrediction:
        return self.predict_matrix(self.encode_example(example))[0]

    def combine_with_deterministic(
        self,
        *,
        deterministic_allowed: bool,
        prediction: HybridPrediction | None,
    ) -> dict[str, Any]:
        if not deterministic_allowed:
            return {
                "decision": "deny",
                "source": "deterministic_policy_gate",
                "effect_allowed": False,
                "learned_override": False,
            }
        if prediction is None:
            raise ValueError("Prediction required after deterministic allow.")
        return {
            "decision": prediction.decision,
            "source": "learned_conformance_policy",
            "effect_allowed": prediction.decision == "allow",
            "learned_override": False,
            "prediction": prediction.to_dict(),
        }
