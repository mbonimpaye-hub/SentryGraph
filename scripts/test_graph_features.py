"""Unit tests for graph feature extraction and train-only vocabulary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from sentrygraph.graph_features import GraphFeatureEncoder, example_to_feature_dict


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    root = Path("outputs/learning/graph_conformance_v1")
    train = load(root / "train.jsonl")
    validation = load(root / "validation.jsonl")

    original = example_to_feature_dict(train[0])
    mutated = copy.deepcopy(train[0])
    mutated["labels"] = {"action_permitted": not train[0]["labels"]["action_permitted"]}
    mutated["source"] = {"task_id": "injected-source-token"}
    mutated["split"] = "test"
    mutated["group_id"] = "injected-group"
    assert example_to_feature_dict(mutated) == original

    encoder_a = GraphFeatureEncoder().fit(train)
    encoder_b = GraphFeatureEncoder().fit(list(reversed(train)))
    assert encoder_a.feature_names == encoder_b.feature_names
    np.testing.assert_array_equal(
        encoder_a.transform(train),
        encoder_b.transform(train),
    )

    train_matrix = encoder_a.transform(train)
    validation_matrix = encoder_a.transform(validation)
    assert train_matrix.shape[0] == len(train)
    assert validation_matrix.shape[0] == len(validation)
    assert train_matrix.shape[1] == validation_matrix.shape[1]
    assert train_matrix.dtype == np.float32
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(validation_matrix).all()

    print("PASS: Feature extraction ignores labels and source metadata.")
    print("PASS: Vocabulary and matrices are order deterministic.")
    print("Feature count:", train_matrix.shape[1])
    print("Train shape:", train_matrix.shape)
    print("Validation shape:", validation_matrix.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
