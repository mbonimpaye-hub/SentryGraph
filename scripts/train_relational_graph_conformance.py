"""Train task-held-out relational graph conformance networks.

Architecture selection uses validation only.  Selected configurations are
refitted on train+validation, calibration is prediction-export only, and test
is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_SEEDS = (13, 42, 73, 101, 137)
SPLITS = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class ModelConfiguration:
    hidden_dimension: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float


CONFIGURATIONS = (
    ModelConfiguration(32, 2, 0.10, 1e-3, 1e-4),
    ModelConfiguration(32, 3, 0.20, 1e-3, 1e-4),
    ModelConfiguration(64, 2, 0.20, 5e-4, 1e-4),
    ModelConfiguration(64, 3, 0.30, 5e-4, 1e-4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train task-held-out relational SentryGraph models."
    )
    parser.add_argument("--graphs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated training seeds.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="CPU is the deterministic default; auto may select Apple MPS.",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GraphDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_graphs(records: list[dict[str, Any]]) -> dict[str, Any]:
    node_token_parts = []
    edge_index_parts = []
    edge_type_parts = []
    batch_parts = []
    candidate_indices = []
    labels = []
    metadata = []
    offset = 0

    for graph_index, record in enumerate(records):
        node_tokens = record["node_token_index"].clone()
        node_tokens[0] += offset
        node_token_parts.append(node_tokens)

        edges = record["edge_index"].clone()
        edges += offset
        edge_index_parts.append(edges)
        edge_type_parts.append(record["edge_type"])
        batch_parts.append(
            torch.full(
                (record["num_nodes"],),
                graph_index,
                dtype=torch.long,
            )
        )
        candidate_indices.append(record["candidate_index"] + offset)
        labels.append(record["action_label"])
        metadata.append(record["metadata"])
        offset += record["num_nodes"]

    return {
        "node_token_index": torch.cat(node_token_parts, dim=1),
        "edge_index": torch.cat(edge_index_parts, dim=1),
        "edge_type": torch.cat(edge_type_parts),
        "batch": torch.cat(batch_parts),
        "candidate_index": torch.tensor(candidate_indices, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "metadata": metadata,
        "num_graphs": len(records),
        "num_nodes": offset,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class RelationalLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        relations: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_linear = nn.Linear(dimension, dimension)
        self.forward_weights = nn.Parameter(
            torch.empty(relations, dimension, dimension)
        )
        self.reverse_weights = nn.Parameter(
            torch.empty(relations, dimension, dimension)
        )
        self.normalization = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.forward_weights)
        nn.init.xavier_uniform_(self.reverse_weights)

    @staticmethod
    def messages(
        states: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message = torch.einsum(
            "eh,ehd->ed",
            states[source],
            weights[relation],
        )
        aggregate = states.new_zeros(states.shape)
        aggregate.index_add_(0, target, message)
        degree = states.new_zeros((states.shape[0], 1))
        degree.index_add_(
            0,
            target,
            torch.ones(
                (target.shape[0], 1),
                device=states.device,
                dtype=states.dtype,
            ),
        )
        return aggregate, degree

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        forward, forward_degree = self.messages(
            states,
            source,
            target,
            edge_type,
            self.forward_weights,
        )
        reverse, reverse_degree = self.messages(
            states,
            target,
            source,
            edge_type,
            self.reverse_weights,
        )
        degree = (forward_degree + reverse_degree).clamp_min(1.0)
        updated = self.self_linear(states) + (forward + reverse) / degree
        updated = self.normalization(updated)
        return self.dropout(torch.relu(updated))


class RelationalConformanceNetwork(nn.Module):
    def __init__(
        self,
        tokens: int,
        relations: int,
        configuration: ModelConfiguration,
    ) -> None:
        super().__init__()
        dimension = configuration.hidden_dimension
        self.token_embedding = nn.Embedding(tokens, dimension)
        self.layers = nn.ModuleList(
            RelationalLayer(dimension, relations, configuration.dropout)
            for _ in range(configuration.layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(dimension * 3, dimension),
            nn.ReLU(),
            nn.Dropout(configuration.dropout),
            nn.Linear(dimension, 1),
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def initial_states(
        self,
        node_token_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        rows, token_ids = node_token_index
        embeddings = self.token_embedding(token_ids)
        states = embeddings.new_zeros(
            (num_nodes, embeddings.shape[1])
        )
        states.index_add_(0, rows, embeddings)
        counts = embeddings.new_zeros((num_nodes, 1))
        counts.index_add_(
            0,
            rows,
            torch.ones(
                (len(rows), 1),
                device=embeddings.device,
                dtype=embeddings.dtype,
            ),
        )
        return states / counts.clamp_min(1.0)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        states = self.initial_states(
            batch["node_token_index"],
            batch["num_nodes"],
        )
        for layer in self.layers:
            states = layer(states, batch["edge_index"], batch["edge_type"])

        graph_means = states.new_zeros(
            (batch["num_graphs"], states.shape[1])
        )
        graph_means.index_add_(0, batch["batch"], states)
        counts = torch.bincount(
            batch["batch"], minlength=batch["num_graphs"]
        ).to(states.dtype).unsqueeze(1)
        graph_means /= counts.clamp_min(1.0)

        graph_maxima = torch.stack(
            [
                states[batch["batch"] == index].max(dim=0).values
                for index in range(batch["num_graphs"])
            ]
        )
        candidates = states[batch["candidate_index"]]
        representation = torch.cat(
            [graph_means, graph_maxima, candidates], dim=1
        )
        return self.classifier(representation).squeeze(1)


def loader(
    records: list[dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        GraphDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def metrics(
    truth: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(np.int64)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    binary = len(np.unique(truth)) == 2
    return {
        "examples": len(truth),
        "balanced_accuracy": (
            float(balanced_accuracy_score(truth, prediction))
            if binary
            else None
        ),
        "f1_violation": float(f1_score(truth, prediction, zero_division=0)),
        "recall_violation": float(
            recall_score(truth, prediction, zero_division=0)
        ),
        "specificity_allow": tn / (tn + fp) if (tn + fp) else None,
        "roc_auc": (
            float(roc_auc_score(truth, probability)) if binary else None
        ),
        "average_precision": (
            float(average_precision_score(truth, probability))
            if binary
            else None
        ),
        "confusion_matrix": {
            "true_allow_pred_allow": tn,
            "true_allow_pred_violation": fp,
            "true_violation_pred_allow": fn,
            "true_violation_pred_violation": tp,
        },
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    records: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    probabilities = []
    labels = []
    metadata = []
    for raw_batch in loader(records, batch_size, False, 0):
        batch = move_batch(raw_batch, device)
        logits = model(batch)
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(raw_batch["labels"].to(torch.int64).tolist())
        metadata.extend(raw_batch["metadata"])
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        metadata,
    )


def class_weight(records: list[dict[str, Any]]) -> float:
    positives = sum(record["action_label"] for record in records)
    negatives = len(records) - positives
    if not positives or not negatives:
        raise RuntimeError("Training partition must contain both classes.")
    return negatives / positives


def train_with_validation(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    tokens: int,
    relations: int,
    configuration: ModelConfiguration,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int]:
    set_seed(seed)
    model = RelationalConformanceNetwork(
        tokens, relations, configuration
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weight(train_records), device=device)
    )
    best_score = (-math.inf, -math.inf)
    best_state = None
    best_metrics = None
    best_epoch = 0
    remaining_patience = patience

    for epoch in range(1, epochs + 1):
        model.train()
        for raw_batch in loader(train_records, batch_size, True, seed + epoch):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        truth, probability, _ = predict(
            model, validation_records, batch_size, device
        )
        current_metrics = metrics(truth, probability)
        score = (
            current_metrics["balanced_accuracy"],
            current_metrics["f1_violation"],
        )
        if score > best_score:
            best_score = score
            best_metrics = current_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Validation selection failed.")
    return best_state, best_metrics, best_epoch


def refit(
    records: list[dict[str, Any]],
    tokens: int,
    relations: int,
    configuration: ModelConfiguration,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> RelationalConformanceNetwork:
    set_seed(seed)
    model = RelationalConformanceNetwork(
        tokens, relations, configuration
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weight(records), device=device)
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for raw_batch in loader(records, batch_size, True, seed + epoch):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    columns = [
        "seed",
        "example_id",
        "group_id",
        "task_id",
        "action_violation",
        "probability_violation",
        "predicted_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ensemble_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["example_id"], []).append(row)
    truth = []
    probabilities = []
    for example_id in sorted(grouped):
        items = grouped[example_id]
        labels = {int(item["action_violation"]) for item in items}
        if len(labels) != 1:
            raise RuntimeError(f"Label disagreement for {example_id}.")
        truth.append(labels.pop())
        probabilities.append(
            float(np.mean([item["probability_violation"] for item in items]))
        )
    return metrics(np.asarray(truth), np.asarray(probabilities))


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


def macro_summary(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "balanced_accuracy",
        "f1_violation",
        "recall_violation",
        "specificity_allow",
        "roc_auc",
        "average_precision",
    )
    views = sorted(
        {
            view["view"]
            for fold in fold_results
            for view in fold["views"]
        }
    )
    summary = {}

    for view_name in views:
        binary_metrics = [
            view["ensemble_metrics"]["test"]
            for fold in fold_results
            if fold["test_evaluation"] == "binary_generalization"
            for view in fold["views"]
            if view["view"] == view_name
        ]
        control_metrics = [
            view["ensemble_metrics"]["test"]
            for fold in fold_results
            if fold["test_evaluation"] == "allow_only_negative_control"
            for view in fold["views"]
            if view["view"] == view_name
        ]

        binary_macro = {}
        for metric_name in metric_names:
            values = [
                item[metric_name]
                for item in binary_metrics
                if item[metric_name] is not None
            ]
            binary_macro[metric_name] = (
                float(np.mean(values)) if values else None
            )

        summary[view_name] = {
            "binary_folds": len(binary_metrics),
            "binary_macro": binary_macro,
            "allow_only_control_folds": len(control_metrics),
            "allow_only_control_specificity": (
                float(
                    np.mean(
                        [
                            item["specificity_allow"]
                            for item in control_metrics
                            if item["specificity_allow"] is not None
                        ]
                    )
                )
                if control_metrics
                else None
            ),
        }

    return summary


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    device = select_device(args.device)
    manifest = read_json(args.graphs_root / "manifest.json")

    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix="phase1k-", dir=args.output_dir.parent)
    )
    output = temporary_parent / args.output_dir.name
    output.mkdir()
    fold_results = []

    try:
        for fold in manifest["folds"]:
            fold_name = fold["fold"]
            fold_views = []
            for view_summary in fold["views"]:
                view = view_summary["view"]
                print(
                    f"SELECT {fold_name} {view}",
                    flush=True,
                )
                source = args.graphs_root / "folds" / fold_name / view
                bundles = {
                    split: torch_load(source / f"{split}.pt")["records"]
                    for split in SPLITS
                }
                vocabulary = read_json(source / "vocabulary.json")
                tokens = len(vocabulary["token_to_id"])
                relations = len(vocabulary["relation_to_id"])

                selection_runs = []
                for configuration in CONFIGURATIONS:
                    print(
                        "  candidate",
                        asdict(configuration),
                        flush=True,
                    )
                    runs = []
                    for seed in seeds:
                        _, validation_metrics, best_epoch = train_with_validation(
                            bundles["train"],
                            bundles["validation"],
                            tokens,
                            relations,
                            configuration,
                            seed,
                            args.epochs,
                            args.patience,
                            args.batch_size,
                            device,
                        )
                        runs.append(
                            {
                                "seed": seed,
                                "best_epoch": best_epoch,
                                "validation_metrics": validation_metrics,
                            }
                        )
                    mean_ba = float(
                        np.mean(
                            [
                                run["validation_metrics"]["balanced_accuracy"]
                                for run in runs
                            ]
                        )
                    )
                    mean_f1 = float(
                        np.mean(
                            [run["validation_metrics"]["f1_violation"] for run in runs]
                        )
                    )
                    selection_runs.append(
                        {
                            "configuration": asdict(configuration),
                            "mean_validation_balanced_accuracy": mean_ba,
                            "mean_validation_f1": mean_f1,
                            "runs": runs,
                        }
                    )

                selected = max(
                    selection_runs,
                    key=lambda item: (
                        item["mean_validation_balanced_accuracy"],
                        item["mean_validation_f1"],
                        -item["configuration"]["hidden_dimension"],
                        -item["configuration"]["layers"],
                    ),
                )
                configuration = ModelConfiguration(**selected["configuration"])
                refit_epochs = max(
                    1,
                    int(
                        round(
                            np.median(
                                [run["best_epoch"] for run in selected["runs"]]
                            )
                        )
                    ),
                )
                combined = bundles["train"] + bundles["validation"]
                print(
                    "  selected",
                    asdict(configuration),
                    f"refit_epochs={refit_epochs}",
                    flush=True,
                )
                view_output = output / "folds" / fold_name / view
                model_output = view_output / "models"
                model_output.mkdir(parents=True)
                prediction_rows = {"calibration": [], "test": []}

                for seed in seeds:
                    model = refit(
                        combined,
                        tokens,
                        relations,
                        configuration,
                        seed,
                        refit_epochs,
                        args.batch_size,
                        device,
                    )
                    torch.save(
                        {
                            "model_state_dict": {
                                key: value.detach().cpu()
                                for key, value in model.state_dict().items()
                            },
                            "configuration": asdict(configuration),
                            "tokens": tokens,
                            "relations": relations,
                            "seed": seed,
                            "refit_epochs": refit_epochs,
                        },
                        model_output / f"relational_gnn_seed{seed}.pt",
                    )
                    for split in ("calibration", "test"):
                        truth, probability, metadata = predict(
                            model,
                            bundles[split],
                            args.batch_size,
                            device,
                        )
                        for label, score, item in zip(
                            truth, probability, metadata, strict=True
                        ):
                            prediction_rows[split].append(
                                {
                                    "seed": seed,
                                    "example_id": item["example_id"],
                                    "group_id": item["group_id"],
                                    "task_id": item["task_id"],
                                    "action_violation": int(label),
                                    "probability_violation": float(score),
                                    "predicted_violation": int(score >= 0.5),
                                }
                            )

                for split, rows in prediction_rows.items():
                    write_predictions(view_output / f"predictions_{split}.csv", rows)

                result = {
                    "view": view,
                    "selected_configuration": asdict(configuration),
                    "refit_epochs": refit_epochs,
                    "selection": selection_runs,
                    "ensemble_metrics": {
                        split: ensemble_metrics(rows)
                        for split, rows in prediction_rows.items()
                    },
                }
                write_json(view_output / "result.json", result)
                fold_views.append(result)

            fold_results.append(
                {
                    "fold": fold_name,
                    "test_evaluation": fold["test_evaluation"],
                    "views": fold_views,
                }
            )

        configuration = {
            "source_encoding_id": manifest["encoding_id"],
            "seeds": list(seeds),
            "selection_split": "validation",
            "refit_splits": ["train", "validation"],
            "calibration_role": "prediction_export_only",
            "test_role": "single held-out evaluation after selection",
            "device": str(device),
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "candidate_configurations": [
                asdict(configuration) for configuration in CONFIGURATIONS
            ],
        }
        experiment_id = hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        summary = macro_summary(fold_results)
        results = {
            "schema_version": "0.1",
            "experiment_id": experiment_id,
            "configuration": configuration,
            "folds": fold_results,
            "summary": summary,
        }
        write_json(output / "results.json", results)
        write_json(output / "summary.json", summary)
        files = sorted(path for path in output.rglob("*") if path.is_file())
        (output / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(output)}" for path in files
            )
            + "\n",
            encoding="utf-8",
        )
        output.replace(args.output_dir)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    print("=== PHASE 1K RELATIONAL GRAPH CONFORMANCE ===")
    print("Experiment:", experiment_id)
    print("Source encoding:", manifest["encoding_id"])
    print("Device:", device)
    print("Seeds:", list(seeds))
    print("Selection: validation only")
    print("Calibration: prediction export only")
    print("Test: single evaluation after selection")
    for fold in fold_results:
        print()
        print(fold["fold"], fold["test_evaluation"])
        for view in fold["views"]:
            test = view["ensemble_metrics"]["test"]
            print(
                f"  {view['view']}: selected={view['selected_configuration']}, "
                f"epochs={view['refit_epochs']}, "
                f"BA={test['balanced_accuracy']}, "
                f"F1={test['f1_violation']:.4f}, "
                f"recall={test['recall_violation']:.4f}, "
                f"specificity={test['specificity_allow']:.4f}, "
                f"AUC={test['roc_auc']}"
            )
    print()
    print("Macro task-held-out summary")
    for view_name, view_summary in summary.items():
        item = view_summary["binary_macro"]
        print(
            f"  {view_name}: "
            f"BA={item['balanced_accuracy']:.4f}, "
            f"F1={item['f1_violation']:.4f}, "
            f"recall={item['recall_violation']:.4f}, "
            f"specificity={item['specificity_allow']:.4f}, "
            f"AUC={item['roc_auc']:.4f}, "
            f"AP={item['average_precision']:.4f}, "
            f"control_specificity="
            f"{view_summary['allow_only_control_specificity']:.4f}"
        )
    print()
    print("PASS: Architecture selection used validation only.")
    print("PASS: Calibration and test were excluded from fitting and selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
