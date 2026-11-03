"""Leakage-resistant relational graph tensors for SentryGraph learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import torch


VIEWS = (
    "full_fidelity",
    "structural_generalization",
    "semantic_ablated",
)

UNK_TOKEN = "<UNK>"
UNK_RELATION = "<UNK_RELATION>"


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _attribute_tokens(
    node_type: str,
    attributes: dict[str, Any],
    view: str,
) -> list[str]:
    tokens = [f"node_type::{node_type}"]

    if view == "semantic_ablated":
        tokens.append(f"attribute_count::{len(attributes)}")
        return tokens

    for key in sorted(attributes):
        tokens.append(f"attribute_key::{node_type}::{key}")

        if view == "full_fidelity":
            tokens.append(
                f"attribute_value::{node_type}::{key}="
                f"{canonical(attributes[key])}"
            )

    return tokens


def _candidate_tokens(
    candidate: dict[str, Any],
    view: str,
) -> list[str]:
    arguments = candidate.get("arguments") or {}
    tokens = ["node_type::candidate_action"]

    if view != "semantic_ablated":
        tokens.append(f"candidate_tool::{candidate.get('tool', UNK_TOKEN)}")
        for key in sorted(arguments):
            tokens.append(f"candidate_argument_key::{key}")
            if view == "full_fidelity":
                tokens.append(
                    f"candidate_argument_value::{key}="
                    f"{canonical(arguments[key])}"
                )
    else:
        tokens.append(f"candidate_argument_count::{len(arguments)}")

    return tokens


def graph_symbols(
    example: dict[str, Any],
    view: str,
) -> tuple[list[list[str]], list[tuple[int, int, str]], int]:
    """Return node tokens, typed directed edges, and candidate index.

    Labels, split assignments, source metadata, example IDs, and group IDs are
    deliberately inaccessible to this transformation.
    """

    if view not in VIEWS:
        raise ValueError(f"Unsupported graph view: {view}")

    features = example["features"]
    node_tokens: list[list[str]] = []
    edges: list[tuple[int, int, str]] = []
    identifiers: dict[tuple[str, str], int] = {}

    for namespace, graph_key in (
        ("contract", "contract_graph"),
        ("prefix", "execution_prefix"),
    ):
        graph = features[graph_key]

        for node in graph.get("nodes", []):
            index = len(node_tokens)

            node_identifier = node.get(
                "id",
                node.get("node_id"),
            )

            if node_identifier is None:
                raise ValueError(
                    f"Graph node has no id or node_id: {node}"
                )

            identifiers[
                (namespace, str(node_identifier))
            ] = index

            node_tokens.append(
                _attribute_tokens(
                    str(node.get("type", "unknown")),
                    dict(node.get("attributes") or {}),
                    view,
                )
            )

        for edge in graph.get("edges", []):
            source_key = (namespace, str(edge["source"]))
            target_key = (namespace, str(edge["target"]))

            if source_key not in identifiers or target_key not in identifiers:
                raise ValueError(
                    f"Edge references an unknown node in {graph_key}: {edge}"
                )

            edges.append(
                (
                    identifiers[source_key],
                    identifiers[target_key],
                    str(edge.get("relation", "unknown")),
                )
            )

    candidate_index = len(node_tokens)
    node_tokens.append(
        _candidate_tokens(features["candidate_action"], view)
    )

    contract_roots = [
        index
        for (namespace, identifier), index in identifiers.items()
        if namespace == "contract" and identifier.startswith("contract:")
    ]
    for root in contract_roots:
        edges.append((root, candidate_index, "EVALUATES_CANDIDATE"))

    prefix_indices = [
        index
        for (namespace, _), index in identifiers.items()
        if namespace == "prefix"
    ]
    if prefix_indices:
        edges.append((max(prefix_indices), candidate_index, "NEXT_ACTION"))

    return node_tokens, edges, candidate_index


@dataclass(frozen=True)
class RelationalVocabulary:
    token_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    view: str

    @classmethod
    def fit(
        cls,
        examples: Iterable[dict[str, Any]],
        view: str,
    ) -> "RelationalVocabulary":
        tokens = {UNK_TOKEN}
        relations = {UNK_RELATION}

        for example in examples:
            node_tokens, edges, _ = graph_symbols(example, view)
            for values in node_tokens:
                tokens.update(values)
            relations.update(relation for _, _, relation in edges)

        return cls(
            token_to_id={
                token: index
                for index, token in enumerate(
                    [UNK_TOKEN, *sorted(tokens - {UNK_TOKEN})]
                )
            },
            relation_to_id={
                relation: index
                for index, relation in enumerate(
                    [
                        UNK_RELATION,
                        *sorted(relations - {UNK_RELATION}),
                    ]
                )
            },
            view=view,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "view": self.view,
            "token_to_id": self.token_to_id,
            "relation_to_id": self.relation_to_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationalVocabulary":
        return cls(
            token_to_id={
                str(key): int(item)
                for key, item in value["token_to_id"].items()
            },
            relation_to_id={
                str(key): int(item)
                for key, item in value["relation_to_id"].items()
            },
            view=str(value["view"]),
        )


def encode_example(
    example: dict[str, Any],
    vocabulary: RelationalVocabulary,
) -> dict[str, Any]:
    node_tokens, edges, candidate_index = graph_symbols(
        example,
        vocabulary.view,
    )

    token_rows: list[int] = []
    token_columns: list[int] = []

    for node_index, values in enumerate(node_tokens):
        for token in sorted(set(values)):
            token_rows.append(node_index)
            token_columns.append(
                vocabulary.token_to_id.get(
                    token,
                    vocabulary.token_to_id[UNK_TOKEN],
                )
            )

    if edges:
        edge_index = torch.tensor(
            [[source, target] for source, target, _ in edges],
            dtype=torch.long,
        ).t().contiguous()
        edge_type = torch.tensor(
            [
                vocabulary.relation_to_id.get(
                    relation,
                    vocabulary.relation_to_id[UNK_RELATION],
                )
                for _, _, relation in edges
            ],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    labels = example["labels"]
    source = example["source"]

    return {
        "node_token_index": torch.tensor(
            [token_rows, token_columns],
            dtype=torch.long,
        ),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_nodes": len(node_tokens),
        "candidate_index": candidate_index,
        "action_label": int(not bool(labels["action_permitted"])),
        "trajectory_label": int(
            not bool(labels["trajectory_permitted"])
        ),
        "metadata": {
            "example_id": example["example_id"],
            "group_id": example["group_id"],
            "task_id": source["task_id"],
            "contract_fingerprint": source["contract_fingerprint"],
        },
    }


def tensor_fingerprint(encoded: dict[str, Any]) -> str:
    payload = {
        "node_token_index": encoded["node_token_index"].tolist(),
        "edge_index": encoded["edge_index"].tolist(),
        "edge_type": encoded["edge_type"].tolist(),
        "num_nodes": encoded["num_nodes"],
        "candidate_index": encoded["candidate_index"],
    }
    return stable_hash(payload)

