"""Deterministic feature extraction for SentryGraph conformance examples."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.feature_extraction import DictVectorizer


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def collection_size(value: Any) -> int:
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value)
    return 0


def canonical_value_token(value: Any) -> str:
    """Represent values without Python object identity or ordering noise."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )


def graph_index(
    graph: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    nodes = {
        str(node["node_id"]): node
        for node in graph.get("nodes", [])
    }
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        outgoing[str(edge["source"])].append(edge)
    return nodes, outgoing


def example_to_feature_dict(example: Mapping[str, Any]) -> dict[str, Any]:
    """Extract features without reading labels, split, group, or source IDs."""

    features = example["features"]
    contract = features["contract_graph"]
    prefix = features["execution_prefix"]
    candidate = features["candidate_action"]
    candidate_tool = str(candidate["tool"])
    candidate_arguments = dict(candidate.get("arguments", {}))
    output: dict[str, Any] = {}

    output["candidate.tool"] = candidate_tool
    output["candidate.step"] = float(candidate.get("step", 0))
    output["candidate.argument_count"] = float(len(candidate_arguments))
    output["candidate.null_argument_count"] = float(
        sum(value is None for value in candidate_arguments.values())
    )

    for name, value in sorted(candidate_arguments.items()):
        output[f"candidate.argument.present::{name}"] = 1.0
        output[f"candidate.argument.type::{name}"] = value_type(value)
        output[f"candidate.argument.size::{name}"] = float(collection_size(value))

    contract_nodes, contract_outgoing = graph_index(contract)
    contract_node_types = Counter(
        str(node.get("node_type")) for node in contract.get("nodes", [])
    )
    contract_relations = Counter(
        str(edge.get("relation")) for edge in contract.get("edges", [])
    )

    for node_type, count in sorted(contract_node_types.items()):
        output[f"contract.node_type::{node_type}"] = float(count)
    for relation, count in sorted(contract_relations.items()):
        output[f"contract.relation::{relation}"] = float(count)

    matching_tool_nodes: list[str] = []
    for node_id, node in contract_nodes.items():
        node_type = node.get("node_type")
        attributes = dict(node.get("attributes", {}))

        if node_type == "tool":
            name = str(attributes.get("name"))
            authorization = str(attributes.get("authorization", "unknown"))
            output[f"contract.tool::{authorization}::{name}"] = 1.0
            if name == candidate_tool:
                matching_tool_nodes.append(node_id)
                output[f"candidate.contract_authorization::{authorization}"] = 1.0
                output["candidate.approval_required"] = float(
                    bool(attributes.get("approval_required", False))
                )
                output["candidate.allow_additional_arguments"] = float(
                    bool(attributes.get("allow_additional_arguments", True))
                )

        elif node_type == "effect":
            authorization = str(attributes.get("authorization", "unknown"))
            name = str(attributes.get("name"))
            output[f"contract.effect::{authorization}::{name}"] = 1.0

        elif node_type == "argument_constraint":
            argument = str(attributes.get("argument"))
            operator = str(attributes.get("operator"))
            output[f"contract.constraint.argument::{argument}"] = (
                output.get(f"contract.constraint.argument::{argument}", 0.0) + 1.0
            )
            output[f"contract.constraint.operator::{operator}"] = (
                output.get(f"contract.constraint.operator::{operator}", 0.0) + 1.0
            )

    matched_constraints = 0
    for tool_node_id in matching_tool_nodes:
        for edge in contract_outgoing.get(tool_node_id, []):
            target = contract_nodes.get(str(edge.get("target")))
            if target is None:
                continue
            relation = str(edge.get("relation"))
            attributes = dict(target.get("attributes", {}))

            if relation == "HAS_EFFECT":
                output["candidate.effect"] = str(attributes.get("name"))

            if relation != "CONSTRAINS":
                continue

            matched_constraints += 1
            argument = str(attributes.get("argument"))
            operator = str(attributes.get("operator"))
            expected = attributes.get("value")
            present = argument in candidate_arguments
            observed = candidate_arguments.get(argument)

            output[f"candidate.constraint.operator::{operator}"] = (
                output.get(f"candidate.constraint.operator::{operator}", 0.0) + 1.0
            )
            output[f"candidate.constraint.argument::{argument}"] = (
                output.get(f"candidate.constraint.argument::{argument}", 0.0) + 1.0
            )
            output[f"candidate.constraint.present::{argument}"] = float(present)
            output[f"candidate.constraint.expected_type::{argument}"] = value_type(
                expected
            )
            output[f"candidate.constraint.observed_type::{argument}"] = value_type(
                observed
            )
            output[f"candidate.constraint.expected_size::{argument}"] = float(
                collection_size(expected)
            )
            output[f"candidate.constraint.observed_size::{argument}"] = float(
                collection_size(observed)
            )

            # Raw values are categorical graph attributes. DictVectorizer fits
            # their vocabulary on training only; unseen validation/test values
            # are ignored rather than added to the feature space.
            output[f"candidate.constraint.expected_value::{argument}"] = (
                canonical_value_token(expected)
            )
            if present:
                output[f"candidate.constraint.observed_value::{argument}"] = (
                    canonical_value_token(observed)
                )

    output["candidate.matching_contract_tool_count"] = float(
        len(matching_tool_nodes)
    )
    output["candidate.constraint_count"] = float(matched_constraints)

    prefix_node_types = Counter(
        str(node.get("node_type")) for node in prefix.get("nodes", [])
    )
    prefix_relations = Counter(
        str(edge.get("relation")) for edge in prefix.get("edges", [])
    )
    output["prefix.node_count"] = float(len(prefix.get("nodes", [])))
    output["prefix.edge_count"] = float(len(prefix.get("edges", [])))

    for node_type, count in sorted(prefix_node_types.items()):
        output[f"prefix.node_type::{node_type}"] = float(count)
    for relation, count in sorted(prefix_relations.items()):
        output[f"prefix.relation::{relation}"] = float(count)

    previous_tools = Counter()
    previous_completed = 0
    previous_resources = Counter()
    for node in prefix.get("nodes", []):
        node_type = node.get("node_type")
        attributes = dict(node.get("attributes", {}))
        if node_type == "proposed_action":
            previous_tools[str(attributes.get("tool"))] += 1
        elif node_type == "execution_outcome":
            previous_completed += 1
        elif node_type == "resource":
            previous_resources[str(attributes.get("kind"))] += 1

    for tool, count in sorted(previous_tools.items()):
        output[f"prefix.tool::{tool}"] = float(count)
    for kind, count in sorted(previous_resources.items()):
        output[f"prefix.resource_kind::{kind}"] = float(count)
    output["prefix.completed_actions"] = float(previous_completed)
    output["prefix.same_tool_actions"] = float(previous_tools[candidate_tool])

    return output


class GraphFeatureEncoder:
    """Train-only-fitted sparse categorical/numeric graph encoder."""

    def __init__(self) -> None:
        self.vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
        self._fitted = False

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "GraphFeatureEncoder":
        dictionaries = [example_to_feature_dict(example) for example in examples]
        if not dictionaries:
            raise ValueError("Cannot fit an encoder on an empty dataset.")
        self.vectorizer.fit(dictionaries)
        self._fitted = True
        return self

    def transform(self, examples: Iterable[Mapping[str, Any]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GraphFeatureEncoder must be fitted first.")
        dictionaries = [example_to_feature_dict(example) for example in examples]
        if not dictionaries:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.asarray(self.vectorizer.transform(dictionaries), dtype=np.float32)

    def fit_transform(self, examples: Iterable[Mapping[str, Any]]) -> np.ndarray:
        materialized = list(examples)
        self.fit(materialized)
        return self.transform(materialized)

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("GraphFeatureEncoder must be fitted first.")
        return list(self.vectorizer.get_feature_names_out())

    @property
    def vocabulary(self) -> dict[str, int]:
        if not self._fitted:
            raise RuntimeError("GraphFeatureEncoder must be fitted first.")
        return dict(sorted(self.vectorizer.vocabulary_.items()))

