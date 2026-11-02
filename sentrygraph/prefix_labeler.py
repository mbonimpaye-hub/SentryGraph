"""Leakage-resistant labels and graph prefixes for conformance learning.

Labels are recomputed from frozen contract semantics. Recorded policy decisions,
rules, reasons, denial outcomes, and BLOCKS relations are never used as inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEAKAGE_NODE_TYPES = frozenset({"policy_decision", "task_contract"})
LEAKAGE_RELATIONS = frozenset({"EVALUATED_AS", "GOVERNS", "BLOCKS"})
LEAKAGE_ACTION_ATTRIBUTES = frozenset({"decision", "reason", "rule_id"})


@dataclass(frozen=True)
class ActionLabel:
    """Contract-derived label for one proposed action."""

    permitted: bool
    label: str
    rule_id: str
    contract_id: str | None
    tool: str
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "label": self.label,
            "rule_id": self.rule_id,
            "contract_id": self.contract_id,
            "tool": self.tool,
            "step": self.step,
        }


@dataclass(frozen=True)
class PrefixLabel:
    """One supervised example for graph-conformance learning."""

    action: ActionLabel
    trajectory_permitted: bool
    trajectory_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "trajectory_permitted": self.trajectory_permitted,
            "trajectory_label": self.trajectory_label,
        }


def _contract_payload(contract_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the inner contract from either supported artifact shape."""

    payload = contract_artifact.get("contract", contract_artifact)

    if not isinstance(payload, Mapping):
        raise TypeError("Contract payload must be a mapping.")

    required = {"permitted_tools", "prohibited_tools", "prohibited_effects"}
    missing = required - set(payload)

    if missing:
        raise ValueError(
            "Contract is missing required fields: "
            + ", ".join(sorted(missing))
        )

    return dict(payload)


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen contract artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Contract artifact is not an object: {path}")

    _contract_payload(payload)
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON-lines file as objects."""

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise TypeError(
                f"Expected an object at {path}:{line_number}."
            )

        records.append(value)

    return records


def proposed_actions(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract proposals without consuming their recorded decisions."""

    proposals: list[dict[str, Any]] = []

    for event in events:
        if event.get("phase") != "proposed":
            continue

        tool = event.get("tool")
        arguments = event.get("arguments", {})
        step = event.get("step")

        if not isinstance(tool, str) or not tool:
            raise ValueError("A proposed event has no valid tool name.")

        if not isinstance(arguments, Mapping):
            raise TypeError(
                f"Arguments for proposed tool {tool!r} must be a mapping."
            )

        if not isinstance(step, int):
            raise TypeError(f"Step for proposed tool {tool!r} is not an int.")

        proposals.append(
            {
                "event_id": event.get("event_id"),
                "trajectory_id": event.get("trajectory_id"),
                "step": step,
                "timestamp": event.get("timestamp"),
                "tool": tool,
                "arguments": dict(arguments),
            }
        )

    return sorted(proposals, key=lambda item: item["step"])


def _normalized_operator(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _contains(container: Any, member: Any) -> bool:
    try:
        return member in container
    except TypeError:
        return False


def _constraint_satisfied(
    constraint: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> bool:
    """Evaluate serialized ArgumentConstraint semantics; unknown rules fail closed."""

    argument = constraint.get("argument")
    operator = _normalized_operator(constraint.get("operator", "equals"))
    expected = constraint.get("value")

    if not isinstance(argument, str) or argument not in arguments:
        return False

    observed = arguments[argument]

    if operator in {"equals", "equal", "eq", "=="}:
        return observed == expected

    if operator in {"not_equals", "not_equal", "ne", "!="}:
        return observed != expected

    if operator in {"in", "one_of"}:
        return _contains(expected, observed)

    if operator in {"not_in", "none_of"}:
        return not _contains(expected, observed)

    if operator in {"contains", "includes"}:
        return _contains(observed, expected)

    if operator in {"contains_all", "includes_all"}:
        if not isinstance(expected, Sequence) or isinstance(expected, str):
            return False
        return all(_contains(observed, item) for item in expected)

    if operator in {"contains_any", "includes_any"}:
        if not isinstance(expected, Sequence) or isinstance(expected, str):
            return False
        return any(_contains(observed, item) for item in expected)

    if operator in {"subset", "subset_of"}:
        try:
            return set(observed).issubset(set(expected))
        except TypeError:
            return False

    if operator in {"superset", "superset_of"}:
        try:
            return set(observed).issuperset(set(expected))
        except TypeError:
            return False

    if operator in {"starts_with", "startswith"}:
        return isinstance(observed, str) and isinstance(expected, str) and observed.startswith(expected)

    if operator in {"ends_with", "endswith"}:
        return isinstance(observed, str) and isinstance(expected, str) and observed.endswith(expected)

    if operator in {"matches", "regex", "regex_fullmatch"}:
        if not isinstance(observed, str) or not isinstance(expected, str):
            return False
        return re.fullmatch(expected, observed) is not None

    if operator in {"less_than", "lt", "<"}:
        try:
            return observed < expected
        except TypeError:
            return False

    if operator in {"less_than_or_equal", "lte", "le", "<="}:
        try:
            return observed <= expected
        except TypeError:
            return False

    if operator in {"greater_than", "gt", ">"}:
        try:
            return observed > expected
        except TypeError:
            return False

    if operator in {"greater_than_or_equal", "gte", "ge", ">="}:
        try:
            return observed >= expected
        except TypeError:
            return False

    return False


def _authorization_for(
    contract: Mapping[str, Any],
    tool: str,
) -> dict[str, Any] | None:
    for authorization in contract.get("permitted_tools", []):
        if isinstance(authorization, Mapping) and authorization.get("tool") == tool:
            return dict(authorization)
    return None


def _arguments_authorized(
    authorization: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> bool:
    constraints = authorization.get("constraints", [])

    if not isinstance(constraints, list):
        return False

    if not all(
        isinstance(constraint, Mapping)
        and _constraint_satisfied(constraint, arguments)
        for constraint in constraints
    ):
        return False

    if bool(authorization.get("allow_additional_arguments", True)):
        return True

    constrained = {
        constraint.get("argument")
        for constraint in constraints
        if isinstance(constraint, Mapping)
        and isinstance(constraint.get("argument"), str)
    }
    unexpected = set(arguments) - constrained

    # Compiler v0.3 treats explicit None for optional schema fields as absent.
    return not any(arguments[name] is not None for name in unexpected)


def label_action(
    contract_artifact: Mapping[str, Any],
    *,
    tool: str,
    arguments: Mapping[str, Any],
    step: int,
) -> ActionLabel:
    """Derive an action label solely from the frozen contract and proposal."""

    contract = _contract_payload(contract_artifact)
    contract_id = contract.get("contract_id")

    if tool in set(contract.get("prohibited_tools", [])):
        return ActionLabel(
            False,
            "violation",
            "CONTRACT_PROHIBITED_TOOL",
            contract_id,
            tool,
            step,
        )

    authorization = _authorization_for(contract, tool)

    if authorization is None:
        return ActionLabel(
            False,
            "violation",
            "TOOL_OUTSIDE_CONTRACT",
            contract_id,
            tool,
            step,
        )

    effect = authorization.get("effect")

    if effect in set(contract.get("prohibited_effects", [])):
        return ActionLabel(
            False,
            "violation",
            "PROHIBITED_EFFECT",
            contract_id,
            tool,
            step,
        )

    if bool(authorization.get("approval_required", False)):
        return ActionLabel(
            False,
            "requires_approval",
            "APPROVAL_REQUIRED",
            contract_id,
            tool,
            step,
        )

    if not _arguments_authorized(authorization, arguments):
        return ActionLabel(
            False,
            "violation",
            "ARGUMENT_CONSTRAINT_VIOLATION",
            contract_id,
            tool,
            step,
        )

    return ActionLabel(
        True,
        "conformant",
        "CONTRACT_ALLOW",
        contract_id,
        tool,
        step,
    )


def label_prefixes(
    contract_artifact: Mapping[str, Any],
    proposals: Iterable[Mapping[str, Any]],
) -> list[PrefixLabel]:
    """Label every action and its cumulative trajectory prefix."""

    labels: list[PrefixLabel] = []
    trajectory_permitted = True

    for proposal in sorted(proposals, key=lambda item: int(item["step"])):
        action = label_action(
            contract_artifact,
            tool=str(proposal["tool"]),
            arguments=dict(proposal.get("arguments", {})),
            step=int(proposal["step"]),
        )
        trajectory_permitted = trajectory_permitted and action.permitted
        labels.append(
            PrefixLabel(
                action=action,
                trajectory_permitted=trajectory_permitted,
                trajectory_label=(
                    "conformant_prefix"
                    if trajectory_permitted
                    else "violating_prefix"
                ),
            )
        )

    return labels


def sanitize_execution_prefix(
    graph: Mapping[str, Any],
    *,
    through_step: int,
) -> dict[str, Any]:
    """Return a temporal prefix stripped of deterministic-policy leakage."""

    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("edges", [])

    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise TypeError("Execution graph nodes and edges must be lists.")

    candidate_nodes: dict[str, dict[str, Any]] = {}

    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        if raw_node.get("node_type") in LEAKAGE_NODE_TYPES:
            continue

        node_id = raw_node.get("node_id")
        if not isinstance(node_id, str):
            continue

        attributes = dict(raw_node.get("attributes", {}))
        node_step = attributes.get("step")

        if isinstance(node_step, int) and node_step > through_step:
            continue

        if raw_node.get("node_type") == "proposed_action":
            attributes = {
                key: value
                for key, value in attributes.items()
                if key not in LEAKAGE_ACTION_ATTRIBUTES and key != "status"
            }

        candidate_nodes[node_id] = {
            "node_id": node_id,
            "node_type": raw_node.get("node_type"),
            "attributes": attributes,
        }

    edges: list[dict[str, Any]] = []
    connected_ids: set[str] = set()

    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            continue
        if raw_edge.get("relation") in LEAKAGE_RELATIONS:
            continue
        edge_step = int(raw_edge.get("step", 0))
        if edge_step > through_step:
            continue
        # At the prediction point, the current proposal is observable but its
        # execution outcome is not. Historical outcomes remain available.
        if (
            raw_edge.get("relation") == "COMPLETED_AS"
            and edge_step >= through_step
        ):
            continue
        if raw_edge.get("source") not in candidate_nodes:
            continue
        if raw_edge.get("target") not in candidate_nodes:
            continue

        edge = dict(raw_edge)
        edges.append(edge)
        connected_ids.add(str(edge["source"]))
        connected_ids.add(str(edge["target"]))

    principal_ids = {
        node_id
        for node_id, node in candidate_nodes.items()
        if node.get("node_type") == "principal"
    }
    retained_ids = connected_ids | principal_ids
    nodes = [
        node
        for node_id, node in candidate_nodes.items()
        if node_id in retained_ids
    ]

    return {
        "schema_version": "0.1",
        "trajectory_id": graph.get("trajectory_id"),
        "principal": graph.get("principal"),
        "prefix_through_step": through_step,
        "nodes": nodes,
        "edges": edges,
    }


def load_graph(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Execution graph is not an object: {path}")
    return payload


def recorded_proposal_decisions(
    events: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, str], str]:
    """Return recorded proposal decisions for audit tests only.

    Dataset construction must not include this output in features or labels.
    """

    decisions: dict[tuple[int, str], str] = {}

    for event in events:
        if event.get("phase") != "proposed":
            continue
        decision = event.get("decision")
        if isinstance(decision, str):
            decisions[(int(event["step"]), str(event["tool"]))] = decision

    return decisions
