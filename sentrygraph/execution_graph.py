"""Heterogeneous temporal execution graph for SentryGraph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_identifier(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))

    return repr(value)


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalEdge:
    source: str
    target: str
    relation: str
    step: int
    timestamp: str
    attributes: dict[str, Any] = field(default_factory=dict)


class ExecutionGraph:
    """Maintain an evolving heterogeneous execution graph."""

    RESOURCE_ARGUMENTS = {
        "filename",
        "file_id",
        "email",
        "recipient",
        "recipients",
        "to",
        "event_id",
        "calendar_id",
        "channel",
        "channel_id",
        "account",
        "account_id",
        "destination",
        "url",
    }

    def __init__(
        self,
        trajectory_id: str,
        principal: str = "agent",
    ) -> None:
        self.trajectory_id = trajectory_id
        self.principal = principal
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[TemporalEdge] = []

        self.add_node(
            node_id=f"principal:{principal}",
            node_type="principal",
            attributes={"name": principal},
        )

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> GraphNode:
        attributes = json_safe(attributes or {})

        if node_id in self.nodes:
            existing = self.nodes[node_id]

            if existing.node_type != node_type:
                raise ValueError(
                    f"Node '{node_id}' already exists with type "
                    f"'{existing.node_type}', not '{node_type}'."
                )

            existing.attributes.update(attributes)
            return existing

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            attributes=attributes,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        *,
        source: str,
        target: str,
        relation: str,
        step: int,
        timestamp: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TemporalEdge:
        if source not in self.nodes:
            raise ValueError(f"Unknown source node: {source}")

        if target not in self.nodes:
            raise ValueError(f"Unknown target node: {target}")

        edge = TemporalEdge(
            source=source,
            target=target,
            relation=relation,
            step=step,
            timestamp=timestamp or utc_now(),
            attributes=json_safe(attributes or {}),
        )
        self.edges.append(edge)
        return edge

    def record_proposal(
        self,
        *,
        action_id: str,
        tool: str,
        arguments: dict[str, Any],
        step: int,
        timestamp: str | None = None,
    ) -> None:
        timestamp = timestamp or utc_now()

        principal_id = f"principal:{self.principal}"
        tool_id = f"tool:{tool}"
        action_node_id = f"action:{action_id}"

        self.add_node(
            node_id=tool_id,
            node_type="tool",
            attributes={"name": tool},
        )

        self.add_node(
            node_id=action_node_id,
            node_type="proposed_action",
            attributes={
                "action_id": action_id,
                "tool": tool,
                "arguments": arguments,
                "step": step,
                "status": "proposed",
            },
        )

        self.add_edge(
            source=principal_id,
            target=action_node_id,
            relation="PROPOSED",
            step=step,
            timestamp=timestamp,
        )

        self.add_edge(
            source=action_node_id,
            target=tool_id,
            relation="INVOKES",
            step=step,
            timestamp=timestamp,
        )

        self._record_resource_relations(
            action_node_id=action_node_id,
            arguments=arguments,
            step=step,
            timestamp=timestamp,
        )

    def _record_resource_relations(
        self,
        *,
        action_node_id: str,
        arguments: dict[str, Any],
        step: int,
        timestamp: str,
    ) -> None:
        for argument_name, value in arguments.items():
            if argument_name not in self.RESOURCE_ARGUMENTS:
                continue

            values = (
                value
                if isinstance(value, (list, tuple, set))
                else [value]
            )

            for resource_value in values:
                normalized = str(resource_value)
                resource_id = stable_identifier(
                    "resource",
                    f"{argument_name}:{normalized}",
                )

                self.add_node(
                    node_id=resource_id,
                    node_type="resource",
                    attributes={
                        "kind": argument_name,
                        "value": normalized,
                    },
                )

                self.add_edge(
                    source=action_node_id,
                    target=resource_id,
                    relation="TARGETS",
                    step=step,
                    timestamp=timestamp,
                    attributes={
                        "argument": argument_name,
                    },
                )

    def record_decision(
        self,
        *,
        action_id: str,
        decision: str,
        rule_id: str,
        reason: str,
        step: int,
        contract_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        timestamp = timestamp or utc_now()
        action_node_id = f"action:{action_id}"

        if action_node_id not in self.nodes:
            raise ValueError(
                f"Action '{action_id}' has not been proposed."
            )

        decision_node_id = f"decision:{action_id}"

        self.add_node(
            node_id=decision_node_id,
            node_type="policy_decision",
            attributes={
                "decision": decision,
                "rule_id": rule_id,
                "reason": reason,
                "contract_id": contract_id,
            },
        )

        self.add_edge(
            source=action_node_id,
            target=decision_node_id,
            relation="EVALUATED_AS",
            step=step,
            timestamp=timestamp,
        )

        self.nodes[action_node_id].attributes["status"] = decision

        if contract_id is not None:
            contract_node_id = f"contract:{contract_id}"

            self.add_node(
                node_id=contract_node_id,
                node_type="task_contract",
                attributes={"contract_id": contract_id},
            )

            self.add_edge(
                source=contract_node_id,
                target=action_node_id,
                relation="GOVERNS",
                step=step,
                timestamp=timestamp,
            )

    def record_completion(
        self,
        *,
        action_id: str,
        result: Any,
        error: Any,
        latency_ms: float,
        step: int,
        timestamp: str | None = None,
    ) -> None:
        action_node_id = f"action:{action_id}"

        if action_node_id not in self.nodes:
            raise ValueError(
                f"Action '{action_id}' has not been proposed."
            )

        outcome_node_id = f"outcome:{action_id}"

        self.add_node(
            node_id=outcome_node_id,
            node_type="execution_outcome",
            attributes={
                "result": json_safe(result),
                "error": json_safe(error),
                "latency_ms": latency_ms,
            },
        )

        self.add_edge(
            source=action_node_id,
            target=outcome_node_id,
            relation="COMPLETED_AS",
            step=step,
            timestamp=timestamp or utc_now(),
        )

        self.nodes[action_node_id].attributes["status"] = "completed"

    def record_denial(
        self,
        *,
        action_id: str,
        step: int,
        timestamp: str | None = None,
    ) -> None:
        action_node_id = f"action:{action_id}"

        if action_node_id not in self.nodes:
            raise ValueError(
                f"Action '{action_id}' has not been proposed."
            )

        self.nodes[action_node_id].attributes["status"] = "denied"

        decision_node_id = f"decision:{action_id}"

        if decision_node_id in self.nodes:
            self.add_edge(
                source=decision_node_id,
                target=action_node_id,
                relation="BLOCKS",
                step=step,
                timestamp=timestamp or utc_now(),
            )

    def relations_for_step(self, step: int) -> list[str]:
        return [
            edge.relation
            for edge in self.edges
            if edge.step == step
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "trajectory_id": self.trajectory_id,
            "principal": self.principal,
            "nodes": [
                asdict(node)
                for node in self.nodes.values()
            ],
            "edges": [
                asdict(edge)
                for edge in self.edges
            ],
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )