"""Multi-agent delegation and handoff provenance for SentryGraph."""

from __future__ import annotations

from typing import Any

from sentrygraph.execution_graph import ExecutionGraph, utc_now
from sentrygraph.multi_agent_contracts import DelegationDecision


class MultiAgentExecutionGraph(ExecutionGraph):
    """Execution graph with fail-closed delegation-chain provenance."""

    def record_delegation_proposal(
        self,
        *,
        delegation_id: str,
        parent_principal: str,
        child_principal: str,
        parent_contract_id: str,
        parent_contract_fingerprint: str,
        objective: str,
        depth: int,
        max_depth: int,
        step: int,
        parent_delegation_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        timestamp = timestamp or utc_now()

        if depth < 1 or max_depth < 1 or depth > max_depth:
            raise ValueError(
                "Delegation proposal must satisfy "
                "1 <= depth <= max_depth."
            )

        if parent_principal == child_principal:
            raise ValueError(
                "Delegation parent and child principals must differ."
            )

        parent_id = f"principal:{parent_principal}"
        child_id = f"principal:{child_principal}"
        parent_contract_node = f"contract:{parent_contract_id}"
        delegation_node = f"delegation:{delegation_id}"

        self.add_node(
            node_id=parent_id,
            node_type="principal",
            attributes={"name": parent_principal},
        )
        self.add_node(
            node_id=child_id,
            node_type="principal",
            attributes={
                "name": child_principal,
                "status": "delegation_proposed",
            },
        )
        self.add_node(
            node_id=parent_contract_node,
            node_type="task_contract",
            attributes={
                "contract_id": parent_contract_id,
                "contract_fingerprint": parent_contract_fingerprint,
                "principal": parent_principal,
            },
        )
        self.add_node(
            node_id=delegation_node,
            node_type="delegation",
            attributes={
                "delegation_id": delegation_id,
                "parent_principal": parent_principal,
                "child_principal": child_principal,
                "parent_contract_id": parent_contract_id,
                "objective": objective,
                "depth": depth,
                "max_depth": max_depth,
                "parent_delegation_id": parent_delegation_id,
                "status": "proposed",
            },
        )
        self.add_edge(
            source=parent_id,
            target=delegation_node,
            relation="HANDOFF_PROPOSED",
            step=step,
            timestamp=timestamp,
        )
        self.add_edge(
            source=parent_contract_node,
            target=delegation_node,
            relation="GOVERNS",
            step=step,
            timestamp=timestamp,
        )

        if parent_delegation_id is not None:
            parent_delegation_node = (
                f"delegation:{parent_delegation_id}"
            )
            if parent_delegation_node not in self.nodes:
                raise ValueError(
                    "Parent delegation is absent from the execution graph: "
                    f"{parent_delegation_id}."
                )
            self.add_edge(
                source=parent_delegation_node,
                target=delegation_node,
                relation="SUBDELEGATES_TO",
                step=step,
                timestamp=timestamp,
                attributes={"depth": depth},
            )

    def record_delegation_decision(
        self,
        *,
        delegation_id: str,
        decision: DelegationDecision,
        step: int,
        timestamp: str | None = None,
    ) -> None:
        timestamp = timestamp or utc_now()
        delegation_node = f"delegation:{delegation_id}"

        if delegation_node not in self.nodes:
            raise ValueError(
                f"Delegation {delegation_id!r} has not been proposed."
            )

        proposal = self.nodes[delegation_node].attributes
        decision_node = f"delegation_decision:{delegation_id}"

        self.add_node(
            node_id=decision_node,
            node_type="delegation_decision",
            attributes={
                "allowed": decision.allowed,
                "rule_id": decision.rule_id,
                "reason": decision.reason,
            },
        )
        self.add_edge(
            source=delegation_node,
            target=decision_node,
            relation="EVALUATED_AS",
            step=step,
            timestamp=timestamp,
        )

        if not decision.allowed:
            proposal["status"] = "denied"
            child_id = f"principal:{proposal['child_principal']}"
            self.nodes[child_id].attributes["status"] = "delegation_denied"
            self.add_edge(
                source=decision_node,
                target=delegation_node,
                relation="HANDOFF_DENIED",
                step=step,
                timestamp=timestamp,
            )
            self.add_edge(
                source=decision_node,
                target=delegation_node,
                relation="BLOCKS",
                step=step,
                timestamp=timestamp,
            )
            return

        delegation = decision.delegation
        if delegation is None:
            raise ValueError(
                "Allowed delegation decision has no compiled contract."
            )

        if delegation.delegation_id != delegation_id:
            raise ValueError(
                "Decision delegation ID does not match the proposal."
            )

        expected = {
            "parent_principal": delegation.parent_principal,
            "child_principal": delegation.child_principal,
            "parent_contract_id": delegation.parent_contract_id,
            "depth": delegation.depth,
            "max_depth": delegation.max_depth,
            "parent_delegation_id": delegation.parent_delegation_id,
        }
        for key, value in expected.items():
            if proposal.get(key) != value:
                raise ValueError(
                    f"Accepted delegation does not match proposed {key}."
                )

        parent_id = f"principal:{delegation.parent_principal}"
        child_id = f"principal:{delegation.child_principal}"
        parent_contract_node = (
            f"contract:{delegation.parent_contract_id}"
        )
        child_contract = delegation.child_contract
        child_contract_node = f"contract:{child_contract.contract_id}"

        self.add_node(
            node_id=child_contract_node,
            node_type="task_contract",
            attributes={
                "contract_id": child_contract.contract_id,
                "contract_fingerprint": child_contract.fingerprint(),
                "principal": child_contract.principal,
                "delegation_id": delegation_id,
                "depth": delegation.depth,
                "max_depth": delegation.max_depth,
            },
        )
        self.add_edge(
            source=parent_id,
            target=child_id,
            relation="DELEGATES_TO",
            step=step,
            timestamp=timestamp,
            attributes={
                "delegation_id": delegation_id,
                "depth": delegation.depth,
            },
        )
        self.add_edge(
            source=parent_contract_node,
            target=child_contract_node,
            relation="DERIVES_CONTRACT",
            step=step,
            timestamp=timestamp,
            attributes={"delegation_id": delegation_id},
        )
        self.add_edge(
            source=child_contract_node,
            target=child_id,
            relation="SCOPES",
            step=step,
            timestamp=timestamp,
        )
        self.add_edge(
            source=decision_node,
            target=child_id,
            relation="HANDOFF_ACCEPTED",
            step=step,
            timestamp=timestamp,
        )
        proposal["status"] = "accepted"
        proposal["child_contract_id"] = child_contract.contract_id
        proposal["child_contract_fingerprint"] = (
            child_contract.fingerprint()
        )
        self.nodes[child_id].attributes["status"] = "active"

    def record_agent_proposal(
        self,
        *,
        action_id: str,
        principal: str,
        contract_id: str,
        tool: str,
        arguments: dict[str, Any],
        step: int,
        delegation_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Record a tool action attributed to one scoped agent contract."""

        timestamp = timestamp or utc_now()
        principal_id = f"principal:{principal}"
        contract_node = f"contract:{contract_id}"

        if principal_id not in self.nodes:
            raise ValueError(f"Unknown action principal: {principal}")
        if contract_node not in self.nodes:
            raise ValueError(f"Unknown action contract: {contract_id}")

        contract_principal = self.nodes[contract_node].attributes.get(
            "principal"
        )
        if contract_principal != principal:
            raise ValueError(
                "Action principal is not scoped by the supplied contract."
            )

        delegation_node = None
        if delegation_id is not None:
            delegation_node = f"delegation:{delegation_id}"
            if delegation_node not in self.nodes:
                raise ValueError(
                    f"Unknown action delegation: {delegation_id}"
                )
            delegation = self.nodes[delegation_node].attributes
            if delegation.get("status") != "accepted":
                raise ValueError(
                    "Actions cannot execute under an unaccepted delegation."
                )
            if delegation.get("child_principal") != principal:
                raise ValueError(
                    "Action principal does not match delegation child."
                )
            if delegation.get("child_contract_id") != contract_id:
                raise ValueError(
                    "Action contract does not match delegated contract."
                )

        tool_id = f"tool:{tool}"
        action_node = f"action:{action_id}"
        self.add_node(
            node_id=tool_id,
            node_type="tool",
            attributes={"name": tool},
        )
        self.add_node(
            node_id=action_node,
            node_type="proposed_action",
            attributes={
                "action_id": action_id,
                "principal": principal,
                "contract_id": contract_id,
                "delegation_id": delegation_id,
                "tool": tool,
                "arguments": arguments,
                "step": step,
                "status": "proposed",
            },
        )
        self.add_edge(
            source=principal_id,
            target=action_node,
            relation="PROPOSED",
            step=step,
            timestamp=timestamp,
        )
        self.add_edge(
            source=action_node,
            target=tool_id,
            relation="INVOKES",
            step=step,
            timestamp=timestamp,
        )
        self.add_edge(
            source=contract_node,
            target=action_node,
            relation="GOVERNS",
            step=step,
            timestamp=timestamp,
        )
        if delegation_node is not None:
            self.add_edge(
                source=delegation_node,
                target=action_node,
                relation="DELEGATED_UNDER",
                step=step,
                timestamp=timestamp,
            )
        self._record_resource_relations(
            action_node_id=action_node,
            arguments=arguments,
            step=step,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        value = super().to_dict()
        value["schema_version"] = "0.2"
        value["multi_agent"] = True
        return value

