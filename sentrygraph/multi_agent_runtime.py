"""Runtime authority registry for delegated SentryGraph agents."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from sentrygraph.contract_compiler import (
    EffectClass,
    TaskContract,
    ToolAuthorization,
)
from sentrygraph.multi_agent_contracts import (
    DelegationContract,
    DelegationDecision,
    evaluate_delegation_request,
)
from sentrygraph.multi_agent_execution_graph import (
    MultiAgentExecutionGraph,
)
from sentrygraph.policy_gate import (
    Decision,
    DeterministicPolicyGate,
    PolicyDecision,
)


class MultiAgentRuntimeError(ValueError):
    """Raised when runtime delegation state is internally inconsistent."""


@dataclass
class AgentRuntimeContext:
    """One principal bound to one active task contract."""

    principal: str
    contract: TaskContract
    delegation_id: str | None
    parent_principal: str | None
    depth: int
    max_depth: int
    active: bool = True
    revoked_reason: str | None = None


@dataclass(frozen=True)
class ScopedActionDecision:
    """Pre-effect decision for an action in a principal's authority scope."""

    action_id: str
    principal: str
    contract_id: str | None
    delegation_id: str | None
    step: int
    policy: PolicyDecision

    @property
    def permitted(self) -> bool:
        return self.policy.decision == Decision.ALLOW


def _request_identifier(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "delegation-request-" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:16]


def _authorization_payload(
    authorization: ToolAuthorization,
) -> dict[str, Any]:
    """Return stable JSON data without requiring a ``to_dict`` method."""

    return {
        "tool": authorization.tool,
        "effect": authorization.effect.value,
        "constraints": [
            {
                "argument": constraint.argument,
                "operator": getattr(
                    constraint.operator,
                    "value",
                    str(constraint.operator),
                ),
                "value": constraint.value,
            }
            for constraint in authorization.constraints
        ],
        "allow_additional_arguments": (
            authorization.allow_additional_arguments
        ),
        "approval_required": authorization.approval_required,
    }


class MultiAgentAuthorityRuntime:
    """Bind principals to attenuated contracts before any tool execution.

    This component owns delegation lifecycle and pre-effect authorization. It
    deliberately does not execute tools; an adapter must call ``propose_action``
    first and execute only when ``ScopedActionDecision.permitted`` is true.
    """

    def __init__(
        self,
        *,
        root_contract: TaskContract,
        trajectory_id: str | None = None,
        execution_graph: MultiAgentExecutionGraph | None = None,
        denied_tools: set[str] | None = None,
    ) -> None:
        if not root_contract.principal.strip():
            raise MultiAgentRuntimeError(
                "Root contract principal must be non-empty."
            )

        self.root_contract = root_contract
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.execution_graph = execution_graph or MultiAgentExecutionGraph(
            trajectory_id=self.trajectory_id,
            principal=root_contract.principal,
        )
        self.denied_tools = frozenset(denied_tools or set())
        self.step = 0
        self.contexts: dict[str, AgentRuntimeContext] = {}
        self.delegations: dict[str, DelegationContract] = {}

        root_context = AgentRuntimeContext(
            principal=root_contract.principal,
            contract=root_contract,
            delegation_id=None,
            parent_principal=None,
            depth=0,
            max_depth=0,
        )
        self.contexts[root_contract.principal] = root_context
        self._record_root_scope(root_context)

    def _next_step(self) -> int:
        self.step += 1
        return self.step

    def _record_root_scope(self, context: AgentRuntimeContext) -> None:
        principal_node = f"principal:{context.principal}"
        contract_node = f"contract:{context.contract.contract_id}"
        self.execution_graph.add_node(
            node_id=principal_node,
            node_type="principal",
            attributes={"name": context.principal, "status": "active"},
        )
        self.execution_graph.add_node(
            node_id=contract_node,
            node_type="task_contract",
            attributes={
                "contract_id": context.contract.contract_id,
                "contract_fingerprint": context.contract.fingerprint(),
                "principal": context.principal,
                "root": True,
            },
        )
        self.execution_graph.add_edge(
            source=contract_node,
            target=principal_node,
            relation="SCOPES",
            step=0,
        )

    def context_for(self, principal: str) -> AgentRuntimeContext | None:
        return self.contexts.get(principal)

    def active_contract_for(self, principal: str) -> TaskContract | None:
        context = self.context_for(principal)
        if context is None or not context.active:
            return None
        return context.contract

    def request_delegation(
        self,
        *,
        parent_principal: str,
        child_principal: str,
        objective: str,
        permitted_tools: tuple[ToolAuthorization, ...],
        additional_prohibited_tools: Iterable[str] = (),
        additional_prohibited_effects: Iterable[EffectClass] = (),
        child_delegation_allowed: bool = False,
        max_depth: int | None = None,
    ) -> DelegationDecision:
        """Evaluate, graph, and activate one attenuated handoff."""

        parent = self.context_for(parent_principal)
        if parent is None or not parent.active:
            return DelegationDecision(
                allowed=False,
                rule_id="DELEGATION_PARENT_CONTEXT_INACTIVE",
                reason=(
                    f"Parent principal {parent_principal!r} has no active "
                    "runtime authority context."
                ),
            )

        existing = self.context_for(child_principal)
        if existing is not None and existing.active:
            return DelegationDecision(
                allowed=False,
                rule_id="DELEGATION_CHILD_ALREADY_ACTIVE",
                reason=(
                    f"Child principal {child_principal!r} already has an "
                    "active authority context."
                ),
            )

        depth = parent.depth + 1
        if parent.delegation_id is None:
            effective_max_depth = max_depth if max_depth is not None else 1
        else:
            effective_max_depth = parent.max_depth
            if max_depth is not None and max_depth != effective_max_depth:
                return DelegationDecision(
                    allowed=False,
                    rule_id="DELEGATION_DEPTH_POLICY_MISMATCH",
                    reason=(
                        "Subdelegation cannot replace the inherited maximum "
                        "delegation depth."
                    ),
                )

        if depth > effective_max_depth:
            return DelegationDecision(
                allowed=False,
                rule_id="DELEGATION_DEPTH_EXCEEDED",
                reason=(
                    f"Requested delegation depth {depth} exceeds the "
                    f"inherited maximum depth {effective_max_depth}."
                ),
            )

        request_payload = {
            "parent_contract_fingerprint": parent.contract.fingerprint(),
            "parent_principal": parent_principal,
            "child_principal": child_principal,
            "objective": objective,
            "permitted_tools": [
                _authorization_payload(authorization)
                for authorization in permitted_tools
            ],
            "depth": depth,
            "max_depth": effective_max_depth,
            "parent_delegation_id": parent.delegation_id,
        }
        request_id = _request_identifier(request_payload)

        decision = evaluate_delegation_request(
            parent_contract=parent.contract,
            child_principal=child_principal,
            objective=objective,
            permitted_tools=permitted_tools,
            additional_prohibited_tools=additional_prohibited_tools,
            additional_prohibited_effects=additional_prohibited_effects,
            child_delegation_allowed=child_delegation_allowed,
            depth=depth,
            max_depth=effective_max_depth,
            parent_delegation_id=parent.delegation_id,
        )
        delegation_id = (
            decision.delegation.delegation_id
            if decision.delegation is not None
            else request_id
        )
        step = self._next_step()
        self.execution_graph.record_delegation_proposal(
            delegation_id=delegation_id,
            parent_principal=parent_principal,
            child_principal=child_principal,
            parent_contract_id=parent.contract.contract_id,
            parent_contract_fingerprint=parent.contract.fingerprint(),
            objective=objective,
            depth=depth,
            max_depth=effective_max_depth,
            parent_delegation_id=parent.delegation_id,
            step=step,
        )
        self.execution_graph.record_delegation_decision(
            delegation_id=delegation_id,
            decision=decision,
            step=step,
        )

        if decision.allowed:
            delegation = decision.delegation
            if delegation is None:
                raise MultiAgentRuntimeError(
                    "Allowed delegation has no compiled contract."
                )
            self.delegations[delegation.delegation_id] = delegation
            self.contexts[child_principal] = AgentRuntimeContext(
                principal=child_principal,
                contract=delegation.child_contract,
                delegation_id=delegation.delegation_id,
                parent_principal=parent_principal,
                depth=delegation.depth,
                max_depth=delegation.max_depth,
            )

        return decision

    def _context_denial(
        self,
        principal: str,
        context: AgentRuntimeContext | None,
    ) -> PolicyDecision:
        if context is None:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="AGENT_CONTEXT_UNKNOWN",
                reason=(
                    f"Principal {principal!r} has no runtime authority "
                    "context."
                ),
                contract_id=None,
            )
        return PolicyDecision(
            decision=Decision.DENY,
            rule_id="AGENT_CONTEXT_INACTIVE",
            reason=(
                f"Principal {principal!r} has an inactive runtime authority "
                "context."
            ),
            contract_id=context.contract.contract_id,
        )

    def _record_unscoped_denial(
        self,
        *,
        action_id: str,
        principal: str,
        tool: str,
        arguments: dict[str, Any],
        policy: PolicyDecision,
        context: AgentRuntimeContext | None,
        step: int,
    ) -> None:
        """Preserve evidence even when no active scoped context exists."""

        principal_node = f"principal:{principal}"
        action_node = f"action:{action_id}"
        tool_node = f"tool:{tool}"
        decision_node = f"decision:{action_id}"
        self.execution_graph.add_node(
            node_id=principal_node,
            node_type="principal",
            attributes={
                "name": principal,
                "status": "unknown" if context is None else "revoked",
            },
        )
        self.execution_graph.add_node(
            node_id=tool_node,
            node_type="tool",
            attributes={"name": tool},
        )
        self.execution_graph.add_node(
            node_id=action_node,
            node_type="proposed_action",
            attributes={
                "action_id": action_id,
                "principal": principal,
                "contract_id": policy.contract_id,
                "tool": tool,
                "arguments": arguments,
                "step": step,
                "status": "denied",
            },
        )
        self.execution_graph.add_node(
            node_id=decision_node,
            node_type="policy_decision",
            attributes={
                "decision": policy.decision.value,
                "rule_id": policy.rule_id,
                "reason": policy.reason,
            },
        )
        self.execution_graph.add_edge(
            source=principal_node,
            target=action_node,
            relation="PROPOSED",
            step=step,
        )
        self.execution_graph.add_edge(
            source=action_node,
            target=tool_node,
            relation="INVOKES",
            step=step,
        )
        self.execution_graph.add_edge(
            source=action_node,
            target=decision_node,
            relation="EVALUATED_AS",
            step=step,
        )
        self.execution_graph.add_edge(
            source=decision_node,
            target=action_node,
            relation="BLOCKS",
            step=step,
        )
        if context is not None:
            contract_node = f"contract:{context.contract.contract_id}"
            self.execution_graph.add_edge(
                source=contract_node,
                target=action_node,
                relation="GOVERNS",
                step=step,
            )

    def propose_action(
        self,
        *,
        principal: str,
        tool: str,
        arguments: dict[str, Any],
        action_id: str | None = None,
    ) -> ScopedActionDecision:
        """Authorize and graph one action before any external effect."""

        action_id = action_id or str(uuid.uuid4())
        step = self._next_step()
        context = self.context_for(principal)

        if context is None or not context.active:
            policy = self._context_denial(principal, context)
            self._record_unscoped_denial(
                action_id=action_id,
                principal=principal,
                tool=tool,
                arguments=arguments,
                policy=policy,
                context=context,
                step=step,
            )
            return ScopedActionDecision(
                action_id=action_id,
                principal=principal,
                contract_id=policy.contract_id,
                delegation_id=(
                    context.delegation_id if context is not None else None
                ),
                step=step,
                policy=policy,
            )

        gate = DeterministicPolicyGate(
            denied_tools=set(self.denied_tools),
            contract=context.contract,
        )
        policy = gate.evaluate(tool=tool, arguments=arguments)
        self.execution_graph.record_agent_proposal(
            action_id=action_id,
            principal=principal,
            contract_id=context.contract.contract_id,
            delegation_id=context.delegation_id,
            tool=tool,
            arguments=arguments,
            step=step,
        )
        self.execution_graph.record_decision(
            action_id=action_id,
            decision=policy.decision.value,
            rule_id=policy.rule_id,
            reason=policy.reason,
            contract_id=policy.contract_id,
            step=step,
        )
        if not policy.permitted:
            self.execution_graph.record_denial(
                action_id=action_id,
                step=step,
            )

        return ScopedActionDecision(
            action_id=action_id,
            principal=principal,
            contract_id=context.contract.contract_id,
            delegation_id=context.delegation_id,
            step=step,
            policy=policy,
        )

    def record_completion(
        self,
        *,
        decision: ScopedActionDecision,
        result: Any,
        error: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record an effect result only for a previously allowed action."""

        if not decision.permitted:
            raise MultiAgentRuntimeError(
                "A denied action cannot be recorded as completed."
            )
        self.execution_graph.record_completion(
            action_id=decision.action_id,
            result=result,
            error=error,
            latency_ms=latency_ms,
            step=decision.step,
        )

    def revoke_delegation(
        self,
        *,
        delegation_id: str,
        reason: str,
    ) -> tuple[str, ...]:
        """Revoke one delegation and recursively deactivate descendants."""

        delegation = self.delegations.get(delegation_id)
        if delegation is None:
            raise MultiAgentRuntimeError(
                f"Unknown accepted delegation: {delegation_id}."
            )

        affected: list[str] = []
        queue = [delegation.child_principal]
        step = self._next_step()
        revocation_node = f"revocation:{delegation_id}:{step}"
        delegation_node = f"delegation:{delegation_id}"
        self.execution_graph.add_node(
            node_id=revocation_node,
            node_type="delegation_revocation",
            attributes={
                "delegation_id": delegation_id,
                "reason": reason,
                "status": "effective",
            },
        )
        self.execution_graph.add_edge(
            source=revocation_node,
            target=delegation_node,
            relation="REVOKES",
            step=step,
            attributes={"reason": reason},
        )

        while queue:
            principal = queue.pop(0)
            context = self.contexts[principal]
            if not context.active:
                continue
            context.active = False
            context.revoked_reason = reason
            affected.append(principal)
            principal_node = f"principal:{principal}"
            self.execution_graph.nodes[principal_node].attributes[
                "status"
            ] = "revoked"
            self.execution_graph.add_edge(
                source=revocation_node,
                target=principal_node,
                relation="DEACTIVATES",
                step=step,
            )
            for candidate in self.contexts.values():
                if candidate.parent_principal == principal:
                    queue.append(candidate.principal)

        return tuple(affected)
