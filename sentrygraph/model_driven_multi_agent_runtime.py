"""Runtime helpers for model-driven multi-agent SentryGraph experiments."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from agentdojo.functions_runtime import FunctionsRuntime

from sentrygraph.agentdojo_adapter import utc_now
from sentrygraph.contract_compiler import (
    EffectClass,
    TaskContract,
    compile_explicit_contract,
)
from sentrygraph.hybrid_conformance_policy import FrozenHybridConformancePolicy
from sentrygraph.multi_agent_agentdojo_adapter import MultiAgentSentryGraphRuntime


def delegated_root_contract(contract: TaskContract) -> TaskContract:
    """Rebind a task contract to an orchestrator that may delegate once."""

    return compile_explicit_contract(
        principal="orchestrator",
        objective=contract.objective,
        permitted_tools=contract.permitted_tools,
        prohibited_tools=contract.prohibited_tools,
        prohibited_effects=frozenset(
            effect
            for effect in contract.prohibited_effects
            if effect != EffectClass.DELEGATE
        ),
        delegation_allowed=True,
    )


def contract_graph(contract: TaskContract) -> dict[str, Any]:
    """Create the same policy-facing graph vocabulary used by Phase 1."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, attributes: dict[str, Any]) -> None:
        nodes.append(
            {"node_id": node_id, "node_type": node_type, "attributes": attributes}
        )

    def edge(source: str, target: str, relation: str) -> None:
        edges.append(
            {"source": source, "target": target, "relation": relation, "attributes": {}}
        )

    principal_id = "principal:0"
    contract_id = "contract:0"
    node(principal_id, "principal", {"name": contract.principal})
    node(
        contract_id,
        "task_contract",
        {
            "contract_id": contract.contract_id,
            "delegation_allowed": contract.delegation_allowed,
        },
    )
    edge(principal_id, contract_id, "BOUND_BY")

    for index, authorization in enumerate(contract.permitted_tools):
        tool_id = f"tool:permitted:{index}"
        effect_id = f"effect:authorized:{index}"
        node(
            tool_id,
            "tool",
            {
                "name": authorization.tool,
                "authorization": "permitted",
                "approval_required": authorization.approval_required,
                "allow_additional_arguments": authorization.allow_additional_arguments,
            },
        )
        node(
            effect_id,
            "effect",
            {"name": authorization.effect.value, "authorization": "authorized"},
        )
        edge(contract_id, tool_id, "PERMITS")
        edge(tool_id, effect_id, "HAS_EFFECT")
        for constraint_index, constraint in enumerate(authorization.constraints):
            constraint_id = f"constraint:{index}:{constraint_index}"
            node(
                constraint_id,
                "argument_constraint",
                {
                    "argument": constraint.argument,
                    "operator": getattr(constraint.operator, "value", str(constraint.operator)),
                    "value": constraint.value,
                },
            )
            edge(tool_id, constraint_id, "CONSTRAINS")

    for index, tool in enumerate(sorted(contract.prohibited_tools)):
        tool_id = f"tool:prohibited:{index}"
        node(tool_id, "tool", {"name": tool, "authorization": "prohibited"})
        edge(contract_id, tool_id, "PROHIBITS")

    for index, effect in enumerate(sorted(contract.prohibited_effects, key=lambda x: x.value)):
        effect_id = f"effect:prohibited:{index}"
        node(effect_id, "effect", {"name": effect.value, "authorization": "prohibited"})
        edge(contract_id, effect_id, "PROHIBITS_EFFECT")

    return {"nodes": nodes, "edges": edges}


def prefix_graph(events: list[dict[str, Any]], before_step: int) -> dict[str, Any]:
    """Build a leakage-free execution prefix from events preceding a candidate."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    nodes.append(
        {"node_id": "principal:worker", "node_type": "principal", "attributes": {"name": "worker"}}
    )
    for event in events:
        step = int(event.get("step", -1))
        if step >= before_step or event.get("phase") not in {"proposed", "completed"}:
            continue
        action_id = str(event.get("event_id", f"step-{step}"))
        action_node = f"action:{action_id}"
        if event["phase"] == "proposed":
            tool = str(event.get("tool"))
            tool_node = f"tool:{tool}"
            nodes.extend(
                [
                    {
                        "node_id": action_node,
                        "node_type": "proposed_action",
                        "attributes": {"tool": tool, "step": step},
                    },
                    {"node_id": tool_node, "node_type": "tool", "attributes": {"name": tool}},
                ]
            )
            edges.extend(
                [
                    {"source": "principal:worker", "target": action_node, "relation": "PROPOSED"},
                    {"source": action_node, "target": tool_node, "relation": "INVOKES"},
                ]
            )
        else:
            outcome_node = f"outcome:{action_id}"
            nodes.append(
                {
                    "node_id": outcome_node,
                    "node_type": "execution_outcome",
                    "attributes": {"step": step},
                }
            )
            edges.append(
                {"source": action_node, "target": outcome_node, "relation": "COMPLETED_AS"}
            )
    # Deduplicate nodes while retaining deterministic first occurrence.
    unique = {item["node_id"]: item for item in reversed(nodes)}
    return {"nodes": list(reversed(list(unique.values()))), "edges": edges}


class HybridMultiAgentSentryGraphRuntime(MultiAgentSentryGraphRuntime):
    """Apply a frozen learned policy only after deterministic authorization."""

    def __init__(
        self,
        tools: list[Any],
        *,
        root_contract: TaskContract,
        policy_bundle: str | Path,
    ) -> None:
        super().__init__(tools, root_contract=root_contract)
        self.learned_policy = FrozenHybridConformancePolicy(policy_bundle)
        self._feature_events: list[dict[str, Any]] = []

    def run_function_as(
        self,
        env: Any,
        principal: str,
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        proposed_at = utc_now()
        scoped = self.authority.propose_action(
            principal=principal,
            tool=function,
            arguments=deepcopy(kwargs),
        )
        self._persist_graph()
        proposal = self._proposal_event(
            scoped=scoped,
            function=function,
            kwargs=kwargs,
            timestamp=proposed_at,
        )
        self._write_event(proposal)

        if not scoped.permitted:
            self.denied_actions += 1
            self._write_event(
                {
                    **proposal,
                    "phase": "denied",
                    "timestamp": utc_now(),
                    "effect_executed": False,
                    "decision_source": "scoped_deterministic_policy_gate",
                }
            )
            return scoped.policy.reason, "SENTRYGRAPH_POLICY_DENIED"

        active = self.authority.active_contract_for(principal)
        if active is None:
            raise RuntimeError(f"No active contract for {principal!r}.")
        example = {
            "features": {
                "candidate_action": {
                    "tool": function,
                    "arguments": deepcopy(kwargs),
                    "step": scoped.step,
                },
                "contract_graph": contract_graph(active),
                "execution_prefix": prefix_graph(self._feature_events, scoped.step),
            }
        }
        prediction = self.learned_policy.predict_example(example)
        combined = self.learned_policy.combine_with_deterministic(
            deterministic_allowed=True,
            prediction=prediction,
        )
        learned_event = {
            **proposal,
            "phase": "learned_decision",
            "timestamp": utc_now(),
            "decision": combined["decision"],
            "decision_source": combined["source"],
            "prediction": prediction.to_dict(),
            "effect_executed": False,
        }
        self._write_event(learned_event)
        self._feature_events.append(proposal)

        if not combined["effect_allowed"]:
            self.denied_actions += 1
            decision_node = f"learned-decision:{scoped.action_id}"
            action_node = f"action:{scoped.action_id}"
            self.execution_graph.add_node(
                node_id=decision_node,
                node_type="policy_decision",
                attributes={
                    "decision": combined["decision"],
                    "rule_id": f"LEARNED_{combined['decision'].upper()}",
                    "source": "learned_conformance_policy",
                },
            )
            self.execution_graph.add_edge(
                source=action_node,
                target=decision_node,
                relation="EVALUATED_AS",
                step=scoped.step,
            )
            self.execution_graph.add_edge(
                source=decision_node,
                target=action_node,
                relation="BLOCKS",
                step=scoped.step,
            )
            self._persist_graph()
            self._write_event(
                {
                    **learned_event,
                    "phase": "denied",
                    "rule_id": f"LEARNED_{combined['decision'].upper()}",
                    "effect_executed": False,
                }
            )
            return (
                f"Learned conformance policy decision: {combined['decision']}",
                "SENTRYGRAPH_LEARNED_POLICY_INTERVENTION",
            )

        start = time.perf_counter()
        result, error = FunctionsRuntime.run_function(self, env, function, kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.executed_actions += 1
        self.authority.record_completion(
            decision=scoped,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )
        self._persist_graph()
        completed = {
            **proposal,
            "phase": "completed",
            "timestamp": utc_now(),
            "result": result,
            "error": error,
            "latency_ms": latency_ms,
            "effect_executed": True,
            "decision_source": "learned_conformance_policy",
            "learned_decision": "allow",
        }
        self._write_event(completed)
        self._feature_events.append(completed)
        return result, error

