"""AgentDojo adapter for scoped multi-agent SentryGraph execution."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from agentdojo.functions_runtime import FunctionsRuntime

from sentrygraph.agentdojo_adapter import json_safe, utc_now
from sentrygraph.contract_compiler import (
    EffectClass,
    TaskContract,
    ToolAuthorization,
)
from sentrygraph.multi_agent_contracts import DelegationDecision
from sentrygraph.multi_agent_runtime import (
    MultiAgentAuthorityRuntime,
    ScopedActionDecision,
)


class MultiAgentSentryGraphRuntime(FunctionsRuntime):
    """Execute AgentDojo tools only under an active scoped agent contract."""

    def __init__(
        self,
        tools: list[Any],
        *,
        root_contract: TaskContract,
    ) -> None:
        super().__init__(tools)

        self.trace_path = Path(
            os.getenv(
                "SENTRYGRAPH_TRACE_PATH",
                "outputs/multi_agent_monitor_trace.jsonl",
            )
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path = Path(
            os.getenv(
                "SENTRYGRAPH_GRAPH_PATH",
                "outputs/multi_agent_execution_graph.json",
            )
        )
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_id = os.getenv("SENTRYGRAPH_TRAJECTORY_ID")
        denied_tools = {
            tool.strip()
            for tool in os.getenv("SENTRYGRAPH_DENY_TOOLS", "").split(",")
            if tool.strip()
        }

        self.root_principal = root_contract.principal
        self.authority = MultiAgentAuthorityRuntime(
            root_contract=root_contract,
            trajectory_id=trajectory_id,
            denied_tools=denied_tools,
        )
        self.trajectory_id = self.authority.trajectory_id
        self.execution_graph = self.authority.execution_graph
        self.executed_actions = 0
        self.denied_actions = 0
        self._persist_graph()

    def _write_event(self, event: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(
                    json_safe(event),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _persist_graph(self) -> None:
        self.execution_graph.save(self.graph_path)

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
        """Compile and activate a bounded child authority context."""

        decision = self.authority.request_delegation(
            parent_principal=parent_principal,
            child_principal=child_principal,
            objective=objective,
            permitted_tools=permitted_tools,
            additional_prohibited_tools=additional_prohibited_tools,
            additional_prohibited_effects=additional_prohibited_effects,
            child_delegation_allowed=child_delegation_allowed,
            max_depth=max_depth,
        )
        self._persist_graph()
        delegation = decision.delegation
        self._write_event(
            {
                "schema_version": "0.4",
                "trajectory_id": self.trajectory_id,
                "step": self.authority.step,
                "phase": (
                    "delegation_accepted"
                    if decision.allowed
                    else "delegation_denied"
                ),
                "timestamp": utc_now(),
                "parent_principal": parent_principal,
                "child_principal": child_principal,
                "delegation_id": (
                    delegation.delegation_id if delegation else None
                ),
                "child_contract_id": (
                    delegation.child_contract.contract_id
                    if delegation
                    else None
                ),
                "decision": "allow" if decision.allowed else "deny",
                "rule_id": decision.rule_id,
                "reason": decision.reason,
                "effect_executed": False,
                "graph_path": str(self.graph_path),
            }
        )
        return decision

    def revoke_delegation(
        self,
        *,
        delegation_id: str,
        reason: str,
    ) -> tuple[str, ...]:
        affected = self.authority.revoke_delegation(
            delegation_id=delegation_id,
            reason=reason,
        )
        self._persist_graph()
        self._write_event(
            {
                "schema_version": "0.4",
                "trajectory_id": self.trajectory_id,
                "step": self.authority.step,
                "phase": "delegation_revoked",
                "timestamp": utc_now(),
                "delegation_id": delegation_id,
                "reason": reason,
                "affected_principals": affected,
                "effect_executed": False,
                "graph_path": str(self.graph_path),
            }
        )
        return affected

    def _proposal_event(
        self,
        *,
        scoped: ScopedActionDecision,
        function: str,
        kwargs: dict[str, Any],
        timestamp: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "0.4",
            "event_id": scoped.action_id,
            "trajectory_id": self.trajectory_id,
            "step": scoped.step,
            "phase": "proposed",
            "timestamp": timestamp,
            "principal": scoped.principal,
            "delegation_id": scoped.delegation_id,
            "tool": function,
            "arguments": deepcopy(kwargs),
            "decision": scoped.policy.decision.value,
            "decision_source": "scoped_deterministic_policy_gate",
            "rule_id": scoped.policy.rule_id,
            "reason": scoped.policy.reason,
            "contract_id": scoped.contract_id,
            "graph_path": str(self.graph_path),
        }

    def run_function_as(
        self,
        env: Any,
        principal: str,
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Authorize, execute, and evidence a principal-scoped tool call."""

        proposed_at = utc_now()
        scoped = self.authority.propose_action(
            principal=principal,
            tool=function,
            arguments=deepcopy(kwargs),
        )
        self._persist_graph()
        self._write_event(
            self._proposal_event(
                scoped=scoped,
                function=function,
                kwargs=kwargs,
                timestamp=proposed_at,
            )
        )

        if not scoped.permitted:
            self.denied_actions += 1
            denied_at = utc_now()
            self._persist_graph()
            self._write_event(
                {
                    "schema_version": "0.4",
                    "event_id": scoped.action_id,
                    "trajectory_id": self.trajectory_id,
                    "step": scoped.step,
                    "phase": "denied",
                    "timestamp": denied_at,
                    "principal": principal,
                    "delegation_id": scoped.delegation_id,
                    "tool": function,
                    "decision": scoped.policy.decision.value,
                    "rule_id": scoped.policy.rule_id,
                    "reason": scoped.policy.reason,
                    "contract_id": scoped.contract_id,
                    "effect_executed": False,
                    "graph_path": str(self.graph_path),
                }
            )
            return scoped.policy.reason, "SENTRYGRAPH_POLICY_DENIED"

        start_time = time.perf_counter()
        try:
            tool_result, error = super().run_function(
                env,
                function,
                kwargs,
            )
        except Exception as exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.authority.record_completion(
                decision=scoped,
                result=None,
                error={
                    "type": type(exception).__name__,
                    "message": str(exception),
                },
                latency_ms=latency_ms,
            )
            self._persist_graph()
            self._write_event(
                {
                    "schema_version": "0.4",
                    "event_id": scoped.action_id,
                    "trajectory_id": self.trajectory_id,
                    "step": scoped.step,
                    "phase": "execution_exception",
                    "timestamp": utc_now(),
                    "principal": principal,
                    "delegation_id": scoped.delegation_id,
                    "tool": function,
                    "contract_id": scoped.contract_id,
                    "latency_ms": latency_ms,
                    "exception_type": type(exception).__name__,
                    "exception": str(exception),
                    "effect_executed": "unknown",
                    "graph_path": str(self.graph_path),
                }
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.executed_actions += 1
        self.authority.record_completion(
            decision=scoped,
            result=tool_result,
            error=error,
            latency_ms=latency_ms,
        )
        self._persist_graph()
        self._write_event(
            {
                "schema_version": "0.4",
                "event_id": scoped.action_id,
                "trajectory_id": self.trajectory_id,
                "step": scoped.step,
                "phase": "completed",
                "timestamp": utc_now(),
                "principal": principal,
                "delegation_id": scoped.delegation_id,
                "tool": function,
                "result": tool_result,
                "error": error,
                "contract_id": scoped.contract_id,
                "latency_ms": latency_ms,
                "effect_executed": True,
                "graph_path": str(self.graph_path),
            }
        )
        return tool_result, error

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Execute as the root principal for FunctionsRuntime compatibility."""

        return self.run_function_as(
            env,
            self.root_principal,
            function,
            kwargs,
        )
