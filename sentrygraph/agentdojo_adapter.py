"""AgentDojo enforcement adapter for SentryGraph."""

from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdojo.functions_runtime import FunctionsRuntime

from sentrygraph.contract_compiler import TaskContract
from sentrygraph.execution_graph import ExecutionGraph
from sentrygraph.policy_gate import (
    Decision,
    DeterministicPolicyGate,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class SentryGraphRuntime(FunctionsRuntime):
    """Mediate AgentDojo tool calls before external effects."""

    def __init__(
        self,
        tools: list[Any],
        contract: TaskContract | None = None,
        principal: str = "agent",
    ) -> None:
        super().__init__(tools)

        self.trace_path = Path(
            os.getenv(
                "SENTRYGRAPH_TRACE_PATH",
                "outputs/monitor_trace.jsonl",
            )
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self.graph_path = Path(
            os.getenv(
                "SENTRYGRAPH_GRAPH_PATH",
                "outputs/execution_graph.json",
            )
        )
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        self.trajectory_id = os.getenv(
            "SENTRYGRAPH_TRAJECTORY_ID",
            str(uuid.uuid4()),
        )

        self.contract = contract
        self.policy_gate = (
            DeterministicPolicyGate.from_environment(
                contract=contract
            )
        )

        self.execution_graph = ExecutionGraph(
            trajectory_id=self.trajectory_id,
            principal=principal,
        )

        self.step = 0
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

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Evaluate and record a pending action before execution."""

        self.step += 1

        event_id = str(uuid.uuid4())
        proposed_at = utc_now()

        policy_decision = self.policy_gate.evaluate(
            tool=function,
            arguments=kwargs,
        )

        # Build and persist the pre-effect execution graph.
        self.execution_graph.record_proposal(
            action_id=event_id,
            tool=function,
            arguments=deepcopy(kwargs),
            step=self.step,
            timestamp=proposed_at,
        )

        self.execution_graph.record_decision(
            action_id=event_id,
            decision=policy_decision.decision.value,
            rule_id=policy_decision.rule_id,
            reason=policy_decision.reason,
            contract_id=policy_decision.contract_id,
            step=self.step,
            timestamp=proposed_at,
        )

        self._persist_graph()

        # Persist the corresponding pre-effect evidence record.
        self._write_event(
            {
                "schema_version": "0.3",
                "event_id": event_id,
                "trajectory_id": self.trajectory_id,
                "step": self.step,
                "phase": "proposed",
                "timestamp": proposed_at,
                "tool": function,
                "arguments": deepcopy(kwargs),
                "decision": policy_decision.decision.value,
                "decision_source": "deterministic_policy_gate",
                "rule_id": policy_decision.rule_id,
                "reason": policy_decision.reason,
                "contract_id": policy_decision.contract_id,
                "graph_path": str(self.graph_path),
            }
        )

        if policy_decision.decision != Decision.ALLOW:
            self.denied_actions += 1

            denied_at = utc_now()

            self.execution_graph.record_denial(
                action_id=event_id,
                step=self.step,
                timestamp=denied_at,
            )
            self._persist_graph()

            self._write_event(
                {
                    "schema_version": "0.3",
                    "event_id": event_id,
                    "trajectory_id": self.trajectory_id,
                    "step": self.step,
                    "phase": "denied",
                    "timestamp": denied_at,
                    "tool": function,
                    "decision": policy_decision.decision.value,
                    "rule_id": policy_decision.rule_id,
                    "reason": policy_decision.reason,
                    "contract_id": policy_decision.contract_id,
                    "effect_executed": False,
                    "graph_path": str(self.graph_path),
                }
            )

            return (
                policy_decision.reason,
                "SENTRYGRAPH_POLICY_DENIED",
            )

        start_time = time.perf_counter()

        try:
            self.executed_actions += 1

            tool_result, error = super().run_function(
                env,
                function,
                kwargs,
            )

        except Exception as exception:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            exception_at = utc_now()

            self.execution_graph.record_completion(
                action_id=event_id,
                result=None,
                error={
                    "type": type(exception).__name__,
                    "message": str(exception),
                },
                latency_ms=latency_ms,
                step=self.step,
                timestamp=exception_at,
            )
            self._persist_graph()

            self._write_event(
                {
                    "schema_version": "0.3",
                    "event_id": event_id,
                    "trajectory_id": self.trajectory_id,
                    "step": self.step,
                    "phase": "execution_exception",
                    "timestamp": exception_at,
                    "tool": function,
                    "contract_id": policy_decision.contract_id,
                    "latency_ms": latency_ms,
                    "exception_type": type(exception).__name__,
                    "exception": str(exception),
                    "effect_executed": "unknown",
                    "graph_path": str(self.graph_path),
                }
            )
            raise

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        completed_at = utc_now()

        self.execution_graph.record_completion(
            action_id=event_id,
            result=tool_result,
            error=error,
            latency_ms=latency_ms,
            step=self.step,
            timestamp=completed_at,
        )
        self._persist_graph()

        self._write_event(
            {
                "schema_version": "0.3",
                "event_id": event_id,
                "trajectory_id": self.trajectory_id,
                "step": self.step,
                "phase": "completed",
                "timestamp": completed_at,
                "tool": function,
                "result": tool_result,
                "error": error,
                "contract_id": policy_decision.contract_id,
                "latency_ms": latency_ms,
                "effect_executed": True,
                "graph_path": str(self.graph_path),
            }
        )

        return tool_result, error