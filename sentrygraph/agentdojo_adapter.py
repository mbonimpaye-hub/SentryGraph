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

from sentrygraph.policy_gate import (
    Decision,
    DeterministicPolicyGate,
)


def _json_safe(value: Any) -> Any:
    """Convert arbitrary values into JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))

    return repr(value)


class SentryGraphRuntime(FunctionsRuntime):
    """Mediate every AgentDojo tool invocation before execution."""

    def __init__(self, tools: list[Any]) -> None:
        super().__init__(tools)

        trace_location = os.getenv(
            "SENTRYGRAPH_TRACE_PATH",
            "outputs/monitor_trace.jsonl",
        )

        self.trace_path = Path(trace_location)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self.trajectory_id = os.getenv(
            "SENTRYGRAPH_TRAJECTORY_ID",
            str(uuid.uuid4()),
        )

        self.policy_gate = DeterministicPolicyGate.from_environment()
        self.step = 0
        self.executed_actions = 0
        self.denied_actions = 0

    def _write_event(self, event: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(
                    _json_safe(event),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Evaluate a pending action before allowing its effect."""
        self.step += 1

        event_id = str(uuid.uuid4())
        proposed_at = datetime.now(timezone.utc).isoformat()

        policy_decision = self.policy_gate.evaluate(
            tool=function,
            arguments=kwargs,
        )

        self._write_event(
            {
                "schema_version": "0.2",
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
            }
        )

        if policy_decision.decision == Decision.DENY:
            self.denied_actions += 1

            self._write_event(
                {
                    "schema_version": "0.2",
                    "event_id": event_id,
                    "trajectory_id": self.trajectory_id,
                    "step": self.step,
                    "phase": "denied",
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "tool": function,
                    "decision": Decision.DENY.value,
                    "rule_id": policy_decision.rule_id,
                    "reason": policy_decision.reason,
                    "effect_executed": False,
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

            self._write_event(
                {
                    "schema_version": "0.2",
                    "event_id": event_id,
                    "trajectory_id": self.trajectory_id,
                    "step": self.step,
                    "phase": "execution_exception",
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "tool": function,
                    "latency_ms": latency_ms,
                    "exception_type": type(exception).__name__,
                    "exception": str(exception),
                }
            )
            raise

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        self._write_event(
            {
                "schema_version": "0.2",
                "event_id": event_id,
                "trajectory_id": self.trajectory_id,
                "step": self.step,
                "phase": "completed",
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "tool": function,
                "result": tool_result,
                "error": error,
                "latency_ms": latency_ms,
                "effect_executed": True,
            }
        )

        return tool_result, error