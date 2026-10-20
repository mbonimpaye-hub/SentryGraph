"""Test authorized pre-effect interception using a real AgentDojo tool."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_adapter import SentryGraphRuntime


TRACE_PATH = Path("outputs/test_interception.jsonl")


def main() -> None:
    """Test an authorized, read-only AgentDojo action."""

    # Begin with a clean evidence file.
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    # Configure this test independently from other tests.
    os.environ["SENTRYGRAPH_TRACE_PATH"] = str(TRACE_PATH)
    os.environ["SENTRYGRAPH_TRAJECTORY_ID"] = (
        "workspace-readonly-interception-test"
    )

    # Ensure no tool is prohibited in this authorization test.
    os.environ.pop("SENTRYGRAPH_DENY_TOOLS", None)

    # Load the native AgentDojo workspace environment.
    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})

    environment_before = environment.model_dump(mode="json")

    # Initialize the SentryGraph-mediated runtime.
    runtime = SentryGraphRuntime(suite.tools)

    # Execute a genuine, read-only AgentDojo tool.
    tool_result, error = runtime.run_function(
        environment,
        "get_current_day",
        {},
    )

    environment_after = environment.model_dump(mode="json")

    # Validate native execution.
    assert error is None, (
        f"Tool execution returned an unexpected error: {error}"
    )
    assert runtime.step == 1
    assert runtime.executed_actions == 1
    assert runtime.denied_actions == 0

    # A read-only tool must not modify the environment.
    assert environment_before == environment_after, (
        "The read-only tool unexpectedly modified the environment"
    )

    # Confirm that SentryGraph produced an evidence file.
    assert TRACE_PATH.exists(), (
        "SentryGraph did not create an evidence file"
    )

    with TRACE_PATH.open("r", encoding="utf-8") as trace_file:
        events = [
            json.loads(line)
            for line in trace_file
            if line.strip()
        ]

    # One proposed event and one completed event are expected.
    assert len(events) == 2, (
        f"Expected two evidence events, found {len(events)}"
    )

    proposed_event = events[0]
    completed_event = events[1]

    # Validate event ordering.
    assert proposed_event["phase"] == "proposed"
    assert completed_event["phase"] == "completed"

    # Both records must describe the same action.
    assert (
        proposed_event["event_id"]
        == completed_event["event_id"]
    )
    assert (
        proposed_event["trajectory_id"]
        == completed_event["trajectory_id"]
        == "workspace-readonly-interception-test"
    )
    assert proposed_event["step"] == completed_event["step"] == 1
    assert (
        proposed_event["tool"]
        == completed_event["tool"]
        == "get_current_day"
    )

    # Validate the deterministic policy decision.
    assert proposed_event["decision"] == "allow"
    assert (
        proposed_event["decision_source"]
        == "deterministic_policy_gate"
    )
    assert proposed_event["rule_id"] == "DEFAULT_ALLOW"
    assert (
        proposed_event["reason"]
        == "No deterministic prohibition matched."
    )

    # Validate execution evidence.
    assert completed_event["effect_executed"] is True
    assert completed_event["error"] is None
    assert completed_event["latency_ms"] >= 0

    print(
        "PASS: SentryGraph intercepted and authorized "
        "a real AgentDojo action"
    )
    print("Tool: get_current_day")
    print(f"Result: {tool_result}")
    print(f"Execution error: {error}")
    print(
        "Environment modified: "
        f"{environment_before != environment_after}"
    )
    print(f"Executed actions: {runtime.executed_actions}")
    print(f"Denied actions: {runtime.denied_actions}")
    print(f"Evidence events: {len(events)}")
    print(
        "Event order: "
        f"{proposed_event['phase']} -> "
        f"{completed_event['phase']}"
    )
    print(f"Decision: {proposed_event['decision']}")
    print(f"Rule: {proposed_event['rule_id']}")
    print(
        f"Latency: {completed_event['latency_ms']:.3f} ms"
    )
    print(f"Evidence file: {TRACE_PATH}")


if __name__ == "__main__":
    main()