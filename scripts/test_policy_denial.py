"""Verify that SentryGraph prevents a prohibited effect."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_adapter import SentryGraphRuntime


TRACE_PATH = Path("outputs/test_policy_denial.jsonl")


def main() -> None:
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    os.environ["SENTRYGRAPH_TRACE_PATH"] = str(TRACE_PATH)
    os.environ["SENTRYGRAPH_TRAJECTORY_ID"] = (
        "workspace-policy-denial-test"
    )
    os.environ["SENTRYGRAPH_DENY_TOOLS"] = "create_file"

    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})

    environment_before = environment.model_dump(mode="json")

    runtime = SentryGraphRuntime(suite.tools)

    result, error = runtime.run_function(
        environment,
        "create_file",
        {
            "filename": "unauthorized.txt",
            "content": "This file must never be created.",
        },
    )

    environment_after = environment.model_dump(mode="json")

    with TRACE_PATH.open("r", encoding="utf-8") as trace_file:
        events = [
            json.loads(line)
            for line in trace_file
            if line.strip()
        ]

    assert environment_before == environment_after
    assert runtime.step == 1
    assert runtime.executed_actions == 0
    assert runtime.denied_actions == 1
    assert error == "SENTRYGRAPH_POLICY_DENIED"

    assert len(events) == 2
    assert events[0]["phase"] == "proposed"
    assert events[1]["phase"] == "denied"
    assert events[0]["decision"] == "deny"
    assert events[1]["effect_executed"] is False

    assert not any(
        event["phase"] == "completed"
        for event in events
    )

    print("PASS: SentryGraph prevented a prohibited action")
    print("Tool: create_file")
    print(f"Result: {result}")
    print(f"Error: {error}")
    print(
        f"Environment modified: "
        f"{environment_before != environment_after}"
    )
    print(f"Executed actions: {runtime.executed_actions}")
    print(f"Denied actions: {runtime.denied_actions}")
    print(
        f"Event order: "
        f"{events[0]['phase']} -> {events[1]['phase']}"
    )
    print(f"Evidence file: {TRACE_PATH}")


if __name__ == "__main__":
    main()