"""Test task-conditioned and argument-level contract enforcement."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_adapter import SentryGraphRuntime
from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.policy_gate import DeterministicPolicyGate


TRACE_PATH = Path("outputs/test_contract_enforcement.jsonl")


def main() -> None:
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    os.environ["SENTRYGRAPH_TRACE_PATH"] = str(TRACE_PATH)
    os.environ["SENTRYGRAPH_TRAJECTORY_ID"] = (
        "workspace-contract-enforcement-test"
    )
    os.environ.pop("SENTRYGRAPH_DENY_TOOLS", None)

    contract = compile_explicit_contract(
        principal="agentdojo-workspace-user",
        objective="Create only the authorized report file.",
        permitted_tools=(
            ToolAuthorization(
                tool="create_file",
                effect=EffectClass.WRITE,
                constraints=(
                    ArgumentConstraint(
                        argument="filename",
                        operator=ConstraintOperator.EQUALS,
                        value="authorized-report.txt",
                    ),
                ),
                justification=(
                    "The task authorizes creation of only this file."
                ),
            ),
        ),
        prohibited_tools=frozenset(
            {
                "send_email",
                "share_file",
                "delete_file",
            }
        ),
        prohibited_effects=frozenset(
            {
                EffectClass.COMMUNICATE,
                EffectClass.DELETE,
                EffectClass.DELEGATE,
            }
        ),
    )

    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})

    runtime = SentryGraphRuntime(suite.tools)

    # Attach the task-specific contract to the enforcement gate.
    runtime.policy_gate = DeterministicPolicyGate(
        contract=contract
    )

    initial_environment = environment.model_dump(mode="json")

    # Case 1: authorized filename.
    allowed_result, allowed_error = runtime.run_function(
        environment,
        "create_file",
        {
            "filename": "authorized-report.txt",
            "content": "Authorized SentryGraph report.",
        },
    )

    environment_after_allowed = environment.model_dump(mode="json")

    assert allowed_error is None
    assert environment_after_allowed != initial_environment
    assert runtime.executed_actions == 1
    assert runtime.denied_actions == 0

    # Case 2: same tool, but an unauthorized filename.
    denied_result, denied_error = runtime.run_function(
        environment,
        "create_file",
        {
            "filename": "unauthorized.txt",
            "content": "This file must not be created.",
        },
    )

    environment_after_denied = environment.model_dump(mode="json")

    # The denied action must cause no additional state change.
    assert environment_after_denied == environment_after_allowed
    assert denied_error == "SENTRYGRAPH_POLICY_DENIED"
    assert runtime.executed_actions == 1
    assert runtime.denied_actions == 1

    with TRACE_PATH.open("r", encoding="utf-8") as trace_file:
        events = [
            json.loads(line)
            for line in trace_file
            if line.strip()
        ]

    assert len(events) == 4

    allowed_proposal = events[0]
    allowed_completion = events[1]
    denied_proposal = events[2]
    denied_event = events[3]

    assert allowed_proposal["phase"] == "proposed"
    assert allowed_completion["phase"] == "completed"
    assert allowed_proposal["decision"] == "allow"
    assert allowed_proposal["rule_id"] == "CONTRACT_ALLOW"

    assert denied_proposal["phase"] == "proposed"
    assert denied_event["phase"] == "denied"
    assert denied_proposal["decision"] == "deny"
    assert (
        denied_proposal["rule_id"]
        == "ARGUMENT_CONSTRAINT_VIOLATION"
    )
    assert denied_event["effect_executed"] is False

    assert not any(
        event["phase"] == "completed"
        and event["step"] == 2
        for event in events
    )

    print(
        "PASS: SentryGraph enforced a task-conditioned "
        "argument constraint"
    )
    print(f"Contract ID: {contract.contract_id}")
    print(f"Authorized result: {allowed_result}")
    print(f"Authorized error: {allowed_error}")
    print(f"Denied result: {denied_result}")
    print(f"Denied error: {denied_error}")
    print("Authorized state change: True")
    print("Unauthorized state change: False")
    print(f"Executed actions: {runtime.executed_actions}")
    print(f"Denied actions: {runtime.denied_actions}")
    print(
        "Decision sequence: "
        "allow -> completed -> deny -> blocked"
    )
    print(f"Evidence file: {TRACE_PATH}")


if __name__ == "__main__":
    main()