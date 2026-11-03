"""Real AgentDojo state-change audit for Phase 2D."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentdojo.task_suite import get_suite

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.multi_agent_agentdojo_adapter import (
    MultiAgentSentryGraphRuntime,
)


TRACE_PATH = Path("outputs/test_multi_agent_runtime.jsonl")
GRAPH_PATH = Path("outputs/test_multi_agent_runtime_graph.json")


def main() -> None:
    for path in (TRACE_PATH, GRAPH_PATH):
        if path.exists():
            path.unlink()

    os.environ["SENTRYGRAPH_TRACE_PATH"] = str(TRACE_PATH)
    os.environ["SENTRYGRAPH_GRAPH_PATH"] = str(GRAPH_PATH)
    os.environ["SENTRYGRAPH_TRAJECTORY_ID"] = (
        "multi-agent-agentdojo-runtime-test"
    )
    os.environ.pop("SENTRYGRAPH_DENY_TOOLS", None)

    create_report = ToolAuthorization(
        tool="create_file",
        effect=EffectClass.WRITE,
        constraints=(
            ArgumentConstraint(
                argument="filename",
                operator=ConstraintOperator.EQUALS,
                value="delegated-report.txt",
            ),
        ),
        justification="Create only the explicitly delegated report.",
    )
    root_contract = compile_explicit_contract(
        principal="orchestrator",
        objective="Delegate creation of one report file.",
        permitted_tools=(
            create_report,
            ToolAuthorization(
                tool="send_email",
                effect=EffectClass.COMMUNICATE,
            ),
        ),
        prohibited_tools=frozenset({"delete_file"}),
        delegation_allowed=True,
    )

    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})
    runtime = MultiAgentSentryGraphRuntime(
        suite.tools,
        root_contract=root_contract,
    )
    delegation_decision = runtime.request_delegation(
        parent_principal="orchestrator",
        child_principal="report-writer",
        objective="Create only delegated-report.txt.",
        permitted_tools=(create_report,),
        max_depth=1,
    )
    assert delegation_decision.allowed
    delegation = delegation_decision.delegation
    assert delegation is not None

    initial_environment = environment.model_dump(mode="json")
    allowed_result, allowed_error = runtime.run_function_as(
        environment,
        "report-writer",
        "create_file",
        {
            "filename": "delegated-report.txt",
            "content": "Authorized delegated report.",
        },
    )
    environment_after_allowed = environment.model_dump(mode="json")
    assert allowed_error is None
    assert environment_after_allowed != initial_environment

    _, wrong_argument_error = runtime.run_function_as(
        environment,
        "report-writer",
        "create_file",
        {
            "filename": "attacker-controlled.txt",
            "content": "This file must not be created.",
        },
    )
    environment_after_wrong_argument = environment.model_dump(mode="json")
    assert wrong_argument_error == "SENTRYGRAPH_POLICY_DENIED"
    assert environment_after_wrong_argument == environment_after_allowed

    _, outside_scope_error = runtime.run_function_as(
        environment,
        "report-writer",
        "send_email",
        {
            "recipients": ["attacker@example.com"],
            "subject": "Unauthorized",
            "body": "This message must not be sent.",
        },
    )
    environment_after_outside_scope = environment.model_dump(mode="json")
    assert outside_scope_error == "SENTRYGRAPH_POLICY_DENIED"
    assert environment_after_outside_scope == environment_after_allowed

    affected = runtime.revoke_delegation(
        delegation_id=delegation.delegation_id,
        reason="Delegated work completed.",
    )
    assert affected == ("report-writer",)

    _, revoked_error = runtime.run_function_as(
        environment,
        "report-writer",
        "create_file",
        {
            "filename": "delegated-report.txt",
            "content": "A revoked agent must not write this.",
        },
    )
    environment_after_revoked = environment.model_dump(mode="json")
    assert revoked_error == "SENTRYGRAPH_POLICY_DENIED"
    assert environment_after_revoked == environment_after_allowed

    assert runtime.executed_actions == 1
    assert runtime.denied_actions == 3
    events = [
        json.loads(line)
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    relations = {edge["relation"] for edge in graph["edges"]}
    assert {
        "HANDOFF_ACCEPTED",
        "DELEGATED_UNDER",
        "COMPLETED_AS",
        "BLOCKS",
        "REVOKES",
        "DEACTIVATES",
    }.issubset(relations)
    assert sum(event["phase"] == "completed" for event in events) == 1
    assert sum(event["phase"] == "denied" for event in events) == 3
    assert not any(
        event["phase"] == "completed"
        and event.get("tool") in {"send_email"}
        for event in events
    )

    print("PASS: Authorized child action changed AgentDojo state.")
    print("PASS: Wrong-argument child action caused no state change.")
    print("PASS: Tool outside child scope caused no state change.")
    print("PASS: Revoked child action caused no state change.")
    print("PASS: Multi-agent evidence records principal and delegation scope.")
    print("Authorized result:", allowed_result)
    print("Executed actions:", runtime.executed_actions)
    print("Denied actions:", runtime.denied_actions)
    print("Evidence events:", len(events))
    print("Graph nodes:", len(graph["nodes"]))
    print("Graph edges:", len(graph["edges"]))
    print("Relations:", sorted(relations))
    print("Evidence file:", TRACE_PATH)
    print("Graph file:", GRAPH_PATH)


if __name__ == "__main__":
    main()
