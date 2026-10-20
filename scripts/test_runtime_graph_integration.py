"""Test automatic runtime-to-execution-graph integration."""

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


TRACE_PATH = Path("outputs/test_runtime_graph.jsonl")
GRAPH_PATH = Path("outputs/test_runtime_graph.json")


def main() -> None:
    for path in (TRACE_PATH, GRAPH_PATH):
        if path.exists():
            path.unlink()

    os.environ["SENTRYGRAPH_TRACE_PATH"] = str(TRACE_PATH)
    os.environ["SENTRYGRAPH_GRAPH_PATH"] = str(GRAPH_PATH)
    os.environ["SENTRYGRAPH_TRAJECTORY_ID"] = (
        "runtime-graph-integration-test"
    )
    os.environ.pop("SENTRYGRAPH_DENY_TOOLS", None)

    contract = compile_explicit_contract(
        principal="agentdojo-workspace-user",
        objective="Read the date and create only the authorized file.",
        permitted_tools=(
            ToolAuthorization(
                tool="get_current_day",
                effect=EffectClass.READ,
            ),
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
            ),
        ),
    )

    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})

    runtime = SentryGraphRuntime(
        suite.tools,
        contract=contract,
        principal="agentdojo-workspace-user",
    )

    # Authorized read.
    _, read_error = runtime.run_function(
        environment,
        "get_current_day",
        {},
    )
    assert read_error is None

    # Unauthorized write argument.
    environment_before_denial = environment.model_dump(mode="json")

    _, denied_error = runtime.run_function(
        environment,
        "create_file",
        {
            "filename": "unauthorized.txt",
            "content": "This content must not be written.",
        },
    )

    environment_after_denial = environment.model_dump(mode="json")

    assert denied_error == "SENTRYGRAPH_POLICY_DENIED"
    assert environment_before_denial == environment_after_denial

    assert TRACE_PATH.exists()
    assert GRAPH_PATH.exists()

    events = [
        json.loads(line)
        for line in TRACE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    graph = json.loads(
        GRAPH_PATH.read_text(encoding="utf-8")
    )

    assert len(events) == 4

    first_event_id = events[0]["event_id"]
    second_event_id = events[2]["event_id"]

    assert events[0]["phase"] == "proposed"
    assert events[1]["phase"] == "completed"
    assert events[2]["phase"] == "proposed"
    assert events[3]["phase"] == "denied"

    action_nodes = {
        node["attributes"]["action_id"]: node
        for node in graph["nodes"]
        if node["node_type"] == "proposed_action"
    }

    assert first_event_id in action_nodes
    assert second_event_id in action_nodes

    assert (
        action_nodes[first_event_id]["attributes"]["status"]
        == "completed"
    )
    assert (
        action_nodes[second_event_id]["attributes"]["status"]
        == "denied"
    )

    first_relations = {
        edge["relation"]
        for edge in graph["edges"]
        if edge["step"] == 1
    }

    second_relations = {
        edge["relation"]
        for edge in graph["edges"]
        if edge["step"] == 2
    }

    assert "COMPLETED_AS" in first_relations
    assert "BLOCKS" not in first_relations

    assert "BLOCKS" in second_relations
    assert "COMPLETED_AS" not in second_relations

    assert events[0]["contract_id"] == contract.contract_id
    assert events[2]["contract_id"] == contract.contract_id

    print(
        "PASS: Real AgentDojo actions automatically populated "
        "the execution graph"
    )
    print(f"Contract ID: {contract.contract_id}")
    print(f"Trajectory ID: {runtime.trajectory_id}")
    print(f"Evidence events: {len(events)}")
    print(f"Graph nodes: {len(graph['nodes'])}")
    print(f"Graph edges: {len(graph['edges'])}")
    print(f"Step 1 relations: {sorted(first_relations)}")
    print(f"Step 2 relations: {sorted(second_relations)}")
    print("Step 1 status: completed")
    print("Step 2 status: denied")
    print("Denied action changed environment: False")
    print(f"Evidence file: {TRACE_PATH}")
    print(f"Graph file: {GRAPH_PATH}")


if __name__ == "__main__":
    main()