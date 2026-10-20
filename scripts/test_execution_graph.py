"""Test SentryGraph's heterogeneous temporal execution graph."""

from __future__ import annotations

from pathlib import Path

from sentrygraph.execution_graph import ExecutionGraph


OUTPUT_PATH = Path("outputs/test_execution_graph.json")


def main() -> None:
    graph = ExecutionGraph(
        trajectory_id="execution-graph-test",
        principal="agentdojo-workspace-user",
    )

    # Step 1: authorized action that completes.
    graph.record_proposal(
        action_id="action-1",
        tool="create_file",
        arguments={
            "filename": "authorized-report.txt",
            "content": "Authorized report",
        },
        step=1,
    )

    graph.record_decision(
        action_id="action-1",
        decision="allow",
        rule_id="CONTRACT_ALLOW",
        reason="Tool and arguments satisfy the contract.",
        contract_id="contract-example",
        step=1,
    )

    graph.record_completion(
        action_id="action-1",
        result={"file_id": "26"},
        error=None,
        latency_ms=0.25,
        step=1,
    )

    # Step 2: same tool targeting an unauthorized resource.
    graph.record_proposal(
        action_id="action-2",
        tool="create_file",
        arguments={
            "filename": "unauthorized.txt",
            "content": "Unauthorized content",
        },
        step=2,
    )

    graph.record_decision(
        action_id="action-2",
        decision="deny",
        rule_id="ARGUMENT_CONSTRAINT_VIOLATION",
        reason="Filename violates the task contract.",
        contract_id="contract-example",
        step=2,
    )

    graph.record_denial(
        action_id="action-2",
        step=2,
    )

    step_one_relations = graph.relations_for_step(1)
    step_two_relations = graph.relations_for_step(2)

    assert "PROPOSED" in step_one_relations
    assert "INVOKES" in step_one_relations
    assert "TARGETS" in step_one_relations
    assert "EVALUATED_AS" in step_one_relations
    assert "GOVERNS" in step_one_relations
    assert "COMPLETED_AS" in step_one_relations

    assert "PROPOSED" in step_two_relations
    assert "INVOKES" in step_two_relations
    assert "TARGETS" in step_two_relations
    assert "EVALUATED_AS" in step_two_relations
    assert "GOVERNS" in step_two_relations
    assert "BLOCKS" in step_two_relations
    assert "COMPLETED_AS" not in step_two_relations

    assert (
        graph.nodes["action:action-1"].attributes["status"]
        == "completed"
    )
    assert (
        graph.nodes["action:action-2"].attributes["status"]
        == "denied"
    )

    graph.save(OUTPUT_PATH)

    assert OUTPUT_PATH.exists()

    print("PASS: SentryGraph constructed a temporal execution graph")
    print(f"Trajectory: {graph.trajectory_id}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Step 1 relations: {step_one_relations}")
    print(f"Step 2 relations: {step_two_relations}")
    print("Step 1 status: completed")
    print("Step 2 status: denied")
    print(f"Graph file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()