"""Phase 2B tests for multi-agent delegation execution graphs."""

from __future__ import annotations

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.multi_agent_contracts import (
    compile_delegated_contract,
    evaluate_delegation_request,
)
from sentrygraph.multi_agent_execution_graph import (
    MultiAgentExecutionGraph,
)


def main() -> None:
    authorization = ToolAuthorization(
        tool="create_file",
        effect=EffectClass.WRITE,
        constraints=(
            ArgumentConstraint(
                argument="filename",
                operator=ConstraintOperator.EQUALS,
                value="authorized-report.txt",
            ),
        ),
        allow_additional_arguments=False,
        justification="Only the delegated report may be created.",
    )
    parent = compile_explicit_contract(
        principal="orchestrator-agent",
        objective="Coordinate one authorized report.",
        permitted_tools=(authorization,),
        prohibited_tools=frozenset({"delete_file", "send_email"}),
        prohibited_effects=frozenset(
            {EffectClass.DELETE, EffectClass.COMMUNICATE}
        ),
        delegation_allowed=True,
    )
    delegation = compile_delegated_contract(
        parent_contract=parent,
        child_principal="report-writer-agent",
        objective="Create the authorized report.",
        permitted_tools=(authorization,),
        depth=1,
        max_depth=1,
        child_delegation_allowed=False,
    )
    allowed = evaluate_delegation_request(
        parent_contract=parent,
        child_principal="report-writer-agent",
        objective="Create the authorized report.",
        permitted_tools=(authorization,),
        depth=1,
        max_depth=1,
        child_delegation_allowed=False,
    )
    assert allowed.allowed
    assert allowed.delegation is not None
    assert allowed.delegation.delegation_id == delegation.delegation_id

    graph = MultiAgentExecutionGraph(
        trajectory_id="multi-agent-graph-test",
        principal="orchestrator-agent",
    )
    graph.record_delegation_proposal(
        delegation_id=delegation.delegation_id,
        parent_principal=delegation.parent_principal,
        child_principal=delegation.child_principal,
        parent_contract_id=delegation.parent_contract_id,
        parent_contract_fingerprint=delegation.parent_contract_fingerprint,
        objective=delegation.objective,
        depth=delegation.depth,
        max_depth=delegation.max_depth,
        step=1,
    )
    graph.record_delegation_decision(
        delegation_id=delegation.delegation_id,
        decision=allowed,
        step=1,
    )
    graph.record_agent_proposal(
        action_id="child-action-1",
        principal=delegation.child_principal,
        contract_id=delegation.child_contract.contract_id,
        delegation_id=delegation.delegation_id,
        tool="create_file",
        arguments={"filename": "authorized-report.txt"},
        step=2,
    )
    graph.record_decision(
        action_id="child-action-1",
        decision="allow",
        rule_id="CONTRACT_ALLOW",
        reason="Delegated child action satisfies its scoped contract.",
        contract_id=delegation.child_contract.contract_id,
        step=2,
    )
    graph.record_completion(
        action_id="child-action-1",
        result={"created": "authorized-report.txt"},
        error=None,
        latency_ms=0.2,
        step=2,
    )

    denied = evaluate_delegation_request(
        parent_contract=parent,
        child_principal="malicious-agent",
        objective="Send unauthorized email.",
        permitted_tools=(
            ToolAuthorization(
                tool="send_email",
                effect=EffectClass.COMMUNICATE,
                justification="Must be denied.",
            ),
        ),
    )
    assert not denied.allowed
    denied_id = "delegation-denied-example"
    graph.record_delegation_proposal(
        delegation_id=denied_id,
        parent_principal="orchestrator-agent",
        child_principal="malicious-agent",
        parent_contract_id=parent.contract_id,
        parent_contract_fingerprint=parent.fingerprint(),
        objective="Send unauthorized email.",
        depth=1,
        max_depth=1,
        step=3,
    )
    graph.record_delegation_decision(
        delegation_id=denied_id,
        decision=denied,
        step=3,
    )

    relations = {edge.relation for edge in graph.edges}
    expected = {
        "HANDOFF_PROPOSED",
        "HANDOFF_ACCEPTED",
        "HANDOFF_DENIED",
        "DELEGATES_TO",
        "DERIVES_CONTRACT",
        "SCOPES",
        "DELEGATED_UNDER",
        "PROPOSED",
        "INVOKES",
        "GOVERNS",
        "EVALUATED_AS",
        "COMPLETED_AS",
        "BLOCKS",
    }
    assert expected.issubset(relations)
    assert graph.nodes[
        f"delegation:{delegation.delegation_id}"
    ].attributes["status"] == "accepted"
    assert graph.nodes[
        f"delegation:{denied_id}"
    ].attributes["status"] == "denied"
    assert graph.nodes["action:child-action-1"].attributes[
        "principal"
    ] == "report-writer-agent"

    try:
        graph.record_agent_proposal(
            action_id="denied-child-action",
            principal="malicious-agent",
            contract_id=parent.contract_id,
            delegation_id=denied_id,
            tool="send_email",
            arguments={"recipients": ["attacker@example.com"]},
            step=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Denied delegations must not authorize child actions."
        )

    output = graph.to_dict()
    assert output["schema_version"] == "0.2"
    assert output["multi_agent"] is True

    print("PASS: Accepted handoffs derive and scope child contracts.")
    print("PASS: Child actions are attributed to child principal and contract.")
    print("PASS: Denied handoffs create HANDOFF_DENIED and BLOCKS evidence.")
    print("PASS: Actions under denied delegations fail closed.")
    print("Relations:", sorted(relations))
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))


if __name__ == "__main__":
    main()
