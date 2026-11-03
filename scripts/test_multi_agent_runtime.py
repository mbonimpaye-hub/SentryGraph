"""Regression audit for Phase 2C runtime authority enforcement."""

from __future__ import annotations

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.multi_agent_runtime import (
    MultiAgentAuthorityRuntime,
    MultiAgentRuntimeError,
)


def authorization(
    tool: str,
    effect: EffectClass,
    *,
    filename: str | None = None,
) -> ToolAuthorization:
    constraints = ()
    if filename is not None:
        constraints = (
            ArgumentConstraint(
                argument="filename",
                operator=ConstraintOperator.EQUALS,
                value=filename,
            ),
        )
    return ToolAuthorization(
        tool=tool,
        effect=effect,
        constraints=constraints,
        allow_additional_arguments=False,
        justification="Phase 2C runtime audit.",
    )


def main() -> None:
    create_report = authorization(
        "create_file",
        EffectClass.WRITE,
        filename="delegated-report.txt",
    )
    send_email = authorization(
        "send_email",
        EffectClass.COMMUNICATE,
    )
    root = compile_explicit_contract(
        principal="orchestrator",
        objective="Create the report without unrelated effects.",
        permitted_tools=(create_report, send_email),
        prohibited_tools=frozenset({"delete_file"}),
        delegation_allowed=True,
    )
    runtime = MultiAgentAuthorityRuntime(
        root_contract=root,
        trajectory_id="phase-2c-runtime-audit",
    )

    child_decision = runtime.request_delegation(
        parent_principal="orchestrator",
        child_principal="report-writer",
        objective="Create only the delegated report.",
        permitted_tools=(create_report,),
        child_delegation_allowed=True,
        max_depth=2,
    )
    assert child_decision.allowed
    child = child_decision.delegation
    assert child is not None

    allowed = runtime.propose_action(
        principal="report-writer",
        tool="create_file",
        arguments={"filename": "delegated-report.txt"},
    )
    assert allowed.permitted
    runtime.record_completion(
        decision=allowed,
        result={"file_id": "phase-2c-report"},
        latency_ms=0.1,
    )

    wrong_argument = runtime.propose_action(
        principal="report-writer",
        tool="create_file",
        arguments={"filename": "attacker-controlled.txt"},
    )
    assert not wrong_argument.permitted
    assert (
        wrong_argument.policy.rule_id
        == "ARGUMENT_CONSTRAINT_VIOLATION"
    )

    outside_child_scope = runtime.propose_action(
        principal="report-writer",
        tool="send_email",
        arguments={},
    )
    assert not outside_child_scope.permitted
    assert outside_child_scope.policy.rule_id in {
        "CONTRACT_PROHIBITED_TOOL",
        "TOOL_OUTSIDE_CONTRACT",
    }

    unknown = runtime.propose_action(
        principal="unregistered-agent",
        tool="create_file",
        arguments={"filename": "delegated-report.txt"},
    )
    assert not unknown.permitted
    assert unknown.policy.rule_id == "AGENT_CONTEXT_UNKNOWN"

    grandchild_decision = runtime.request_delegation(
        parent_principal="report-writer",
        child_principal="format-worker",
        objective="Create only the delegated report.",
        permitted_tools=(create_report,),
        child_delegation_allowed=False,
    )
    assert grandchild_decision.allowed
    grandchild = grandchild_decision.delegation
    assert grandchild is not None
    assert grandchild.depth == 2

    too_deep = runtime.request_delegation(
        parent_principal="format-worker",
        child_principal="fourth-agent",
        objective="Attempt an excessive delegation.",
        permitted_tools=(create_report,),
    )
    assert not too_deep.allowed

    affected = runtime.revoke_delegation(
        delegation_id=child.delegation_id,
        reason="Parent terminated delegated work.",
    )
    assert affected == ("report-writer", "format-worker")
    assert runtime.active_contract_for("report-writer") is None
    assert runtime.active_contract_for("format-worker") is None

    after_revocation = runtime.propose_action(
        principal="format-worker",
        tool="create_file",
        arguments={"filename": "delegated-report.txt"},
    )
    assert not after_revocation.permitted
    assert after_revocation.policy.rule_id == "AGENT_CONTEXT_INACTIVE"

    try:
        runtime.record_completion(
            decision=after_revocation,
            result={"file_id": "must-not-exist"},
        )
    except MultiAgentRuntimeError:
        pass
    else:
        raise AssertionError("Denied action was recorded as completed.")

    graph = runtime.execution_graph.to_dict()
    relations = {
        edge["relation"]
        for edge in graph["edges"]
    }
    expected = {
        "HANDOFF_PROPOSED",
        "HANDOFF_ACCEPTED",
        "SUBDELEGATES_TO",
        "DELEGATED_UNDER",
        "GOVERNS",
        "BLOCKS",
        "REVOKES",
        "DEACTIVATES",
    }
    assert expected.issubset(relations)

    print("PASS: Accepted agents are bound to attenuated child contracts.")
    print("PASS: Child tool and argument violations fail closed.")
    print("PASS: Unknown and inactive principals cannot execute actions.")
    print("PASS: Delegation depth is enforced at runtime.")
    print("PASS: Revocation recursively deactivates descendants.")
    print("PASS: Denied actions cannot be recorded as completed.")
    print("Relations:", sorted(relations))
    print("Contexts:", sorted(runtime.contexts))
    print("Steps:", runtime.step)


if __name__ == "__main__":
    main()
