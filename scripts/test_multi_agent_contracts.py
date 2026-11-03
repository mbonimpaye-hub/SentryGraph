"""Phase 2A tests for fail-closed hierarchical delegation contracts."""

from __future__ import annotations

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.multi_agent_contracts import (
    DelegationContractError,
    authorization_is_attenuated,
    compile_delegated_contract,
    compile_subdelegated_contract,
    evaluate_delegation_request,
)


def authorization(
    *,
    tool: str = "create_file",
    filename: str = "authorized-report.txt",
    allow_additional_arguments: bool = False,
    approval_required: bool = False,
) -> ToolAuthorization:
    return ToolAuthorization(
        tool=tool,
        effect=EffectClass.WRITE,
        constraints=(
            ArgumentConstraint(
                argument="filename",
                operator=ConstraintOperator.EQUALS,
                value=filename,
            ),
        ),
        allow_additional_arguments=allow_additional_arguments,
        approval_required=approval_required,
        justification="Delegation-contract test authorization.",
    )


def parent_contract(*, delegation_allowed: bool = True):
    return compile_explicit_contract(
        principal="orchestrator-agent",
        objective="Create one authorized report.",
        permitted_tools=(authorization(),),
        prohibited_tools=frozenset({"delete_file", "send_email"}),
        prohibited_effects=frozenset(
            {
                EffectClass.DELETE,
                EffectClass.COMMUNICATE,
            }
        ),
        delegation_allowed=delegation_allowed,
    )


def expect_denial(**kwargs):
    decision = evaluate_delegation_request(**kwargs)
    assert not decision.allowed
    assert decision.rule_id == "DELEGATION_AUTHORITY_VIOLATION"
    assert decision.delegation is None
    return decision


def main() -> None:
    parent = parent_contract()
    child_authorization = authorization()

    delegation = compile_delegated_contract(
        parent_contract=parent,
        child_principal="report-writer-agent",
        objective="Create the authorized report file.",
        permitted_tools=(child_authorization,),
        depth=1,
        max_depth=2,
        child_delegation_allowed=True,
    )

    assert delegation.parent_principal == "orchestrator-agent"
    assert delegation.child_principal == "report-writer-agent"
    assert delegation.child_contract.delegation_allowed
    assert delegation.child_contract.authorization_for("create_file")
    assert delegation.child_contract.authorization_for("delete_file") is None
    assert "delete_file" in delegation.child_contract.prohibited_tools
    assert delegation.fingerprint() == delegation.fingerprint()

    second = compile_delegated_contract(
        parent_contract=parent,
        child_principal="report-writer-agent",
        objective="Create the authorized report file.",
        permitted_tools=(child_authorization,),
        depth=1,
        max_depth=2,
        child_delegation_allowed=True,
    )
    assert delegation.delegation_id == second.delegation_id
    assert delegation.fingerprint() == second.fingerprint()

    grandchild = compile_subdelegated_contract(
        parent_delegation=delegation,
        child_principal="storage-worker-agent",
        objective="Create the authorized report without further delegation.",
        permitted_tools=(child_authorization,),
        child_delegation_allowed=False,
    )
    assert grandchild.depth == 2
    assert grandchild.max_depth == 2
    assert grandchild.parent_delegation_id == delegation.delegation_id
    assert not grandchild.child_contract.delegation_allowed

    grandchild_denial = evaluate_delegation_request(
        parent_contract=grandchild.child_contract,
        child_principal="forbidden-great-grandchild",
        objective="Must not be created.",
        permitted_tools=(child_authorization,),
        depth=3,
        max_depth=2,
    )
    assert not grandchild_denial.allowed

    expect_denial(
        parent_contract=parent_contract(delegation_allowed=False),
        child_principal="child",
        objective="Unauthorized delegation.",
        permitted_tools=(child_authorization,),
    )

    expect_denial(
        parent_contract=parent,
        child_principal="child",
        objective="Request an unauthorized tool.",
        permitted_tools=(
            ToolAuthorization(
                tool="send_email",
                effect=EffectClass.COMMUNICATE,
                justification="Must be denied.",
            ),
        ),
    )

    expect_denial(
        parent_contract=parent,
        child_principal="child",
        objective="Loosen the parent filename constraint.",
        permitted_tools=(authorization(filename="other.txt"),),
    )

    expect_denial(
        parent_contract=parent,
        child_principal="child",
        objective="Permit unbounded additional arguments.",
        permitted_tools=(
            authorization(allow_additional_arguments=True),
        ),
    )

    expect_denial(
        parent_contract=parent,
        child_principal="child",
        objective="Delegate beyond the maximum depth.",
        permitted_tools=(child_authorization,),
        depth=1,
        max_depth=1,
        child_delegation_allowed=True,
    )

    approval_parent = compile_explicit_contract(
        principal="approval-parent",
        objective="Approval-protected report.",
        permitted_tools=(authorization(approval_required=True),),
        delegation_allowed=True,
    )
    expect_denial(
        parent_contract=approval_parent,
        child_principal="child",
        objective="Remove required approval.",
        permitted_tools=(authorization(approval_required=False),),
    )

    allowed, _ = authorization_is_attenuated(
        child_authorization,
        child_authorization,
    )
    assert allowed

    print("PASS: Valid child authority is an exact or stricter subset.")
    print("PASS: Unauthorized tools and weakened constraints fail closed.")
    print("PASS: Approval requirements cannot be removed by delegation.")
    print("PASS: Delegation depth and subdelegation are bounded.")
    print("PASS: Subdelegation depth is derived from parent provenance.")
    print("PASS: Delegation identifiers and fingerprints are deterministic.")


if __name__ == "__main__":
    main()
