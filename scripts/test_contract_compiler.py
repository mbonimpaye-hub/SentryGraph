"""Tests for SentryGraph's task-contract representation."""

from __future__ import annotations

import json

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)


def main() -> None:
    contract = compile_explicit_contract(
        principal="agentdojo-workspace-user",
        objective=(
            "Create the authorized report file and perform no "
            "other external side effects."
        ),
        permitted_tools=(
            ToolAuthorization(
                tool="get_current_day",
                effect=EffectClass.READ,
                justification=(
                    "The date is needed for the report."
                ),
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
                justification=(
                    "Only the named report file may be created."
                ),
            ),
        ),
        prohibited_tools=frozenset(
            {
                "send_email",
                "delete_email",
                "delete_file",
                "share_file",
            }
        ),
        prohibited_effects=frozenset(
            {
                EffectClass.DELETE,
                EffectClass.COMMUNICATE,
                EffectClass.DELEGATE,
            }
        ),
        delegation_allowed=False,
    )

    read_authorization = contract.authorization_for(
        "get_current_day"
    )
    assert read_authorization is not None
    assert read_authorization.arguments_authorized({})

    file_authorization = contract.authorization_for(
        "create_file"
    )
    assert file_authorization is not None

    assert file_authorization.arguments_authorized(
        {
            "filename": "authorized-report.txt",
            "content": "Authorized content",
        }
    )

    assert not file_authorization.arguments_authorized(
        {
            "filename": "unauthorized.txt",
            "content": "Unauthorized content",
        }
    )

    assert contract.authorization_for("send_email") is None
    assert "send_email" in contract.prohibited_tools
    assert contract.delegation_allowed is False

    first_fingerprint = contract.fingerprint()
    second_fingerprint = contract.fingerprint()

    assert first_fingerprint == second_fingerprint
    assert len(first_fingerprint) == 64

    print("PASS: SentryGraph compiled a valid task contract")
    print(f"Contract ID: {contract.contract_id}")
    print(f"Principal: {contract.principal}")
    print(f"Objective: {contract.objective}")
    print(
        "Permitted tools:",
        [
            authorization.tool
            for authorization in contract.permitted_tools
        ],
    )
    print(
        "Prohibited tools:",
        sorted(contract.prohibited_tools),
    )
    print(f"Delegation allowed: {contract.delegation_allowed}")
    print(f"Fingerprint: {contract.fingerprint()}")
    print("\nContract:")
    print(json.dumps(contract.to_dict(), indent=2))


if __name__ == "__main__":
    main()