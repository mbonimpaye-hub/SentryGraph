"""Tests for exact AgentDojo write-task contracts."""

from __future__ import annotations

from copy import deepcopy

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_contract_factory import (
    SUPPORTED_WRITE_TASKS,
    UnsupportedTaskContract,
    compile_workspace_contract,
    compile_workspace_write_contract,
)
from sentrygraph.policy_gate import (
    Decision,
    DeterministicPolicyGate,
)


EXPECTED_SIDE_EFFECTS = {
    "user_task_12": {
        "create_calendar_event",
    },
    "user_task_32": {
        "create_file",
        "share_file",
    },
    "user_task_33": {
        "send_email",
    },
    "user_task_35": {
        "delete_file",
    },
}


def ground_truth_calls(
    task,
    base_environment,
):
    environment = task.init_environment(
        base_environment.model_copy(deep=True)
    )

    return task.ground_truth(environment)


def contract_semantics(contract):
    """Return contract content without volatile metadata."""

    semantics = contract.to_dict()
    semantics.pop("created_at", None)

    return semantics


def main() -> int:
    suite = get_suite(
        "v1.2",
        "workspace",
    )

    base_environment = (
        suite.load_and_inject_default_environment({})
    )

    tested = []

    for task_id in sorted(SUPPORTED_WRITE_TASKS):
        task = suite.get_user_task_by_id(
            task_id
        )

        result = compile_workspace_write_contract(
            task=task,
            base_environment=base_environment,
        )

        generic_result = compile_workspace_contract(
            task=task,
            base_environment=base_environment,
        )

        assert (
            contract_semantics(result.contract)
            == contract_semantics(
                generic_result.contract
            )
        )

        gate = DeterministicPolicyGate(
            contract=result.contract
        )

        calls = ground_truth_calls(
            task,
            base_environment,
        )

        observed_side_effects = set()

        for call in calls:
            if call.function not in EXPECTED_SIDE_EFFECTS[task_id]:
                continue

            observed_side_effects.add(
                call.function
            )

            original_arguments = dict(
                call.args
            )

            allowed = gate.evaluate(
                call.function,
                original_arguments,
            )

            assert allowed.decision == Decision.ALLOW, (
                task_id,
                call.function,
                allowed,
            )

            additional_arguments = deepcopy(
                original_arguments
            )

            additional_arguments[
                "__unexpected_argument__"
            ] = "not-authorized"

            unexpected = gate.evaluate(
                call.function,
                additional_arguments,
            )

            assert unexpected.decision == Decision.DENY
            assert (
                unexpected.rule_id
                == "ARGUMENT_CONSTRAINT_VIOLATION"
            )

            modified_arguments = deepcopy(
                original_arguments
            )

            first_argument = next(
                iter(modified_arguments)
            )

            modified_arguments[first_argument] = (
                "__unauthorized_value__"
            )

            modified = gate.evaluate(
                call.function,
                modified_arguments,
            )

            assert modified.decision == Decision.DENY
            assert (
                modified.rule_id
                == "ARGUMENT_CONSTRAINT_VIOLATION"
            )

        assert (
            observed_side_effects
            == EXPECTED_SIDE_EFFECTS[task_id]
        )

        unrelated = gate.evaluate(
            "delete_email",
            {
                "message_id": "unauthorized",
            },
        )

        assert unrelated.decision == Decision.DENY

        tested.append(
            {
                "task_id": task_id,
                "authorized_side_effects": sorted(
                    observed_side_effects
                ),
                "permitted_tools": list(
                    result.permitted_tools
                ),
                "prohibited_side_effects": list(
                    result.rejected_side_effect_tools
                ),
            }
        )

    # Confirm that the generic dispatcher preserves read-only behavior.
    read_task = suite.get_user_task_by_id(
        "user_task_0"
    )

    read_result = compile_workspace_contract(
        task=read_task,
        base_environment=base_environment,
    )

    read_gate = DeterministicPolicyGate(
        contract=read_result.contract
    )

    denied_write = read_gate.evaluate(
        "send_email",
        {
            "recipients": [
                "attacker@example.com"
            ],
            "subject": "Unauthorized",
            "body": "Unauthorized",
        },
    )

    assert denied_write.decision == Decision.DENY

    # Confirm that an unregistered write task fails closed.
    unsupported_task = suite.get_user_task_by_id(
        "user_task_13"
    )

    try:
        compile_workspace_write_contract(
            task=unsupported_task,
            base_environment=base_environment,
        )
    except UnsupportedTaskContract:
        pass
    else:
        raise AssertionError(
            "Unsupported write task did not fail closed."
        )

    print(
        "PASS: Exact write-task contracts compiled "
        "and failed closed"
    )
    print(
        "Supported write tasks:",
        sorted(SUPPORTED_WRITE_TASKS),
    )

    for record in tested:
        print()
        print(
            f"Task: {record['task_id']}"
        )
        print(
            "  Authorized side effects:",
            record["authorized_side_effects"],
        )
        print(
            "  Permitted tools:",
            record["permitted_tools"],
        )
        print(
            "  Prohibited side effects:",
            record[
                "prohibited_side_effects"
            ],
        )

    print()
    print(
        "Verified properties:"
    )
    print(
        "  - Ground-truth side effects are allowed"
    )
    print(
        "  - Modified arguments are denied"
    )
    print(
        "  - Additional arguments are denied"
    )
    print(
        "  - Unrelated side effects are denied"
    )
    print(
        "  - Read-only behavior is preserved"
    )
    print(
        "  - Unsupported write tasks fail closed"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
