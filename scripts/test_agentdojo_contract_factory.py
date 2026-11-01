"""Offline tests for the AgentDojo read-only contract factory."""

from __future__ import annotations

import json
from pathlib import Path

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_contract_factory import (
    WORKSPACE_SIDE_EFFECT_TOOLS,
    UnsupportedTaskContract,
    compile_workspace_read_only_contract,
)


EXPECTED_READ_ONLY_TASKS = {
    "user_task_0",
    "user_task_1",
    "user_task_2",
    "user_task_3",
    "user_task_5",
    "user_task_10",
    "user_task_11",
    "user_task_14",
    "user_task_16",
    "user_task_17",
    "user_task_22",
    "user_task_23",
    "user_task_24",
    "user_task_26",
    "user_task_27",
    "user_task_28",
    "user_task_30",
    "user_task_39",
}


def main() -> None:
    suite = get_suite("v1.2", "workspace")
    environment = (
        suite.load_and_inject_default_environment({})
    )

    compiled = {}
    rejected = {}

    for task_id, task in suite.user_tasks.items():
        try:
            result = compile_workspace_read_only_contract(
                task=task,
                base_environment=environment,
            )

            compiled[task_id] = result

        except UnsupportedTaskContract as error:
            rejected[task_id] = str(error)

    compiled_ids = set(compiled)

    assert compiled_ids == EXPECTED_READ_ONLY_TASKS, (
        "Unexpected read-only task coverage.\n"
        f"Missing: {sorted(EXPECTED_READ_ONLY_TASKS - compiled_ids)}\n"
        f"Unexpected: {sorted(compiled_ids - EXPECTED_READ_ONLY_TASKS)}"
    )

    for task_id, result in compiled.items():
        contract = result.contract

        assert contract.delegation_allowed is False

        for prohibited_tool in WORKSPACE_SIDE_EFFECT_TOOLS:
            assert contract.authorization_for(
                prohibited_tool
            ) is None

            assert prohibited_tool in contract.prohibited_tools

        for permitted_tool in result.permitted_tools:
            authorization = contract.authorization_for(
                permitted_tool
            )

            assert authorization is not None
            assert authorization.effect.value == "read"

        assert len(contract.fingerprint()) == 64

    calendar = compiled["user_task_0"]

    assert calendar.authorized_families == ("calendar",)

    assert set(calendar.permitted_tools) == {
        "get_current_day",
        "get_day_calendar_events",
        "search_calendar_events",
    }

    email = compiled["user_task_14"]

    assert email.authorized_families == ("email",)
    assert "search_emails" in email.permitted_tools
    assert "search_calendar_events" not in email.permitted_tools

    files = compiled["user_task_26"]

    assert files.authorized_families == ("files",)
    assert "list_files" in files.permitted_tools
    assert "send_email" not in files.permitted_tools

    multi_domain = compiled["user_task_23"]

    assert multi_domain.authorized_families == (
        "calendar",
        "email",
    )

    state_changing_task = suite.get_user_task_by_id(
        "user_task_6"
    )

    try:
        compile_workspace_read_only_contract(
            task=state_changing_task,
            base_environment=environment,
        )
    except UnsupportedTaskContract as error:
        assert "create_calendar_event" in str(error)
    else:
        raise AssertionError(
            "State-changing task was incorrectly accepted."
        )

    output = {
        "benchmark_version": "v1.2",
        "suite": "workspace",
        "compiled_task_count": len(compiled),
        "compiled_task_ids": sorted(compiled),
        "rejected_task_count": len(rejected),
        "rejected_tasks": rejected,
        "contracts": {
            task_id: result.to_dict()
            for task_id, result in compiled.items()
        },
    }

    output_path = Path(
        "outputs/workspace_read_only_contracts.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print(
        "PASS: Workspace read-only contracts compiled safely"
    )
    print(f"Compiled tasks: {len(compiled)}")
    print(f"Rejected tasks: {len(rejected)}")
    print(
        "Compiled task IDs:",
        sorted(compiled),
    )
    print(f"Artifact: {output_path}")


if __name__ == "__main__":
    main()
