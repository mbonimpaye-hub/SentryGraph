"""Task-specific AgentDojo contract construction for SentryGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sentrygraph.contract_compiler import (
    EffectClass,
    TaskContract,
    ToolAuthorization,
    compile_explicit_contract,
)


WORKSPACE_TOOL_FAMILIES: dict[str, frozenset[str]] = {
    "calendar": frozenset(
        {
            "get_current_day",
            "search_calendar_events",
            "get_day_calendar_events",
        }
    ),
    "email": frozenset(
        {
            "get_unread_emails",
            "get_sent_emails",
            "get_received_emails",
            "get_draft_emails",
            "search_emails",
        }
    ),
    "contacts": frozenset(
        {
            "search_contacts_by_name",
            "search_contacts_by_email",
        }
    ),
    "files": frozenset(
        {
            "list_files",
            "search_files",
            "search_files_by_filename",
            "get_file_by_id",
        }
    ),
}


WORKSPACE_SIDE_EFFECT_TOOLS = frozenset(
    {
        "send_email",
        "delete_email",
        "create_calendar_event",
        "cancel_calendar_event",
        "reschedule_calendar_event",
        "add_calendar_event_participants",
        "append_to_file",
        "create_file",
        "delete_file",
        "share_file",
    }
)


WORKSPACE_READ_TOOLS = frozenset().union(
    *WORKSPACE_TOOL_FAMILIES.values()
)


TOOL_TO_FAMILY = {
    tool: family
    for family, tools in WORKSPACE_TOOL_FAMILIES.items()
    for tool in tools
}


@dataclass(frozen=True)
class ContractFactoryResult:
    """Contract plus auditable compilation metadata."""

    contract: TaskContract
    task_id: str
    ground_truth_tools: tuple[str, ...]
    authorized_families: tuple[str, ...]
    permitted_tools: tuple[str, ...]
    rejected_side_effect_tools: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "contract_id": self.contract.contract_id,
            "contract_fingerprint": self.contract.fingerprint(),
            "ground_truth_tools": list(
                self.ground_truth_tools
            ),
            "authorized_families": list(
                self.authorized_families
            ),
            "permitted_tools": list(
                self.permitted_tools
            ),
            "rejected_side_effect_tools": list(
                self.rejected_side_effect_tools
            ),
            "contract": self.contract.to_dict(),
        }


class UnsupportedTaskContract(ValueError):
    """Raised when the initial factory cannot safely compile a task."""


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _ground_truth_tools(
    task: Any,
    base_environment: Any,
) -> tuple[str, ...]:
    task_environment = task.init_environment(
        base_environment.model_copy(deep=True)
    )

    calls = task.ground_truth(task_environment)

    tools = []

    for call in calls:
        function = getattr(call, "function", None)

        if not function:
            raise UnsupportedTaskContract(
                f"Task {task.ID} contains a ground-truth call "
                "without a function name."
            )

        tools.append(function)

    if not tools:
        raise UnsupportedTaskContract(
            f"Task {task.ID} has no ground-truth tool calls."
        )

    return _ordered_unique(tools)


def _families_for_tools(
    tools: Iterable[str],
) -> tuple[str, ...]:
    families = []

    for tool in tools:
        family = TOOL_TO_FAMILY.get(tool)

        if family is None:
            raise UnsupportedTaskContract(
                f"Read tool {tool!r} has no registered tool family."
            )

        families.append(family)

    return tuple(sorted(set(families)))


def compile_workspace_read_only_contract(
    task: Any,
    base_environment: Any,
    principal: str = "agentdojo-workspace-user",
) -> ContractFactoryResult:
    """Compile an evaluation contract for a read-only workspace task.

    The benchmark ground-truth trace identifies the required resource
    domains. The contract permits read-only discovery tools from those
    domains while prohibiting every workspace side-effecting tool.

    This is an evaluation-oracle compiler. It must not be described as
    a deployment-time natural-language policy compiler.
    """

    ground_truth_tools = _ground_truth_tools(
        task,
        base_environment,
    )

    side_effect_tools = tuple(
        sorted(
            set(ground_truth_tools)
            & WORKSPACE_SIDE_EFFECT_TOOLS
        )
    )

    if side_effect_tools:
        raise UnsupportedTaskContract(
            f"Task {task.ID} is not read-only; its ground-truth "
            f"uses side-effect tools: {list(side_effect_tools)}"
        )

    unknown_tools = tuple(
        sorted(
            set(ground_truth_tools)
            - WORKSPACE_READ_TOOLS
        )
    )

    if unknown_tools:
        raise UnsupportedTaskContract(
            f"Task {task.ID} uses unclassified tools: "
            f"{list(unknown_tools)}"
        )

    families = _families_for_tools(
        ground_truth_tools
    )

    permitted_tool_names = tuple(
        sorted(
            {
                tool
                for family in families
                for tool in WORKSPACE_TOOL_FAMILIES[family]
            }
        )
    )

    permitted_tools = tuple(
        ToolAuthorization(
            tool=tool,
            effect=EffectClass.READ,
            justification=(
                f"Read-only {TOOL_TO_FAMILY[tool]} discovery "
                f"is authorized for benchmark task {task.ID}."
            ),
        )
        for tool in permitted_tool_names
    )

    contract = compile_explicit_contract(
        principal=principal,
        objective=task.PROMPT,
        permitted_tools=permitted_tools,
        prohibited_tools=WORKSPACE_SIDE_EFFECT_TOOLS,
        prohibited_effects=frozenset(
            {
                EffectClass.WRITE,
                EffectClass.DELETE,
                EffectClass.COMMUNICATE,
                EffectClass.DELEGATE,
            }
        ),
        delegation_allowed=False,
    )

    return ContractFactoryResult(
        contract=contract,
        task_id=task.ID,
        ground_truth_tools=ground_truth_tools,
        authorized_families=families,
        permitted_tools=permitted_tool_names,
        rejected_side_effect_tools=side_effect_tools,
    )
