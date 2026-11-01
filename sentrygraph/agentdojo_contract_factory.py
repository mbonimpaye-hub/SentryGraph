"""Task-specific AgentDojo contract construction for SentryGraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
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


WORKSPACE_SIDE_EFFECT_CLASSES: dict[str, EffectClass] = {
    "send_email": EffectClass.COMMUNICATE,
    "delete_email": EffectClass.DELETE,
    "create_calendar_event": EffectClass.WRITE,
    "cancel_calendar_event": EffectClass.DELETE,
    "reschedule_calendar_event": EffectClass.WRITE,
    "add_calendar_event_participants": EffectClass.WRITE,
    "append_to_file": EffectClass.WRITE,
    "create_file": EffectClass.WRITE,
    "delete_file": EffectClass.DELETE,
    "share_file": EffectClass.COMMUNICATE,
}


WORKSPACE_READ_TOOLS = frozenset().union(
    *WORKSPACE_TOOL_FAMILIES.values()
)


TOOL_TO_FAMILY = {
    tool: family
    for family, tools in WORKSPACE_TOOL_FAMILIES.items()
    for tool in tools
}


# Write tasks are admitted explicitly, not inferred automatically.
# Each tuple lists the read domains needed to discover the authorized
# target before executing the exact benchmark-defined side effect.
SUPPORTED_WRITE_TASKS: dict[str, tuple[str, ...]] = {
    "user_task_12": ("calendar",),
    "user_task_32": ("files",),
    "user_task_33": ("contacts", "files"),
    "user_task_35": ("files",),
}


@dataclass(frozen=True)
class GroundTruthCall:
    """Normalized benchmark ground-truth tool call."""

    tool: str
    arguments: dict[str, Any]


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
        authorized_side_effect_tools = sorted(
            set(self.permitted_tools)
            & WORKSPACE_SIDE_EFFECT_TOOLS
        )

        return {
            "task_id": self.task_id,
            "contract_id": self.contract.contract_id,
            "contract_fingerprint": (
                self.contract.fingerprint()
            ),
            "ground_truth_tools": list(
                self.ground_truth_tools
            ),
            "authorized_families": list(
                self.authorized_families
            ),
            "permitted_tools": list(
                self.permitted_tools
            ),
            "authorized_side_effect_tools": (
                authorized_side_effect_tools
            ),
            "rejected_side_effect_tools": list(
                self.rejected_side_effect_tools
            ),
            "contract": self.contract.to_dict(),
        }


class UnsupportedTaskContract(ValueError):
    """Raised when the factory cannot safely compile a task."""


def _ordered_unique(
    values: Iterable[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _ground_truth_calls(
    task: Any,
    base_environment: Any,
) -> tuple[GroundTruthCall, ...]:
    task_environment = task.init_environment(
        base_environment.model_copy(deep=True)
    )

    raw_calls = task.ground_truth(
        task_environment
    )

    normalized = []

    for call in raw_calls:
        function = getattr(
            call,
            "function",
            None,
        )

        arguments = getattr(
            call,
            "args",
            None,
        )

        if not function:
            raise UnsupportedTaskContract(
                f"Task {task.ID} contains a ground-truth "
                "call without a function name."
            )

        if arguments is None:
            raise UnsupportedTaskContract(
                f"Task {task.ID} contains ground-truth "
                f"tool {function!r} without arguments."
            )

        if not isinstance(arguments, dict):
            try:
                arguments = dict(arguments)
            except Exception as error:
                raise UnsupportedTaskContract(
                    f"Task {task.ID} contains non-mapping "
                    f"arguments for tool {function!r}."
                ) from error

        normalized.append(
            GroundTruthCall(
                tool=str(function),
                arguments=dict(arguments),
            )
        )

    if not normalized:
        raise UnsupportedTaskContract(
            f"Task {task.ID} has no ground-truth "
            "tool calls."
        )

    return tuple(normalized)


def _ground_truth_tools(
    task: Any,
    base_environment: Any,
) -> tuple[str, ...]:
    calls = _ground_truth_calls(
        task,
        base_environment,
    )

    return _ordered_unique(
        call.tool
        for call in calls
    )


def _families_for_tools(
    tools: Iterable[str],
) -> tuple[str, ...]:
    families = []

    for tool in tools:
        family = TOOL_TO_FAMILY.get(tool)

        if family is None:
            raise UnsupportedTaskContract(
                f"Read tool {tool!r} has no registered "
                "tool family."
            )

        families.append(family)

    return tuple(
        sorted(set(families))
    )


def _read_authorizations(
    families: Iterable[str],
    task_id: str,
) -> tuple[ToolAuthorization, ...]:
    permitted_tool_names = sorted(
        {
            tool
            for family in families
            for tool in WORKSPACE_TOOL_FAMILIES[family]
        }
    )

    return tuple(
        ToolAuthorization(
            tool=tool,
            effect=EffectClass.READ,
            justification=(
                f"Read-only {TOOL_TO_FAMILY[tool]} "
                f"discovery is authorized for "
                f"benchmark task {task_id}."
            ),
        )
        for tool in permitted_tool_names
    )


def _side_effect_authorization(
    task_id: str,
    call: GroundTruthCall,
) -> ToolAuthorization:
    effect = WORKSPACE_SIDE_EFFECT_CLASSES.get(
        call.tool
    )

    if effect is None:
        raise UnsupportedTaskContract(
            f"Task {task_id} uses unsupported "
            f"side-effect tool {call.tool!r}."
        )

    constraints = tuple(
        ArgumentConstraint(
            argument=argument,
            operator=ConstraintOperator.EQUALS,
            value=value,
        )
        for argument, value
        in sorted(call.arguments.items())
    )

    return ToolAuthorization(
        tool=call.tool,
        effect=effect,
        constraints=constraints,
        allow_additional_arguments=False,
        justification=(
            f"Exact benchmark-defined side effect "
            f"authorized for task {task_id}."
        ),
    )


def compile_workspace_read_only_contract(
    task: Any,
    base_environment: Any,
    principal: str = "agentdojo-workspace-user",
) -> ContractFactoryResult:
    """Compile an oracle contract for a read-only workspace task.

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
            f"Task {task.ID} is not read-only; its "
            f"ground-truth uses side-effect tools: "
            f"{list(side_effect_tools)}"
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

    read_authorizations = _read_authorizations(
        families,
        task.ID,
    )

    permitted_tool_names = tuple(
        authorization.tool
        for authorization in read_authorizations
    )

    contract = compile_explicit_contract(
        principal=principal,
        objective=task.PROMPT,
        permitted_tools=read_authorizations,
        prohibited_tools=WORKSPACE_SIDE_EFFECT_TOOLS,
        prohibited_effects=frozenset(
            {
                EffectClass.WRITE,
                EffectClass.DELETE,
                EffectClass.COMMUNICATE,
                EffectClass.DELEGATE,
                EffectClass.UNKNOWN,
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
        rejected_side_effect_tools=tuple(
            sorted(WORKSPACE_SIDE_EFFECT_TOOLS)
        ),
    )


def compile_workspace_write_contract(
    task: Any,
    base_environment: Any,
    principal: str = "agentdojo-workspace-user",
) -> ContractFactoryResult:
    """Compile an exact oracle contract for a supported write task."""

    if task.ID not in SUPPORTED_WRITE_TASKS:
        raise UnsupportedTaskContract(
            f"Write task {task.ID} is not in the "
            "explicit supported-write-task registry."
        )

    calls = _ground_truth_calls(
        task,
        base_environment,
    )

    side_effect_calls = tuple(
        call
        for call in calls
        if call.tool in WORKSPACE_SIDE_EFFECT_TOOLS
    )

    if not side_effect_calls:
        raise UnsupportedTaskContract(
            f"Registered write task {task.ID} has no "
            "side-effecting ground-truth call."
        )

    side_effect_tool_names = [
        call.tool
        for call in side_effect_calls
    ]

    duplicate_side_effect_tools = sorted(
        {
            tool
            for tool in side_effect_tool_names
            if side_effect_tool_names.count(tool) > 1
        }
    )

    if duplicate_side_effect_tools:
        raise UnsupportedTaskContract(
            f"Task {task.ID} invokes the same side-effect "
            "tool multiple times, which cannot be represented "
            "safely by the current authorization model: "
            f"{duplicate_side_effect_tools}"
        )

    unknown_ground_truth_tools = sorted(
        {
            call.tool
            for call in calls
        }
        - WORKSPACE_READ_TOOLS
        - WORKSPACE_SIDE_EFFECT_TOOLS
    )

    if unknown_ground_truth_tools:
        raise UnsupportedTaskContract(
            f"Task {task.ID} uses unclassified tools: "
            f"{unknown_ground_truth_tools}"
        )

    families = tuple(
        sorted(
            set(
                SUPPORTED_WRITE_TASKS[task.ID]
            )
        )
    )

    unknown_families = sorted(
        set(families)
        - set(WORKSPACE_TOOL_FAMILIES)
    )

    if unknown_families:
        raise UnsupportedTaskContract(
            f"Task {task.ID} has unknown read families: "
            f"{unknown_families}"
        )

    read_authorizations = _read_authorizations(
        families,
        task.ID,
    )

    side_effect_authorizations = tuple(
        _side_effect_authorization(
            task.ID,
            call,
        )
        for call in side_effect_calls
    )

    permitted_authorizations = (
        read_authorizations
        + side_effect_authorizations
    )

    permitted_tool_names = tuple(
        authorization.tool
        for authorization in permitted_authorizations
    )

    authorized_side_effect_tools = frozenset(
        side_effect_tool_names
    )

    prohibited_side_effect_tools = (
        WORKSPACE_SIDE_EFFECT_TOOLS
        - authorized_side_effect_tools
    )

    authorized_effects = {
        authorization.effect
        for authorization in side_effect_authorizations
    }

    all_side_effect_classes = {
        EffectClass.WRITE,
        EffectClass.COMMUNICATE,
        EffectClass.DELETE,
        EffectClass.DELEGATE,
        EffectClass.UNKNOWN,
    }

    prohibited_effects = frozenset(
        all_side_effect_classes
        - authorized_effects
    )

    contract = compile_explicit_contract(
        principal=principal,
        objective=task.PROMPT,
        permitted_tools=permitted_authorizations,
        prohibited_tools=frozenset(
            prohibited_side_effect_tools
        ),
        prohibited_effects=prohibited_effects,
        delegation_allowed=False,
    )

    return ContractFactoryResult(
        contract=contract,
        task_id=task.ID,
        ground_truth_tools=_ordered_unique(
            call.tool
            for call in calls
        ),
        authorized_families=families,
        permitted_tools=permitted_tool_names,
        rejected_side_effect_tools=tuple(
            sorted(prohibited_side_effect_tools)
        ),
    )


def compile_workspace_contract(
    task: Any,
    base_environment: Any,
    principal: str = "agentdojo-workspace-user",
) -> ContractFactoryResult:
    """Compile either a read-only or supported write contract."""

    if task.ID in SUPPORTED_WRITE_TASKS:
        return compile_workspace_write_contract(
            task=task,
            base_environment=base_environment,
            principal=principal,
        )

    return compile_workspace_read_only_contract(
        task=task,
        base_environment=base_environment,
        principal=principal,
    )
