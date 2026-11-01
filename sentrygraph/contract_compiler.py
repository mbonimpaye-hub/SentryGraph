"""Task-contract representation and compilation for SentryGraph."""

from __future__ import annotations

import hashlib
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EffectClass(str, Enum):
    READ = "read"
    WRITE = "write"
    COMMUNICATE = "communicate"
    DELETE = "delete"
    DELEGATE = "delegate"
    UNKNOWN = "unknown"


class ConstraintOperator(str, Enum):
    EQUALS = "equals"
    ONE_OF = "one_of"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"


@dataclass(frozen=True)
class ArgumentConstraint:
    """Constraint applied to one tool argument."""

    argument: str
    operator: ConstraintOperator
    value: Any

    def satisfied_by(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        if self.argument not in arguments:
            return False

        observed = arguments[self.argument]

        if self.operator == ConstraintOperator.EQUALS:
            return observed == self.value

        if self.operator == ConstraintOperator.ONE_OF:
            return observed in self.value

        if self.operator == ConstraintOperator.PREFIX:
            return (
                isinstance(observed, str)
                and observed.startswith(str(self.value))
            )

        if self.operator == ConstraintOperator.SUFFIX:
            return (
                isinstance(observed, str)
                and observed.endswith(str(self.value))
            )

        if self.operator == ConstraintOperator.CONTAINS:
            return (
                isinstance(observed, str)
                and str(self.value) in observed
            )

        return False


@dataclass(frozen=True)
class ToolAuthorization:
    """Authorization conditions for one tool."""

    tool: str
    effect: EffectClass
    constraints: tuple[ArgumentConstraint, ...] = ()
    allow_additional_arguments: bool = True
    approval_required: bool = False
    justification: str = ""

    def arguments_authorized(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        if not all(
            constraint.satisfied_by(arguments)
            for constraint in self.constraints
        ):
            return False

        if not self.allow_additional_arguments:
            constrained_arguments = {
                constraint.argument
                for constraint in self.constraints
            }

            unexpected_arguments = (
                set(arguments)
                - constrained_arguments
            )

            # Agent tool schemas may materialize omitted optional
            # fields as explicit None values. Such fields have no
            # operational effect and are normalized as absent.
            if any(
                arguments[argument] is not None
                for argument in unexpected_arguments
            ):
                return False

        return True


@dataclass(frozen=True)
class TaskContract:
    """Machine-readable boundary of authorized agent behaviour."""

    contract_id: str
    principal: str
    objective: str
    permitted_tools: tuple[ToolAuthorization, ...]

    prohibited_tools: frozenset[str] = field(
        default_factory=frozenset
    )

    prohibited_effects: frozenset[EffectClass] = field(
        default_factory=frozenset
    )

    delegation_allowed: bool = False
    compiler: str = "explicit_policy"
    compiler_version: str = "0.3"

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def authorization_for(
        self,
        tool: str,
    ) -> ToolAuthorization | None:
        for authorization in self.permitted_tools:
            if authorization.tool == tool:
                return authorization

        return None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["prohibited_tools"] = sorted(
            self.prohibited_tools
        )

        result["prohibited_effects"] = sorted(
            effect.value
            for effect in self.prohibited_effects
        )

        for authorization in result["permitted_tools"]:
            authorization["effect"] = (
                authorization["effect"].value
            )

            for constraint in authorization["constraints"]:
                constraint["operator"] = (
                    constraint["operator"].value
                )

        return result

    def fingerprint(self) -> str:
        """Hash stable authorization semantics.

        Creation time is audit metadata and must not alter the
        semantic fingerprint of an otherwise identical contract.
        """

        semantics = self.to_dict()
        semantics.pop("created_at", None)

        canonical = json.dumps(
            semantics,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(canonical).hexdigest()


def compile_explicit_contract(
    *,
    principal: str,
    objective: str,
    permitted_tools: tuple[ToolAuthorization, ...],
    prohibited_tools: frozenset[str] | None = None,
    prohibited_effects: frozenset[EffectClass] | None = None,
    delegation_allowed: bool = False,
) -> TaskContract:
    """Compile a trusted explicit policy into a task contract."""

    source = {
        "principal": principal,
        "objective": objective,
        "permitted_tools": [
            authorization.tool
            for authorization in permitted_tools
        ],
        "prohibited_tools": sorted(
            prohibited_tools or set()
        ),
    }

    source_digest = hashlib.sha256(
        json.dumps(
            source,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    return TaskContract(
        contract_id=f"contract-{source_digest}",
        principal=principal,
        objective=objective,
        permitted_tools=permitted_tools,
        prohibited_tools=(
            prohibited_tools or frozenset()
        ),
        prohibited_effects=(
            prohibited_effects or frozenset()
        ),
        delegation_allowed=delegation_allowed,
    )
