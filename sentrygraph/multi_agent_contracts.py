"""Fail-closed hierarchical delegation contracts for SentryGraph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from sentrygraph.contract_compiler import (
    EffectClass,
    TaskContract,
    ToolAuthorization,
    compile_explicit_contract,
)


class DelegationContractError(ValueError):
    """Raised when requested child authority exceeds parent authority."""


def _authorization_signature(
    authorization: ToolAuthorization,
) -> dict[str, Any]:
    value = authorization.to_dict()
    value.pop("justification", None)
    return value


def _constraint_signatures(
    authorization: ToolAuthorization,
) -> set[str]:
    return {
        json.dumps(
            {
                "argument": constraint.argument,
                "operator": getattr(
                    constraint.operator,
                    "value",
                    str(constraint.operator),
                ),
                "value": constraint.value,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for constraint in authorization.constraints
    }


def authorization_is_attenuated(
    parent: ToolAuthorization,
    child: ToolAuthorization,
) -> tuple[bool, str]:
    """Return whether child authorization is no broader than parent."""

    if child.tool != parent.tool:
        return False, "Tool names differ."

    if child.effect != parent.effect:
        return False, "The child changes the authorized effect class."

    if parent.approval_required and not child.approval_required:
        return False, "The child removes a required approval."

    if (
        not parent.allow_additional_arguments
        and child.allow_additional_arguments
    ):
        return False, "The child permits additional arguments."

    parent_constraints = _constraint_signatures(parent)
    child_constraints = _constraint_signatures(child)

    if not parent_constraints.issubset(child_constraints):
        return False, "The child removes or changes a parent constraint."

    return True, "Child authorization is an attenuation of the parent."


@dataclass(frozen=True)
class DelegationContract:
    """Auditable parent-to-child authority transfer."""

    delegation_id: str
    parent_contract_id: str
    parent_contract_fingerprint: str
    parent_principal: str
    child_principal: str
    objective: str
    child_contract: TaskContract
    depth: int
    max_depth: int
    parent_delegation_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def semantics(self) -> dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "parent_contract_id": self.parent_contract_id,
            "parent_contract_fingerprint": self.parent_contract_fingerprint,
            "parent_principal": self.parent_principal,
            "child_principal": self.child_principal,
            "objective": self.objective,
            "child_contract": {
                key: value
                for key, value in self.child_contract.to_dict().items()
                if key != "created_at"
            },
            "depth": self.depth,
            "max_depth": self.max_depth,
            "parent_delegation_id": self.parent_delegation_id,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.semantics(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = self.semantics()
        value["created_at"] = self.created_at
        value["delegation_fingerprint"] = self.fingerprint()
        return value


@dataclass(frozen=True)
class DelegationDecision:
    allowed: bool
    rule_id: str
    reason: str
    delegation: DelegationContract | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "delegation": (
                self.delegation.to_dict() if self.delegation else None
            ),
        }


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def compile_delegated_contract(
    *,
    parent_contract: TaskContract,
    child_principal: str,
    objective: str,
    permitted_tools: tuple[ToolAuthorization, ...],
    additional_prohibited_tools: Iterable[str] = (),
    additional_prohibited_effects: Iterable[EffectClass] = (),
    child_delegation_allowed: bool = False,
    depth: int = 1,
    max_depth: int = 1,
    parent_delegation_id: str | None = None,
) -> DelegationContract:
    """Compile a child contract whose authority cannot exceed its parent."""

    if not parent_contract.delegation_allowed:
        raise DelegationContractError(
            "Parent contract does not authorize delegation."
        )

    if EffectClass.DELEGATE in parent_contract.prohibited_effects:
        raise DelegationContractError(
            "Parent contract explicitly prohibits delegation effects."
        )

    if not child_principal.strip():
        raise DelegationContractError("Child principal must be non-empty.")

    if child_principal == parent_contract.principal:
        raise DelegationContractError(
            "Parent and child principals must be distinct."
        )

    if depth < 1 or max_depth < 1 or depth > max_depth:
        raise DelegationContractError(
            "Delegation depth must satisfy 1 <= depth <= max_depth."
        )

    if child_delegation_allowed and depth >= max_depth:
        raise DelegationContractError(
            "A child at the maximum depth cannot subdelegate."
        )

    seen_tools: set[str] = set()

    for child_authorization in permitted_tools:
        if child_authorization.tool in seen_tools:
            raise DelegationContractError(
                f"Duplicate child tool authorization: "
                f"{child_authorization.tool!r}."
            )
        seen_tools.add(child_authorization.tool)

        parent_authorization = parent_contract.authorization_for(
            child_authorization.tool
        )
        if parent_authorization is None:
            raise DelegationContractError(
                f"Child requests tool {child_authorization.tool!r} "
                "outside parent authority."
            )

        attenuated, reason = authorization_is_attenuated(
            parent_authorization,
            child_authorization,
        )
        if not attenuated:
            raise DelegationContractError(
                f"Unsafe authorization for "
                f"{child_authorization.tool!r}: {reason}"
            )

    parent_permitted_names = {
        authorization.tool
        for authorization in parent_contract.permitted_tools
    }
    omitted_parent_tools = parent_permitted_names - seen_tools

    prohibited_tools = frozenset(
        set(parent_contract.prohibited_tools)
        | set(additional_prohibited_tools)
        | omitted_parent_tools
    )
    prohibited_effects = frozenset(
        set(parent_contract.prohibited_effects)
        | set(additional_prohibited_effects)
    )

    for authorization in permitted_tools:
        if authorization.tool in prohibited_tools:
            raise DelegationContractError(
                f"Child tool {authorization.tool!r} is also prohibited."
            )
        if authorization.effect in prohibited_effects:
            raise DelegationContractError(
                f"Child effect {authorization.effect.value!r} "
                "is prohibited."
            )

    child_contract = compile_explicit_contract(
        principal=child_principal,
        objective=objective,
        permitted_tools=permitted_tools,
        prohibited_tools=prohibited_tools,
        prohibited_effects=prohibited_effects,
        delegation_allowed=child_delegation_allowed,
    )

    source = {
        "parent_contract_fingerprint": parent_contract.fingerprint(),
        "parent_principal": parent_contract.principal,
        "child_contract_fingerprint": child_contract.fingerprint(),
        "child_principal": child_principal,
        "depth": depth,
        "max_depth": max_depth,
        "parent_delegation_id": parent_delegation_id,
    }
    delegation_id = "delegation-" + hashlib.sha256(
        json.dumps(
            source,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    return DelegationContract(
        delegation_id=delegation_id,
        parent_contract_id=parent_contract.contract_id,
        parent_contract_fingerprint=parent_contract.fingerprint(),
        parent_principal=parent_contract.principal,
        child_principal=child_principal,
        objective=objective,
        child_contract=child_contract,
        depth=depth,
        max_depth=max_depth,
        parent_delegation_id=parent_delegation_id,
    )


def evaluate_delegation_request(**kwargs: Any) -> DelegationDecision:
    """Return an auditable allow/deny result without throwing to callers."""

    try:
        delegation = compile_delegated_contract(**kwargs)
    except DelegationContractError as error:
        return DelegationDecision(
            allowed=False,
            rule_id="DELEGATION_AUTHORITY_VIOLATION",
            reason=str(error),
        )

    return DelegationDecision(
        allowed=True,
        rule_id="DELEGATION_ATTENUATED_ALLOW",
        reason=(
            "Child authority is a verified attenuation of the parent "
            "contract."
        ),
        delegation=delegation,
    )


def compile_subdelegated_contract(
    *,
    parent_delegation: DelegationContract,
    child_principal: str,
    objective: str,
    permitted_tools: tuple[ToolAuthorization, ...],
    additional_prohibited_tools: Iterable[str] = (),
    additional_prohibited_effects: Iterable[EffectClass] = (),
    child_delegation_allowed: bool = False,
) -> DelegationContract:
    """Compile a grandchild while binding depth to its parent record."""

    return compile_delegated_contract(
        parent_contract=parent_delegation.child_contract,
        child_principal=child_principal,
        objective=objective,
        permitted_tools=permitted_tools,
        additional_prohibited_tools=additional_prohibited_tools,
        additional_prohibited_effects=additional_prohibited_effects,
        child_delegation_allowed=child_delegation_allowed,
        depth=parent_delegation.depth + 1,
        max_depth=parent_delegation.max_depth,
        parent_delegation_id=parent_delegation.delegation_id,
    )

