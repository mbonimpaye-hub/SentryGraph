"""Deterministic authorization gate for SentryGraph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sentrygraph.contract_compiler import TaskContract


class Decision(str, Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"
    ISOLATE = "isolate"


@dataclass(frozen=True)
class PolicyDecision:
    """Result of evaluating one proposed action."""

    decision: Decision
    rule_id: str
    reason: str
    contract_id: str | None = None

    @property
    def permitted(self) -> bool:
        return self.decision == Decision.ALLOW


class DeterministicPolicyGate:
    """Enforce non-overridable task-contract restrictions."""

    def __init__(
        self,
        denied_tools: set[str] | None = None,
        contract: TaskContract | None = None,
    ) -> None:
        self.denied_tools = frozenset(denied_tools or set())
        self.contract = contract

    @classmethod
    def from_environment(
        cls,
        contract: TaskContract | None = None,
    ) -> "DeterministicPolicyGate":
        """Construct a gate using environment-level prohibitions."""

        raw_tools = os.getenv("SENTRYGRAPH_DENY_TOOLS", "")

        denied_tools = {
            tool.strip()
            for tool in raw_tools.split(",")
            if tool.strip()
        }

        return cls(
            denied_tools=denied_tools,
            contract=contract,
        )

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate a proposed action before execution."""

        contract_id = (
            self.contract.contract_id
            if self.contract is not None
            else None
        )

        # Environment policy has highest priority.
        if tool in self.denied_tools:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="DENY_TOOL",
                reason=(
                    f"Tool '{tool}' is prohibited by the active "
                    "environment policy."
                ),
                contract_id=contract_id,
            )

        # Preserve the original default behavior when no task
        # contract has been supplied.
        if self.contract is None:
            return PolicyDecision(
                decision=Decision.ALLOW,
                rule_id="DEFAULT_ALLOW",
                reason="No deterministic prohibition matched.",
            )

        # Explicit contract prohibition cannot be overridden.
        if tool in self.contract.prohibited_tools:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="CONTRACT_PROHIBITED_TOOL",
                reason=(
                    f"Tool '{tool}' is explicitly prohibited by "
                    f"contract '{self.contract.contract_id}'."
                ),
                contract_id=contract_id,
            )

        authorization = self.contract.authorization_for(tool)

        # Closed-world rule: anything not explicitly permitted is denied.
        if authorization is None:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="TOOL_OUTSIDE_CONTRACT",
                reason=(
                    f"Tool '{tool}' is not authorized by contract "
                    f"'{self.contract.contract_id}'."
                ),
                contract_id=contract_id,
            )

        if authorization.effect in self.contract.prohibited_effects:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="PROHIBITED_EFFECT",
                reason=(
                    f"Effect class '{authorization.effect.value}' "
                    f"is prohibited by contract "
                    f"'{self.contract.contract_id}'."
                ),
                contract_id=contract_id,
            )

        # Until the human-approval workflow is implemented,
        # approval-required actions fail closed.
        if authorization.approval_required:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="APPROVAL_REQUIRED",
                reason=(
                    f"Tool '{tool}' requires human approval, but "
                    "no valid approval is attached."
                ),
                contract_id=contract_id,
            )

        if not authorization.arguments_authorized(arguments):
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="ARGUMENT_CONSTRAINT_VIOLATION",
                reason=(
                    f"Arguments for tool '{tool}' violate contract "
                    f"'{self.contract.contract_id}'."
                ),
                contract_id=contract_id,
            )

        return PolicyDecision(
            decision=Decision.ALLOW,
            rule_id="CONTRACT_ALLOW",
            reason=(
                f"Tool '{tool}' and its arguments satisfy contract "
                f"'{self.contract.contract_id}'."
            ),
            contract_id=contract_id,
        )