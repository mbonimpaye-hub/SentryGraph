"""Deterministic authorization gate for SentryGraph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"
    ISOLATE = "isolate"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    rule_id: str
    reason: str

    @property
    def permitted(self) -> bool:
        return self.decision == Decision.ALLOW


class DeterministicPolicyGate:
    """Enforce non-overridable tool-level policy rules."""

    def __init__(
        self,
        denied_tools: set[str] | None = None,
    ) -> None:
        self.denied_tools = frozenset(denied_tools or set())

    @classmethod
    def from_environment(cls) -> "DeterministicPolicyGate":
        raw_tools = os.getenv("SENTRYGRAPH_DENY_TOOLS", "")

        denied_tools = {
            tool.strip()
            for tool in raw_tools.split(",")
            if tool.strip()
        }

        return cls(denied_tools=denied_tools)

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate a proposed action before its execution."""
        if tool in self.denied_tools:
            return PolicyDecision(
                decision=Decision.DENY,
                rule_id="DENY_TOOL",
                reason=(
                    f"Tool '{tool}' is prohibited by the active "
                    "task policy."
                ),
            )

        return PolicyDecision(
            decision=Decision.ALLOW,
            rule_id="DEFAULT_ALLOW",
            reason="No deterministic prohibition matched.",
        )