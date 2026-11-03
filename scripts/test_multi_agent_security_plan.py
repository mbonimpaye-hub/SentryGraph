"""Audit Phase 2E scenario coverage and frozen-plan determinism."""

from __future__ import annotations

from collections import Counter

from plan_multi_agent_security_experiments import build_plan
from sentrygraph.multi_agent_scenarios import scenario_catalog


def main() -> None:
    catalog = scenario_catalog()
    ids = [scenario.scenario_id for scenario in catalog]
    assert len(ids) == len(set(ids))
    assert sum(s.category == "benign" for s in catalog) >= 1
    assert sum(s.category == "attack" for s in catalog) >= 8

    required_families = {
        "authorized_handoff",
        "argument_tampering",
        "cross_tool_scope",
        "authority_amplification",
        "constraint_weakening",
        "subdelegation_bypass",
        "depth_bypass",
        "revocation_bypass",
        "identity_scope_confusion",
        "handoff_prompt_injection",
    }
    families = {scenario.family for scenario in catalog}
    assert required_families.issubset(families)

    first = build_plan(3)
    second = build_plan(3)
    assert first == second
    assert first["plan_id"] == second["plan_id"]

    counts = Counter(
        (cell["scenario_id"], cell["trial"])
        for cell in first["cells"]
    )
    assert set(counts.values()) == {2}
    for scenario_id, trial in counts:
        conditions = {
            cell["condition"]
            for cell in first["cells"]
            if cell["scenario_id"] == scenario_id
            and cell["trial"] == trial
        }
        assert conditions == {"baseline", "sentrygraph"}

    fingerprints = first["catalog"]["scenario_fingerprints"]
    assert set(fingerprints) == set(ids)
    assert all(len(value) == 64 for value in fingerprints.values())

    print("PASS: Multi-agent scenario identifiers are unique.")
    print("PASS: Required benign and adversarial families are covered.")
    print("PASS: Every trial has a matched baseline/SentryGraph pair.")
    print("PASS: Scenario fingerprints are frozen into plan identity.")
    print("PASS: Repeated plan construction is byte-semantics deterministic.")
    print("Plan ID:", first["plan_id"])
    print("Scenarios:", len(catalog))
    print("Families:", len(families))
    print("Cells:", len(first["cells"]))


if __name__ == "__main__":
    main()
