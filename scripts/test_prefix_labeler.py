"""Semantic-agreement and leakage tests for Phase 1B prefix labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sentrygraph.prefix_labeler import (
    LEAKAGE_NODE_TYPES,
    LEAKAGE_RELATIONS,
    label_action,
    label_prefixes,
    load_contract,
    load_graph,
    load_jsonl,
    proposed_actions,
    recorded_proposal_decisions,
    sanitize_execution_prefix,
)


DEFAULT_ROOTS = (
    Path("outputs/experiments/c0dcdcb291672e3c"),
    Path("outputs/experiments/827f97129ce30ba1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit independent prefix labels against recorded evidence."
    )
    parser.add_argument(
        "--experiment-roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_ROOTS),
    )
    return parser.parse_args()


def test_unit_semantics() -> None:
    contract = {
        "contract": {
            "contract_id": "contract-unit-test",
            "permitted_tools": [
                {
                    "tool": "create_calendar_event",
                    "effect": "write",
                    "constraints": [
                        {
                            "argument": "title",
                            "operator": "equals",
                            "value": "Authorized meeting",
                        }
                    ],
                    "allow_additional_arguments": False,
                    "approval_required": False,
                }
            ],
            "prohibited_tools": ["delete_file"],
            "prohibited_effects": ["delete", "communicate"],
        }
    }

    allowed = label_action(
        contract,
        tool="create_calendar_event",
        arguments={"title": "Authorized meeting", "location": None},
        step=1,
    )
    assert allowed.permitted
    assert allowed.rule_id == "CONTRACT_ALLOW"

    wrong_argument = label_action(
        contract,
        tool="create_calendar_event",
        arguments={"title": "Injected meeting"},
        step=1,
    )
    assert not wrong_argument.permitted
    assert wrong_argument.rule_id == "ARGUMENT_CONSTRAINT_VIOLATION"

    operational_extra = label_action(
        contract,
        tool="create_calendar_event",
        arguments={
            "title": "Authorized meeting",
            "location": "Unauthorized room",
        },
        step=1,
    )
    assert not operational_extra.permitted

    prohibited = label_action(
        contract,
        tool="delete_file",
        arguments={"file_id": "13"},
        step=1,
    )
    assert not prohibited.permitted
    assert prohibited.rule_id == "CONTRACT_PROHIBITED_TOOL"

    outside = label_action(
        contract,
        tool="list_files",
        arguments={},
        step=1,
    )
    assert not outside.permitted
    assert outside.rule_id == "TOOL_OUTSIDE_CONTRACT"


def assert_prefix_has_no_leakage(
    prefix: dict[str, Any],
    *,
    through_step: int,
) -> None:
    node_ids = {node["node_id"] for node in prefix["nodes"]}

    for node in prefix["nodes"]:
        assert node["node_type"] not in LEAKAGE_NODE_TYPES
        attributes = node.get("attributes", {})
        assert "decision" not in attributes
        assert "reason" not in attributes
        assert "rule_id" not in attributes

        if node["node_type"] == "proposed_action":
            assert "status" not in attributes
            assert int(attributes["step"]) <= through_step

        if node["node_type"] == "execution_outcome":
            action_id = node["node_id"].replace("outcome:", "action:", 1)
            action = next(
                item for item in prefix["nodes"] if item["node_id"] == action_id
            )
            assert int(action["attributes"]["step"]) < through_step

    for edge in prefix["edges"]:
        assert edge["relation"] not in LEAKAGE_RELATIONS
        assert int(edge["step"]) <= through_step
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids

        if edge["relation"] == "COMPLETED_AS":
            assert int(edge["step"]) < through_step


def audit_experiment(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    runs_root = root / "runs"
    run_directories = sorted(path for path in runs_root.iterdir() if path.is_dir())
    counts: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []

    for run_directory in run_directories:
        contract_path = run_directory / "contract.json"
        trace_path = run_directory / "monitor_trace.jsonl"
        graph_path = run_directory / "execution_graph.json"

        if not (contract_path.exists() and trace_path.exists() and graph_path.exists()):
            counts["runs_without_sentrygraph_bundle"] += 1
            continue

        counts["audited_runs"] += 1
        contract = load_contract(contract_path)
        events = load_jsonl(trace_path)
        graph = load_graph(graph_path)
        proposals = proposed_actions(events)
        labels = label_prefixes(contract, proposals)
        recorded = recorded_proposal_decisions(events)

        assert len(proposals) == len(labels)

        for proposal, prefix_label in zip(proposals, labels, strict=True):
            action = prefix_label.action
            key = (int(proposal["step"]), str(proposal["tool"]))
            recorded_decision = recorded.get(key)
            expected_decision = "allow" if action.permitted else "deny"

            counts["proposals"] += 1
            counts[f"derived_{expected_decision}"] += 1
            rules[action.rule_id] += 1

            if recorded_decision != expected_decision:
                mismatches.append(
                    {
                        "run": run_directory.name,
                        "step": proposal["step"],
                        "tool": proposal["tool"],
                        "derived": expected_decision,
                        "recorded": recorded_decision,
                        "derived_rule": action.rule_id,
                    }
                )

            prefix = sanitize_execution_prefix(
                graph,
                through_step=int(proposal["step"]),
            )
            assert_prefix_has_no_leakage(
                prefix,
                through_step=int(proposal["step"]),
            )

            current_actions = [
                node
                for node in prefix["nodes"]
                if node["node_type"] == "proposed_action"
                and int(node["attributes"]["step"]) == int(proposal["step"])
                and node["attributes"]["tool"] == proposal["tool"]
            ]
            assert current_actions, (
                f"Current proposal absent from prefix: {run_directory.name} "
                f"step={proposal['step']} tool={proposal['tool']}"
            )

    return {
        "root": str(root),
        "counts": dict(sorted(counts.items())),
        "derived_rules": dict(sorted(rules.items())),
        "mismatches": mismatches,
    }


def main() -> int:
    args = parse_args()
    test_unit_semantics()

    reports = [audit_experiment(root) for root in args.experiment_roots]
    total_mismatches = sum(len(report["mismatches"]) for report in reports)

    print("=== PHASE 1B PREFIX LABELER AUDIT ===")
    print("Unit semantics: PASS")

    for report in reports:
        print()
        print("Experiment:", report["root"])
        print("Counts:", json.dumps(report["counts"], sort_keys=True))
        print(
            "Derived rules:",
            json.dumps(report["derived_rules"], sort_keys=True),
        )
        print("Mismatches:", len(report["mismatches"]))

        if report["mismatches"]:
            print(json.dumps(report["mismatches"][:10], indent=2))

    print()
    print("Total mismatches:", total_mismatches)

    if total_mismatches:
        print("FAIL: Independent labels disagree with recorded evidence.")
        return 1

    print("PASS: Independent labels match all recorded proposals.")
    print("PASS: Every graph prefix is temporally and policy leakage-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
