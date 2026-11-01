"""Plan reproducible single-agent SentryGraph experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agentdojo.task_suite import get_suite

from sentrygraph.agentdojo_contract_factory import (
    UnsupportedTaskContract,
    compile_workspace_read_only_contract,
)


DEFAULT_MODELS = (
    "gpt-5.4-mini-2026-03-17",
    "gpt-4o-2024-08-06",
)

DEFAULT_ATTACKS = (
    "direct",
    "system_message",
    "ignore_previous",
    "important_instructions",
)

DEFAULT_CONDITIONS = (
    "baseline",
    "sentrygraph",
)

BENCHMARK_VERSION = "v1.2"
SUITE_NAME = "workspace"


def comma_values(value: str) -> tuple[str, ...]:
    """Parse a comma-separated command-line argument."""

    values = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one value is required."
        )

    return values


def ordered_unique(
    values: Iterable[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def discover_supported_tasks(
    suite: Any,
) -> tuple[
    dict[str, Any],
    dict[str, str],
]:
    environment = (
        suite.load_and_inject_default_environment({})
    )

    supported = {}
    rejected = {}

    for task_id, task in suite.user_tasks.items():
        try:
            result = compile_workspace_read_only_contract(
                task=task,
                base_environment=environment,
            )

            supported[task_id] = result

        except UnsupportedTaskContract as error:
            rejected[task_id] = str(error)

    return supported, rejected


def validate_subset(
    requested: tuple[str, ...],
    available: Iterable[str],
    label: str,
) -> tuple[str, ...]:
    available_set = set(available)

    unknown = [
        item
        for item in requested
        if item not in available_set
    ]

    if unknown:
        raise ValueError(
            f"Unknown {label}: {unknown}. "
            f"Available values: {sorted(available_set)}"
        )

    return ordered_unique(requested)


def build_cells(
    models: tuple[str, ...],
    task_ids: tuple[str, ...],
    attacks: tuple[str, ...],
    injection_task_ids: tuple[str, ...],
    conditions: tuple[str, ...],
    trials: int,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    for model in models:
        for task_id in task_ids:
            for trial in range(1, trials + 1):
                for condition in conditions:
                    cells.append(
                        {
                            "phase": "benign",
                            "model": model,
                            "task_id": task_id,
                            "attack": None,
                            "injection_task_id": None,
                            "condition": condition,
                            "trial": trial,
                        }
                    )

                for attack in attacks:
                    for injection_task_id in injection_task_ids:
                        for condition in conditions:
                            cells.append(
                                {
                                    "phase": "attack",
                                    "model": model,
                                    "task_id": task_id,
                                    "attack": attack,
                                    "injection_task_id": (
                                        injection_task_id
                                    ),
                                    "condition": condition,
                                    "trial": trial,
                                }
                            )

    return cells


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a no-cost execution plan for the "
            "single-agent SentryGraph evaluation."
        )
    )

    parser.add_argument(
        "--models",
        type=comma_values,
        default=DEFAULT_MODELS,
        help="Comma-separated API model identifiers.",
    )

    parser.add_argument(
        "--tasks",
        type=comma_values,
        default=None,
        help=(
            "Comma-separated read-only user-task IDs. "
            "The default is all supported read-only tasks."
        ),
    )

    parser.add_argument(
        "--attacks",
        type=comma_values,
        default=DEFAULT_ATTACKS,
        help="Comma-separated AgentDojo attack names.",
    )

    parser.add_argument(
        "--injection-tasks",
        type=comma_values,
        default=None,
        help=(
            "Comma-separated injection-task IDs. "
            "The default is every injection task in the suite."
        ),
    )

    parser.add_argument(
        "--conditions",
        type=comma_values,
        default=DEFAULT_CONDITIONS,
        help="Comma-separated experimental conditions.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Repeated trials per experimental cell.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plans"),
        help="Directory for experiment plans.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.trials < 1:
        raise ValueError(
            "--trials must be at least one."
        )

    suite = get_suite(
        BENCHMARK_VERSION,
        SUITE_NAME,
    )

    supported, rejected = discover_supported_tasks(
        suite
    )

    if args.tasks is None:
        task_ids = tuple(sorted(supported))
    else:
        task_ids = validate_subset(
            args.tasks,
            supported,
            "read-only task IDs",
        )

    if args.injection_tasks is None:
        injection_task_ids = tuple(
            sorted(suite.injection_tasks)
        )
    else:
        injection_task_ids = validate_subset(
            args.injection_tasks,
            suite.injection_tasks,
            "injection-task IDs",
        )

    models = ordered_unique(args.models)
    attacks = ordered_unique(args.attacks)

    conditions = validate_subset(
        args.conditions,
        DEFAULT_CONDITIONS,
        "conditions",
    )

    cells = build_cells(
        models=models,
        task_ids=task_ids,
        attacks=attacks,
        injection_task_ids=injection_task_ids,
        conditions=conditions,
        trials=args.trials,
    )

    benign_cells = [
        cell
        for cell in cells
        if cell["phase"] == "benign"
    ]

    attack_cells = [
        cell
        for cell in cells
        if cell["phase"] == "attack"
    ]

    contract_metadata = {
        task_id: {
            "contract_id": (
                supported[task_id].contract.contract_id
            ),
            "contract_fingerprint": (
                supported[task_id]
                .contract
                .fingerprint()
            ),
            "ground_truth_tools": list(
                supported[task_id].ground_truth_tools
            ),
            "authorized_families": list(
                supported[task_id].authorized_families
            ),
            "permitted_tools": list(
                supported[task_id].permitted_tools
            ),
        }
        for task_id in task_ids
    }

    configuration = {
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "models": list(models),
        "task_ids": list(task_ids),
        "attacks": list(attacks),
        "injection_task_ids": list(
            injection_task_ids
        ),
        "conditions": list(conditions),
        "trials": args.trials,
    }

    plan_id = canonical_hash(configuration)[:16]

    plan = {
        "schema_version": "0.1",
        "plan_id": plan_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": configuration,
        "counts": {
            "supported_read_only_tasks": len(
                supported
            ),
            "rejected_state_changing_tasks": len(
                rejected
            ),
            "selected_tasks": len(task_ids),
            "selected_models": len(models),
            "selected_attacks": len(attacks),
            "selected_injection_tasks": len(
                injection_task_ids
            ),
            "trials": args.trials,
            "benign_trajectories": len(
                benign_cells
            ),
            "attack_trajectories": len(
                attack_cells
            ),
            "total_trajectories": len(cells),
        },
        "contracts": contract_metadata,
        "rejected_tasks": rejected,
        "cells": cells,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_dir
        / f"single_agent_plan_{plan_id}.json"
    )

    output_path.write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    print("=== SINGLE-AGENT EXPERIMENT PLAN ===")
    print(f"Plan ID: {plan_id}")
    print(f"Suite: {SUITE_NAME}")
    print(f"Models: {list(models)}")
    print(f"Selected tasks: {len(task_ids)}")
    print(f"Attacks: {list(attacks)}")
    print(
        "Injection tasks:",
        len(injection_task_ids),
    )
    print(f"Trials: {args.trials}")
    print(
        "Benign trajectories:",
        len(benign_cells),
    )
    print(
        "Attack trajectories:",
        len(attack_cells),
    )
    print(
        "Total trajectories:",
        len(cells),
    )
    print(f"Plan: {output_path}")
    print()
    print(
        "NO API CALLS WERE MADE. "
        "This command only created an execution plan."
    )


if __name__ == "__main__":
    main()
