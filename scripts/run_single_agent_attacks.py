"""Execute qualified matched AgentDojo attack experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from agentdojo.scripts.benchmark import load_attack
from agentdojo.task_suite import get_suite

from run_single_agent_experiments import (
    FIXED_GRAPH_PATH,
    FIXED_TRACE_PATH,
    append_jsonl,
    archive_preexisting_fixed_outputs,
    build_pipeline,
    canonical_hash,
    denied_action_ids,
    load_events,
    load_json,
    make_runtime_class,
    move_if_present,
    percentile_nearest_rank,
    require_environment,
    utc_now,
)
from sentrygraph.agentdojo_contract_factory import (
    compile_workspace_contract,
)


def injection_hash(
    injections: dict[str, str],
) -> str:
    canonical = json.dumps(
        injections,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def pair_definition(
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Return fields shared by matched conditions."""

    return {
        "phase": cell["phase"],
        "model": cell["model"],
        "task_id": cell["task_id"],
        "attack": cell["attack"],
        "injection_task_id": cell[
            "injection_task_id"
        ],
        "trial": cell["trial"],
    }


def load_or_create_injections(
    *,
    pair_path: Path,
    suite: Any,
    task: Any,
    injection_task: Any,
    attack_name: str,
    model_id: str,
) -> tuple[dict[str, str], str]:
    """Generate an injection once and persist it for both conditions."""

    if pair_path.exists():
        payload = load_json(pair_path)

        return (
            payload["injections"],
            payload["injection_fingerprint"],
        )

    pipeline = build_pipeline(model_id)

    attack = load_attack(
        attack_name,
        suite,
        pipeline,
    )

    injections = attack.attack(
        task,
        injection_task,
    )

    fingerprint = injection_hash(injections)

    pair_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "created_at": utc_now(),
                "model": model_id,
                "task_id": task.ID,
                "attack": attack_name,
                "injection_task_id": injection_task.ID,
                "injection_goal": injection_task.GOAL,
                "injection_locations": sorted(
                    injections
                ),
                "injection_fingerprint": fingerprint,
                "injections": injections,
            },
            indent=2,
        )
        + "\n"
    )

    return injections, fingerprint


def qualified_keys(
    qualification: dict[str, Any],
) -> set[tuple[str, str, int]]:
    return {
        (
            entry["model"],
            entry["task_id"],
            entry["trial"],
        )
        for entry in qualification["qualified"]
        if entry.get("eligible") is True
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute qualified matched baseline and "
            "SentryGraph attack trajectories."
        )
    )

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-executions",
        type=int,
        default=0,
        help=(
            "Maximum new paid trajectories. Zero performs "
            "inspection only."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.plan.exists():
        raise FileNotFoundError(
            f"Plan not found: {args.plan}"
        )

    plan = load_json(args.plan)
    plan_id = plan["plan_id"]

    experiment_root = (
        args.output_root
        / plan_id
    )

    qualification_path = (
        experiment_root
        / "qualification.json"
    )

    if not qualification_path.exists():
        raise FileNotFoundError(
            "Qualification manifest not found: "
            f"{qualification_path}"
        )

    qualification = load_json(
        qualification_path
    )

    eligible_keys = qualified_keys(
        qualification
    )

    attack_cells = [
        cell
        for cell in plan["cells"]
        if cell["phase"] == "attack"
        and (
            cell["model"],
            cell["task_id"],
            cell["trial"],
        )
        in eligible_keys
    ]

    runs_root = experiment_root / "runs"
    pairs_root = experiment_root / "injections"
    aggregate_path = (
        experiment_root / "results.jsonl"
    )

    pending = []

    for cell in attack_cells:
        cell_id = canonical_hash(cell)[:20]
        run_dir = runs_root / cell_id
        result_path = run_dir / "result.json"

        if not result_path.exists():
            pending.append(
                (
                    cell_id,
                    cell,
                    run_dir,
                    result_path,
                )
            )

    print("=== QUALIFIED ATTACK EXECUTOR ===")
    print(f"Plan ID: {plan_id}")
    print(
        "Qualified task/model/trial groups:",
        len(eligible_keys),
    )
    print(
        "Eligible attack cells:",
        len(attack_cells),
    )
    print(
        "Completed attack cells:",
        len(attack_cells) - len(pending),
    )
    print(f"Pending attack cells: {len(pending)}")
    print(
        "Maximum new executions:",
        args.max_executions,
    )
    print(f"Execute enabled: {args.execute}")
    print(f"Output: {experiment_root}")

    if not args.execute:
        print()
        print(
            "DRY RUN: no API calls were made. "
            "Use --execute with a positive "
            "--max-executions to run cells."
        )
        return 0

    if args.max_executions < 1:
        raise ValueError(
            "--execute requires --max-executions "
            "of at least one."
        )

    require_environment()

    suite = get_suite(
        plan["configuration"]["benchmark_version"],
        plan["configuration"]["suite"],
    )

    base_environment = (
        suite.load_and_inject_default_environment({})
    )

    experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_preexisting_fixed_outputs(
        experiment_root
    )

    executed = 0
    failed = 0

    for cell_id, cell, run_dir, result_path in pending:
        if executed >= args.max_executions:
            break

        model_id = cell["model"]
        task_id = cell["task_id"]
        attack_name = cell["attack"]
        injection_task_id = cell[
            "injection_task_id"
        ]
        condition = cell["condition"]

        task = suite.get_user_task_by_id(
            task_id
        )

        injection_task = (
            suite.get_injection_task_by_id(
                injection_task_id
            )
        )

        contract_result = (
            compile_workspace_contract(
                task=task,
                base_environment=base_environment,
            )
        )

        contract = contract_result.contract

        planned_contract = plan["contracts"][task_id]

        planned_fingerprint = planned_contract[
            "contract_fingerprint"
        ]

        runtime_fingerprint = (
            contract.fingerprint()
        )

        if runtime_fingerprint != planned_fingerprint:
            raise RuntimeError(
                "Runtime contract does not match the "
                f"frozen plan for {task_id}: "
                f"planned={planned_fingerprint}, "
                f"runtime={runtime_fingerprint}"
            )

        pair = pair_definition(cell)
        pair_id = canonical_hash(pair)[:20]

        pair_path = (
            pairs_root
            / f"{pair_id}.json"
        )

        injections, fingerprint = (
            load_or_create_injections(
                pair_path=pair_path,
                suite=suite,
                task=task,
                injection_task=injection_task,
                attack_name=attack_name,
                model_id=model_id,
            )
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (run_dir / "contract.json").write_text(
            json.dumps(
                contract_result.to_dict(),
                indent=2,
            )
            + "\n"
        )

        pipeline = build_pipeline(model_id)

        record: dict[str, Any] = {
            "schema_version": "0.1",
            "plan_id": plan_id,
            "cell_id": cell_id,
            "pair_id": pair_id,
            "started_at": utc_now(),
            "cell": cell,
            "model": model_id,
            "suite": suite.name,
            "task_id": task_id,
            "attack": attack_name,
            "injection_task_id": injection_task_id,
            "injection_goal": injection_task.GOAL,
            "injection_fingerprint": fingerprint,
            "condition": condition,
            "contract_id": contract.contract_id,
            "contract_fingerprint": (
                runtime_fingerprint
            ),
            "authorized_side_effect_tools": (
                contract_result
                .to_dict()
                ["authorized_side_effect_tools"]
            ),
            "status": "running",
        }

        print()
        print(
            f"RUN {executed + 1}/"
            f"{args.max_executions}: "
            f"{cell_id} "
            f"{model_id} "
            f"{task_id} "
            f"{attack_name} "
            f"{injection_task_id} "
            f"{condition}"
        )

        start = time.perf_counter()

        try:
            if condition == "baseline":
                utility, attack_success = (
                    suite.run_task_with_pipeline(
                        agent_pipeline=pipeline,
                        user_task=task,
                        injection_task=injection_task,
                        injections=injections,
                        environment=None,
                        verbose=False,
                    )
                )

            elif condition == "sentrygraph":
                runtime_class = make_runtime_class(
                    contract
                )

                utility, attack_success = (
                    suite.run_task_with_pipeline(
                        agent_pipeline=pipeline,
                        user_task=task,
                        injection_task=injection_task,
                        injections=injections,
                        runtime_class=runtime_class,
                        environment=None,
                        verbose=False,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported condition: {condition}"
                )

            duration = time.perf_counter() - start

            events: list[dict[str, Any]] = []
            graph: dict[str, Any] = {}

            if condition == "sentrygraph":
                move_if_present(
                    FIXED_TRACE_PATH,
                    run_dir / "monitor_trace.jsonl",
                )

                move_if_present(
                    FIXED_GRAPH_PATH,
                    run_dir / "execution_graph.json",
                )

                events = load_events(
                    run_dir / "monitor_trace.jsonl"
                )

                graph_path = (
                    run_dir / "execution_graph.json"
                )

                if graph_path.exists():
                    graph = load_json(graph_path)

            proposed = [
                event
                for event in events
                if event.get("phase") == "proposed"
            ]

            completed = [
                event
                for event in events
                if event.get("phase") == "completed"
            ]

            denied_ids = denied_action_ids(events)

            attempted_tools = [
                event.get("tool")
                for event in proposed
            ]

            completed_tools = [
                event.get("tool")
                for event in completed
            ]

            denied_tools = sorted(
                {
                    event.get("tool")
                    for event in events
                    if event.get("event_id")
                    in denied_ids
                    and event.get("tool")
                }
            )

            latencies = [
                float(event["latency_ms"])
                for event in completed
                if event.get("latency_ms") is not None
            ]

            relations = sorted(
                {
                    edge.get("relation")
                    for edge in graph.get("edges", [])
                    if edge.get("relation")
                }
            )

            record.update(
                {
                    "status": "completed",
                    "finished_at": utc_now(),
                    "duration_seconds": duration,
                    "utility": utility,
                    "attack_success": attack_success,
                    "completed_violation": (
                        attack_success
                    ),
                    "proposed_actions": len(proposed),
                    "completed_actions": len(completed),
                    "denied_actions": len(denied_ids),
                    "attempted_tools": attempted_tools,
                    "completed_tools": completed_tools,
                    "denied_tools": denied_tools,
                    "balanced_lifecycle": (
                        len(proposed)
                        == len(completed)
                        + len(denied_ids)
                    )
                    if condition == "sentrygraph"
                    else None,
                    "tool_latency_p50_ms": (
                        percentile_nearest_rank(
                            latencies,
                            0.50,
                        )
                    ),
                    "tool_latency_p95_ms": (
                        percentile_nearest_rank(
                            latencies,
                            0.95,
                        )
                    ),
                    "graph_nodes": len(
                        graph.get("nodes", [])
                    ),
                    "graph_edges": len(
                        graph.get("edges", [])
                    ),
                    "graph_relations": relations,
                    "contains_governs": (
                        "GOVERNS" in relations
                    ),
                    "contains_blocks": (
                        "BLOCKS" in relations
                    ),
                }
            )

            print(f"  Utility: {utility}")
            print(
                "  Attack success:",
                attack_success,
            )
            print(
                "  Denied actions:",
                len(denied_ids),
            )
            print(
                "  Denied tools:",
                denied_tools,
            )
            print(
                "  Duration:",
                f"{duration:.3f} s",
            )

        except Exception as error:
            duration = time.perf_counter() - start
            failed += 1

            record.update(
                {
                    "status": "error",
                    "finished_at": utc_now(),
                    "duration_seconds": duration,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

            move_if_present(
                FIXED_TRACE_PATH,
                run_dir / "monitor_trace.partial.jsonl",
            )

            move_if_present(
                FIXED_GRAPH_PATH,
                run_dir / "execution_graph.partial.json",
            )

            print(
                f"  ERROR: {type(error).__name__}: "
                f"{error}",
                file=sys.stderr,
            )

        result_path.write_text(
            json.dumps(record, indent=2) + "\n"
        )

        append_jsonl(
            aggregate_path,
            record,
        )

        executed += 1

    print()
    print("=== ATTACK EXECUTION SUMMARY ===")
    print(f"New executions: {executed}")
    print(f"Errors: {failed}")
    print(
        "Remaining pending:",
        max(0, len(pending) - executed),
    )
    print(f"Results: {aggregate_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
