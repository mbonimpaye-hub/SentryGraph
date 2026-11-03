"""Freeze matched LLM-driven multi-agent experiment cells without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sentrygraph.model_driven_multi_agent_scenarios import protocol_manifest


DEFAULT_IMPLEMENTATION_FILES = (
    "sentrygraph/agentdojo_contract_factory.py",
    "sentrygraph/contract_compiler.py",
    "sentrygraph/multi_agent_contracts.py",
    "sentrygraph/multi_agent_execution_graph.py",
    "sentrygraph/multi_agent_runtime.py",
    "sentrygraph/multi_agent_agentdojo_adapter.py",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def csv_values(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("At least one value is required.")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Values must be unique.")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprints(root: Path) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    missing: list[str] = []
    for relative in DEFAULT_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        else:
            fingerprints[relative] = file_sha256(path)
    if missing:
        raise FileNotFoundError(
            "Missing implementation files: " + ", ".join(missing)
        )
    return fingerprints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a matched model-driven multi-agent protocol."
    )
    parser.add_argument(
        "--models",
        type=csv_values,
        default=["gpt-4o-2024-08-06"],
    )
    parser.add_argument(
        "--tasks",
        type=csv_values,
        default=["user_task_12", "user_task_33", "user_task_35"],
    )
    parser.add_argument(
        "--attacks",
        type=csv_values,
        default=["important_instructions", "tool_knowledge"],
    )
    parser.add_argument(
        "--injection-tasks",
        type=csv_values,
        default=["injection_task_0", "injection_task_1", "injection_task_2"],
    )
    parser.add_argument(
        "--topologies",
        type=csv_values,
        default=["orchestrator_worker"],
    )
    parser.add_argument(
        "--handoff-channels",
        type=csv_values,
        default=["retrieved_content", "delegation_message"],
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--benchmark-version", default="v1.2")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/plans")
    )
    return parser.parse_args()


def validate_selection(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if args.trials < 1:
        raise ValueError("Trials must be at least one.")
    known_topologies = {
        item["topology_id"] for item in protocol["topologies"]
    }
    known_channels = {
        item["channel_id"] for item in protocol["handoff_channels"]
    }
    unknown_topologies = set(args.topologies) - known_topologies
    unknown_channels = set(args.handoff_channels) - known_channels
    if unknown_topologies:
        raise ValueError(f"Unknown topologies: {sorted(unknown_topologies)}")
    if unknown_channels:
        raise ValueError(f"Unknown handoff channels: {sorted(unknown_channels)}")


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    protocol = protocol_manifest()
    validate_selection(args, protocol)
    source_hashes = implementation_fingerprints(args.repository_root)

    common = {
        "suite": args.suite,
        "benchmark_version": args.benchmark_version,
        "models": args.models,
        "tasks": args.tasks,
        "attacks": args.attacks,
        "injection_tasks": args.injection_tasks,
        "topologies": args.topologies,
        "handoff_channels": args.handoff_channels,
        "trials": args.trials,
        "conditions": [
            "baseline",
            "sentrygraph_deterministic",
            "sentrygraph_hybrid",
        ],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "implementation_fingerprints": source_hashes,
    }
    cells: list[dict[str, Any]] = []

    for model in args.models:
        for task_id in args.tasks:
            for topology in args.topologies:
                for trial in range(1, args.trials + 1):
                    for condition in common["conditions"]:
                        cell = {
                            "phase": "benign",
                            "model": model,
                            "task_id": task_id,
                            "topology": topology,
                            "trial": trial,
                            "condition": condition,
                        }
                        cell["cell_id"] = canonical_hash(cell)[:20]
                        cells.append(cell)

            for attack in args.attacks:
                for injection_task_id in args.injection_tasks:
                    for topology in args.topologies:
                        for channel in args.handoff_channels:
                            for trial in range(1, args.trials + 1):
                                for condition in common["conditions"]:
                                    cell = {
                                        "phase": "attack",
                                        "model": model,
                                        "task_id": task_id,
                                        "attack": attack,
                                        "injection_task_id": injection_task_id,
                                        "topology": topology,
                                        "handoff_channel": channel,
                                        "trial": trial,
                                        "condition": condition,
                                    }
                                    cell["cell_id"] = canonical_hash(cell)[:20]
                                    cells.append(cell)

    identity = {"schema_version": "0.1", **common}
    return {
        "schema_version": "0.1",
        "plan_id": canonical_hash(identity)[:16],
        "identity": identity,
        "protocol": protocol,
        "cells": cells,
    }


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"model_driven_multi_agent_plan_{plan['plan_id']}.json"
    )
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benign = [cell for cell in plan["cells"] if cell["phase"] == "benign"]
    attacks = [cell for cell in plan["cells"] if cell["phase"] == "attack"]
    print("=== PHASE 3A MODEL-DRIVEN MULTI-AGENT PLAN ===")
    print("Plan ID:", plan["plan_id"])
    print("Models:", plan["identity"]["models"])
    print("Tasks:", plan["identity"]["tasks"])
    print("Topologies:", plan["identity"]["topologies"])
    print("Handoff channels:", plan["identity"]["handoff_channels"])
    print("Trials:", plan["identity"]["trials"])
    print("Benign cells:", len(benign))
    print("Attack cells:", len(attacks))
    print("Total cells:", len(plan["cells"]))
    print("Plan:", path)
    print("NO API CALLS WERE MADE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
