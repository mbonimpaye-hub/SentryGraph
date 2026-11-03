#!/usr/bin/env python3
"""Freeze an expanded Phase 3 plan bound to a deployable policy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from plan_model_driven_multi_agent_experiments import (
    build_plan,
    canonical_hash,
    csv_values,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_identity(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise RuntimeError(f"Empty policy bundle: {root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(root),
        "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-policy-bundle", type=Path, required=True)
    parser.add_argument(
        "--models", type=csv_values, default=["gpt-4o-2024-08-06"]
    )
    parser.add_argument("--tasks", type=csv_values, default=["user_task_35"])
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
        "--topologies", type=csv_values, default=["orchestrator_worker"]
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plans"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    policy = bundle_identity(args.hybrid_policy_bundle)
    plan["identity"]["hybrid_policy_bundle"] = policy
    plan["identity"]["experiment_status"] = "expanded_model_driven_evaluation"
    plan["plan_id"] = canonical_hash(plan["identity"])[:16]
    plan["schema_version"] = "0.2"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (
        f"expanded_model_driven_multi_agent_plan_{plan['plan_id']}.json"
    )
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benign = [cell for cell in plan["cells"] if cell["phase"] == "benign"]
    attacks = [cell for cell in plan["cells"] if cell["phase"] == "attack"]
    print("=== PHASE 3I EXPANDED MODEL-DRIVEN PLAN ===")
    print("Plan ID:", plan["plan_id"])
    print("Policy fingerprint:", policy["fingerprint"])
    print("Models:", plan["identity"]["models"])
    print("Tasks:", plan["identity"]["tasks"])
    print("Attacks:", plan["identity"]["attacks"])
    print("Injection tasks:", plan["identity"]["injection_tasks"])
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
