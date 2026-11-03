"""Create a frozen matched plan for multi-agent security experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sentrygraph.multi_agent_scenarios import catalog_manifest


def canonical_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze matched baseline/SentryGraph multi-agent cells."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plans"),
    )
    return parser.parse_args()


def build_plan(trials: int) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("Trials must be at least one.")

    catalog = catalog_manifest()
    cells: list[dict[str, Any]] = []
    for scenario in catalog["scenarios"]:
        for trial in range(1, trials + 1):
            for condition in ("baseline", "sentrygraph"):
                cell = {
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario["category"],
                    "family": scenario["family"],
                    "trial": trial,
                    "condition": condition,
                    "scenario_fingerprint": catalog[
                        "scenario_fingerprints"
                    ][scenario["scenario_id"]],
                }
                cell["cell_id"] = canonical_hash(cell)[:20]
                cells.append(cell)

    identity = {
        "schema_version": "0.1",
        "protocol": "matched_multi_agent_security",
        "catalog": catalog["catalog"],
        "scenario_fingerprints": catalog["scenario_fingerprints"],
        "trials": trials,
        "conditions": ["baseline", "sentrygraph"],
    }
    return {
        "schema_version": "0.1",
        "plan_id": canonical_hash(identity)[:16],
        "identity": identity,
        "catalog": catalog,
        "cells": cells,
    }


def main() -> int:
    args = parse_args()
    plan = build_plan(args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"multi_agent_plan_{plan['plan_id']}.json"
    path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    scenarios = plan["catalog"]["scenarios"]
    attacks = [s for s in scenarios if s["category"] == "attack"]
    benign = [s for s in scenarios if s["category"] == "benign"]
    print("=== PHASE 2E MULTI-AGENT EXPERIMENT PLAN ===")
    print("Plan ID:", plan["plan_id"])
    print("Scenarios:", len(scenarios))
    print("Benign scenarios:", len(benign))
    print("Attack scenarios:", len(attacks))
    print("Families:", len({s["family"] for s in scenarios}))
    print("Trials:", args.trials)
    print("Matched cells:", len(plan["cells"]))
    print("Plan:", path)
    print("NO API CALLS WERE MADE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
