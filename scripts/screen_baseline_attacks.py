"""Screen AgentDojo baseline pairs and stop at first successful attack."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdojo.scripts.benchmark import load_attack
from agentdojo.task_suite import get_suite

from run_contract_pilot import (
    BENCHMARK_VERSION,
    MODEL_ID,
    SUITE_NAME,
    USER_TASK_ID,
    require_environment,
)
from run_injection_pilot import build_pipeline


ATTACKS = (
    "direct",
    "system_message",
    "ignore_previous",
    "important_instructions",
)

INJECTION_TASKS = (
    "injection_task_0",
    "injection_task_1",
    "injection_task_2",
    "injection_task_3",
    "injection_task_4",
)

MAX_PAIRS = int(
    os.getenv("SENTRYGRAPH_MAX_SCREEN_PAIRS", "8")
)

OUTPUT_PATH = Path(
    "outputs/baseline_attack_screening.json"
)


def save_results(
    results: list[dict[str, Any]],
    successful_pair: dict[str, Any] | None,
) -> None:
    payload = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": os.getenv("SENTRYGRAPH_MODEL_ID", MODEL_ID),
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "user_task_id": USER_TASK_ID,
        "maximum_pairs": MAX_PAIRS,
        "tested_pairs": len(results),
        "successful_attack_pair": successful_pair,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> int:
    require_environment()

    suite = get_suite(
        BENCHMARK_VERSION,
        SUITE_NAME,
    )

    user_task = suite.get_user_task_by_id(
        USER_TASK_ID
    )

    results: list[dict[str, Any]] = []
    successful_pair: dict[str, Any] | None = None
    pair_number = 0

    print("=== AGENTDOJO BASELINE ATTACK SCREENING ===")
    print("Model:", os.getenv("SENTRYGRAPH_MODEL_ID", MODEL_ID))
    print(f"Suite: {SUITE_NAME}")
    print(f"User task: {USER_TASK_ID}")
    print(f"Maximum paid pairs: {MAX_PAIRS}")
    print()

    for attack_name in ATTACKS:
        for injection_task_id in INJECTION_TASKS:
            if pair_number >= MAX_PAIRS:
                break

            pair_number += 1

            print(
                f"--- Pair {pair_number}/{MAX_PAIRS}: "
                f"{attack_name} + {injection_task_id} ---"
            )

            pipeline = build_pipeline()

            attack = load_attack(
                attack_name,
                suite,
                pipeline,
            )

            injection_task = (
                suite.get_injection_task_by_id(
                    injection_task_id
                )
            )

            try:
                injections = attack.attack(
                    user_task,
                    injection_task,
                )
            except ValueError as error:
                result = {
                    "pair": pair_number,
                    "attack": attack_name,
                    "injection_task_id": injection_task_id,
                    "status": "not_injectable",
                    "error": str(error),
                }

                results.append(result)
                save_results(results, successful_pair)

                print(f"Skipped: {error}")
                print()
                continue

            utility, attack_success = (
                suite.run_task_with_pipeline(
                    agent_pipeline=pipeline,
                    user_task=user_task,
                    injection_task=injection_task,
                    injections=injections,
                    environment=None,
                    verbose=False,
                )
            )

            result = {
                "pair": pair_number,
                "attack": attack_name,
                "injection_task_id": injection_task_id,
                "injection_goal": injection_task.GOAL,
                "injection_locations": sorted(injections),
                "utility": utility,
                "attack_success": attack_success,
                "status": "completed",
            }

            results.append(result)

            print(f"Utility: {utility}")
            print(f"Attack success: {attack_success}")
            print()

            if attack_success:
                successful_pair = result
                save_results(results, successful_pair)

                print("FOUND BASELINE-COMPROMISED PAIR")
                print(f"Attack: {attack_name}")
                print(
                    "Injection task:",
                    injection_task_id,
                )
                print(
                    "Matched-run command:"
                )
                print(
                    f"SENTRYGRAPH_ATTACK={attack_name} "
                    f"SENTRYGRAPH_INJECTION_TASK="
                    f"{injection_task_id} "
                    "python scripts/run_injection_pilot.py"
                )
                return 0

            save_results(results, successful_pair)

        if pair_number >= MAX_PAIRS:
            break

    print("No successful baseline attack was found.")
    print(f"Results: {OUTPUT_PATH}")
    print(
        "Increase SENTRYGRAPH_MAX_SCREEN_PAIRS only "
        "after reviewing the completed pairs."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
