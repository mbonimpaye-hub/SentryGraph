"""Audit AgentDojo workspace tasks that legitimately require side effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdojo.task_suite import get_suite


BENCHMARK_VERSION = "v1.2"
SUITE_NAME = "workspace"

WRITE_TOOLS = {
    "send_email",
    "delete_email",
    "create_calendar_event",
    "cancel_calendar_event",
    "reschedule_calendar_event",
    "add_calendar_event_participants",
    "append_to_file",
    "create_file",
    "delete_file",
    "share_file",
}

OUTPUT_PATH = Path(
    "outputs/audits/workspace_legitimate_write_tasks.json"
)


def serialize_value(value: Any) -> Any:
    """Convert AgentDojo and Pydantic values to JSON-safe objects."""

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            serialize_value(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return serialize_value(
            value.model_dump(mode="json")
        )

    if hasattr(value, "__dict__"):
        return serialize_value(vars(value))

    return str(value)


def call_to_dict(call: Any) -> dict[str, Any]:
    """Normalize one AgentDojo ground-truth function call."""

    function = getattr(call, "function", None)
    arguments = getattr(call, "args", None)

    if arguments is None:
        arguments = getattr(call, "arguments", None)

    return {
        "function": str(function),
        "arguments": serialize_value(arguments or {}),
    }


def main() -> int:
    suite = get_suite(
        BENCHMARK_VERSION,
        SUITE_NAME,
    )

    base_environment = (
        suite.load_and_inject_default_environment({})
    )

    selected: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    for task_id, task in suite.user_tasks.items():
        try:
            task_environment = task.init_environment(
                base_environment.model_copy(deep=True)
            )

            ground_truth = task.ground_truth(
                task_environment
            )

            calls = [
                call_to_dict(call)
                for call in ground_truth
            ]

            legitimate_write_calls = [
                call
                for call in calls
                if call["function"] in WRITE_TOOLS
            ]

            if not legitimate_write_calls:
                continue

            selected.append(
                {
                    "task_id": task_id,
                    "prompt": task.PROMPT,
                    "ground_truth_calls": calls,
                    "legitimate_write_calls": (
                        legitimate_write_calls
                    ),
                    "write_tools": sorted(
                        {
                            call["function"]
                            for call
                            in legitimate_write_calls
                        }
                    ),
                }
            )

        except Exception as error:
            failures[task_id] = (
                f"{type(error).__name__}: {error}"
            )

    selected.sort(
        key=lambda record: record["task_id"]
    )

    artifact = {
        "schema_version": "0.1",
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "write_tool_vocabulary": sorted(WRITE_TOOLS),
        "selected_task_count": len(selected),
        "selected_tasks": selected,
        "audit_failures": failures,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=== AGENTDOJO LEGITIMATE WRITE-TASK AUDIT ===")
    print(f"Benchmark: {BENCHMARK_VERSION}")
    print(f"Suite: {SUITE_NAME}")
    print(f"Write tasks found: {len(selected)}")
    print(f"Audit failures: {len(failures)}")
    print()

    for record in selected:
        print(
            f"{record['task_id']}: "
            f"{record['write_tools']}"
        )
        print(f"  Prompt: {record['prompt']}")

        for call in record["legitimate_write_calls"]:
            print(
                "  Required call: "
                f"{call['function']}"
            )
            print(
                "  Arguments: "
                + json.dumps(
                    call["arguments"],
                    sort_keys=True,
                )
            )

        print()

    print(f"Artifact: {OUTPUT_PATH}")
    print("NO API CALLS WERE MADE.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
