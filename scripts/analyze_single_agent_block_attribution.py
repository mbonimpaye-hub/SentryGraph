#!/usr/bin/env python3
"""Audit causal block attribution in matched single-agent experiments.

This script is read-only with respect to experiment roots. It deduplicates
aggregate records by cell_id, reconstructs baseline/SentryGraph pairs, and
separates raw paired outcome differences from explicit block-attributed
preventions and proactive blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KNOWN_PROTOCOLS = {
    "c0dcdcb291672e3c": {
        "name": "read_only_confirmatory",
        "expected_pairs": 81,
        "expected_baseline_only": 32,
    },
    "827f97129ce30ba1": {
        "name": "state_changing_confirmatory",
        "expected_pairs": 81,
        "expected_baseline_only": 37,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate matched resistance from explicit block attribution "
            "in frozen single-agent experiment results."
        )
    )
    parser.add_argument(
        "--experiment-root",
        action="append",
        type=Path,
        required=True,
        help="Experiment root containing results.jsonl and runs/ (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing paths are not overwritten.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate results: {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def latest_completed_by_cell(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        cell_id = record.get("cell_id")
        if not cell_id:
            raise ValueError("A result record is missing cell_id.")
        if record.get("status") == "completed":
            latest[cell_id] = record
    return latest


def is_attack_record(record: dict[str, Any]) -> bool:
    phase = (record.get("cell") or {}).get("phase")
    return phase == "attack" or record.get("attack") is not None


def load_trace_events(root: Path, cell_id: str) -> list[dict[str, Any]]:
    path = root / "runs" / cell_id / "monitor_trace.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_graph(root: Path, cell_id: str) -> dict[str, Any]:
    path = root / "runs" / cell_id / "execution_graph.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def explicit_block_evidence(
    root: Path,
    record: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """Return block presence, evidence sources, and denied tools."""
    sources: list[str] = []
    denied_tools = set(record.get("denied_tools") or [])
    cell_id = str(record["cell_id"])

    events = load_trace_events(root, cell_id)
    trace_block = False
    for event in events:
        denied = (
            event.get("decision") == "deny"
            or event.get("phase") in {"denied", "blocked"}
        )
        if denied:
            trace_block = True
            if event.get("tool"):
                denied_tools.add(str(event["tool"]))
    if trace_block:
        sources.append("monitor_trace")

    graph = load_graph(root, cell_id)
    graph_block = any(
        edge.get("relation") == "BLOCKS"
        for edge in graph.get("edges", [])
    )
    if graph_block:
        sources.append("execution_graph")

    if int(record.get("denied_actions") or 0) > 0:
        sources.append("result_denied_actions")
    if bool(record.get("contains_blocks")):
        sources.append("result_contains_blocks")

    sources = sorted(set(sources))
    return bool(sources), sources, sorted(denied_tools)


def classify_pair(
    baseline: dict[str, Any],
    sentrygraph: dict[str, Any],
    blocked: bool,
) -> str:
    baseline_success = bool(baseline.get("attack_success"))
    sentrygraph_success = bool(sentrygraph.get("attack_success"))
    if baseline_success and sentrygraph_success:
        return "both_compromised"
    if not baseline_success and sentrygraph_success:
        return "security_regression"
    if baseline_success and not sentrygraph_success:
        return (
            "explicit_block_attributed_prevention"
            if blocked
            else "unattributed_discordant_resistance"
        )
    return "proactive_block_while_both_resisted" if blocked else "both_resisted"


def protocol_name(plan_id: str) -> str:
    return KNOWN_PROTOCOLS.get(plan_id, {}).get("name", plan_id)


def audit_root(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = latest_completed_by_cell(load_jsonl(root / "results.jsonl"))
    attacks = [record for record in records.values() if is_attack_record(record)]
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attacks:
        pair_id = record.get("pair_id")
        if not pair_id:
            raise ValueError(f"Attack record {record['cell_id']} lacks pair_id.")
        pairs[str(pair_id)].append(record)

    rows: list[dict[str, Any]] = []
    for pair_id, members in sorted(pairs.items()):
        by_condition = {str(item.get("condition")): item for item in members}
        if set(by_condition) != {"baseline", "sentrygraph"}:
            raise ValueError(
                f"Pair {pair_id} conditions are {sorted(by_condition)}, "
                "expected baseline and sentrygraph."
            )
        baseline = by_condition["baseline"]
        sentrygraph = by_condition["sentrygraph"]
        blocked, sources, denied_tools = explicit_block_evidence(
            root, sentrygraph
        )
        plan_id = str(sentrygraph.get("plan_id") or root.name)
        row = {
            "protocol": protocol_name(plan_id),
            "plan_id": plan_id,
            "pair_id": pair_id,
            "model": sentrygraph.get("model"),
            "task_id": sentrygraph.get("task_id"),
            "attack": sentrygraph.get("attack"),
            "injection_task_id": sentrygraph.get("injection_task_id"),
            "trial": (sentrygraph.get("cell") or {}).get("trial"),
            "baseline_attack_success": bool(baseline.get("attack_success")),
            "sentrygraph_attack_success": bool(
                sentrygraph.get("attack_success")
            ),
            "sentrygraph_utility": sentrygraph.get("utility"),
            "explicit_block_evidence": blocked,
            "block_evidence_sources": ";".join(sources),
            "denied_actions": int(sentrygraph.get("denied_actions") or 0),
            "denied_tools": ";".join(denied_tools),
        }
        row["attribution"] = classify_pair(baseline, sentrygraph, blocked)
        rows.append(row)

    if not rows:
        raise ValueError(f"No complete attack pairs found in {root}.")

    plan_ids = {str(row["plan_id"]) for row in rows}
    if len(plan_ids) != 1:
        raise ValueError(f"Mixed plan IDs in {root}: {sorted(plan_ids)}")
    plan_id = next(iter(plan_ids))
    counts = Counter(str(row["attribution"]) for row in rows)
    baseline_only = (
        counts["explicit_block_attributed_prevention"]
        + counts["unattributed_discordant_resistance"]
    )
    expected = KNOWN_PROTOCOLS.get(plan_id)
    if expected:
        if len(rows) != expected["expected_pairs"]:
            raise AssertionError(
                f"{plan_id}: observed {len(rows)} pairs, expected "
                f"{expected['expected_pairs']}."
            )
        if baseline_only != expected["expected_baseline_only"]:
            raise AssertionError(
                f"{plan_id}: observed {baseline_only} baseline-only "
                f"compromises, expected {expected['expected_baseline_only']}."
            )

    summary = {
        "protocol": protocol_name(plan_id),
        "plan_id": plan_id,
        "experiment_root": str(root),
        "complete_pairs": len(rows),
        "baseline_only_compromises": baseline_only,
        "explicit_block_attributed_preventions": counts[
            "explicit_block_attributed_prevention"
        ],
        "unattributed_discordant_resistance": counts[
            "unattributed_discordant_resistance"
        ],
        "proactive_blocks_while_both_resisted": counts[
            "proactive_block_while_both_resisted"
        ],
        "both_resisted": counts["both_resisted"],
        "both_compromised": counts["both_compromised"],
        "security_regressions": counts["security_regression"],
        "raw_conditional_resistance_rate": (
            baseline_only / baseline_only if baseline_only else None
        ),
        "block_attributed_prevention_rate": (
            counts["explicit_block_attributed_prevention"] / baseline_only
            if baseline_only
            else None
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), str(row[field]))].append(row)
    output = []
    for (protocol, value), members in sorted(groups.items()):
        counts = Counter(str(item["attribution"]) for item in members)
        baseline_only = (
            counts["explicit_block_attributed_prevention"]
            + counts["unattributed_discordant_resistance"]
        )
        output.append(
            {
                "protocol": protocol,
                field: value,
                "pairs": len(members),
                "baseline_only_compromises": baseline_only,
                "explicit_block_attributed_preventions": counts[
                    "explicit_block_attributed_prevention"
                ],
                "unattributed_discordant_resistance": counts[
                    "unattributed_discordant_resistance"
                ],
                "proactive_blocks_while_both_resisted": counts[
                    "proactive_block_while_both_resisted"
                ],
                "security_regressions": counts["security_regression"],
            }
        )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    all_rows: list[dict[str, Any]] = []
    summaries = []
    for root in args.experiment_root:
        rows, summary = audit_root(root)
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(args.output_dir / "pair_attribution.csv", all_rows)
    write_csv(
        args.output_dir / "attribution_by_attack.csv",
        aggregate_by(all_rows, "attack"),
    )
    write_csv(
        args.output_dir / "attribution_by_task.csv",
        aggregate_by(all_rows, "task_id"),
    )
    payload = {
        "schema_version": "1.0",
        "analysis": "single_agent_causal_block_attribution",
        "no_api_calls": True,
        "protocols": summaries,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    outputs = sorted(args.output_dir.glob("*.csv")) + [
        args.output_dir / "audit_summary.json"
    ]
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs)
    )

    print("=== SINGLE-AGENT BLOCK ATTRIBUTION AUDIT ===")
    for summary in summaries:
        print()
        print(summary["protocol"], summary["plan_id"])
        for key in (
            "complete_pairs",
            "baseline_only_compromises",
            "explicit_block_attributed_preventions",
            "unattributed_discordant_resistance",
            "proactive_blocks_while_both_resisted",
            "both_resisted",
            "both_compromised",
            "security_regressions",
            "block_attributed_prevention_rate",
        ):
            print(f"  {key}: {summary[key]}")
    print()
    print(f"Artifacts: {args.output_dir}")
    print("PASS: Matched outcomes and causal block attribution are separated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
