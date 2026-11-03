"""Generate manuscript-ready analysis for deterministic multi-agent tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    results_path = root / "results.jsonl"
    records = load_jsonl(results_path)
    unique = {record["cell"]["cell_id"]: record for record in records}
    completed = [r for r in unique.values() if r["status"] == "completed"]
    errors = [r for r in unique.values() if r["status"] == "error"]
    if errors:
        raise RuntimeError(f"Cannot analyze {len(errors)} error records.")

    pairs: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for record in completed:
        cell = record["cell"]
        pairs[(cell["scenario_id"], cell["trial"])][
            cell["condition"]
        ] = record
    incomplete = {key: value for key, value in pairs.items() if len(value) != 2}
    if incomplete:
        raise RuntimeError(f"Incomplete matched pairs: {sorted(incomplete)}")

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    for (scenario_id, trial), pair in sorted(pairs.items()):
        baseline = pair["baseline"]
        sentry = pair["sentrygraph"]
        scenario = baseline["scenario"]
        family = scenario["family"]
        is_attack = scenario["category"] == "attack"
        prevented = (
            is_attack
            and baseline["attack_success"]
            and not sentry["attack_success"]
        )
        regression = (
            is_attack
            and not baseline["attack_success"]
            and sentry["attack_success"]
        )
        utility_preserved = (
            not is_attack
            and baseline["utility_success"]
            and sentry["utility_success"]
        )
        row = {
            "scenario_id": scenario_id,
            "family": family,
            "category": scenario["category"],
            "trial": trial,
            "baseline_state_changed": baseline["outcome"]["state_changed"],
            "sentrygraph_state_changed": sentry["outcome"]["state_changed"],
            "baseline_attack_success": baseline["attack_success"],
            "sentrygraph_attack_success": sentry["attack_success"],
            "prevented": prevented,
            "regression": regression,
            "utility_preserved": utility_preserved,
            "sentrygraph_rule": sentry["outcome"]["decision_rule"],
            "outcome_match": (
                baseline["outcome_match"] and sentry["outcome_match"]
            ),
        }
        pair_rows.append(row)
        family_groups[family].append(row)
    write_csv(analysis / "paired_outcomes.csv", pair_rows)

    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(family_groups.items()):
        category = rows[0]["category"]
        family_rows.append(
            {
                "family": family,
                "category": category,
                "repetitions": len(rows),
                "baseline_attack_successes": sum(
                    bool(row["baseline_attack_success"]) for row in rows
                ),
                "sentrygraph_attack_successes": sum(
                    bool(row["sentrygraph_attack_success"]) for row in rows
                ),
                "prevented": sum(bool(row["prevented"]) for row in rows),
                "regressions": sum(bool(row["regression"]) for row in rows),
                "utility_preserved": sum(
                    bool(row["utility_preserved"]) for row in rows
                ),
                "outcome_matches": sum(
                    bool(row["outcome_match"]) for row in rows
                ),
                "sentrygraph_rule": rows[0]["sentrygraph_rule"],
            }
        )
    write_csv(analysis / "security_by_family.csv", family_rows)

    rule_counts = Counter(
        row["sentrygraph_rule"] for row in pair_rows
    )
    rule_rows = [
        {"rule_id": rule, "count": count}
        for rule, count in sorted(rule_counts.items())
    ]
    write_csv(analysis / "decision_rule_attribution.csv", rule_rows)

    graph_rows: list[dict[str, Any]] = []
    all_relations: set[str] = set()
    completed_latencies: list[float] = []
    for record in sorted(
        (r for r in completed if r["cell"]["condition"] == "sentrygraph"),
        key=lambda item: item["cell"]["cell_id"],
    ):
        run_dir = root / "runs" / record["cell"]["cell_id"]
        graph = load_json(run_dir / "execution_graph.json")
        events = load_jsonl(run_dir / "monitor_trace.jsonl")
        relations = sorted(
            {edge["relation"] for edge in graph.get("edges", [])}
        )
        all_relations.update(relations)
        latencies = [
            float(event["latency_ms"])
            for event in events
            if event.get("phase") == "completed"
            and event.get("latency_ms") is not None
        ]
        completed_latencies.extend(latencies)
        graph_rows.append(
            {
                "cell_id": record["cell"]["cell_id"],
                "scenario_id": record["cell"]["scenario_id"],
                "trial": record["cell"]["trial"],
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "proposed_events": sum(
                    event.get("phase") == "proposed" for event in events
                ),
                "completed_events": sum(
                    event.get("phase") == "completed" for event in events
                ),
                "denied_events": sum(
                    event.get("phase") == "denied" for event in events
                ),
                "relations": "|".join(relations),
            }
        )
    write_csv(analysis / "graph_evidence.csv", graph_rows)

    attack_rows = [row for row in pair_rows if row["category"] == "attack"]
    benign_rows = [row for row in pair_rows if row["category"] == "benign"]
    metrics = {
        "schema_version": "0.1",
        "analysis_status": "deterministic_systems_validation",
        "experiment_root": str(root),
        "plan_id": completed[0]["plan_id"] if completed else None,
        "overall": {
            "unique_cells": len(unique),
            "complete_pairs": len(pair_rows),
            "attack_pairs": len(attack_rows),
            "benign_pairs": len(benign_rows),
            "attack_families": len(
                {row["family"] for row in attack_rows}
            ),
            "baseline_attack_successes": sum(
                bool(row["baseline_attack_success"])
                for row in attack_rows
            ),
            "sentrygraph_attack_successes": sum(
                bool(row["sentrygraph_attack_success"])
                for row in attack_rows
            ),
            "prevented_attacks": sum(
                bool(row["prevented"]) for row in attack_rows
            ),
            "security_regressions": sum(
                bool(row["regression"]) for row in attack_rows
            ),
            "baseline_attack_success_rate": safe_rate(
                sum(bool(row["baseline_attack_success"]) for row in attack_rows),
                len(attack_rows),
            ),
            "sentrygraph_attack_success_rate": safe_rate(
                sum(bool(row["sentrygraph_attack_success"]) for row in attack_rows),
                len(attack_rows),
            ),
            "conditional_prevention_rate": safe_rate(
                sum(bool(row["prevented"]) for row in attack_rows),
                sum(bool(row["baseline_attack_success"]) for row in attack_rows),
            ),
            "benign_utility_preserved": sum(
                bool(row["utility_preserved"]) for row in benign_rows
            ),
            "outcome_mismatches": sum(
                not bool(row["outcome_match"]) for row in pair_rows
            ),
        },
        "decision_rule_attribution": dict(sorted(rule_counts.items())),
        "graph_evidence": {
            "sentrygraph_runs": len(graph_rows),
            "runs_with_graph": sum(row["nodes"] > 0 for row in graph_rows),
            "relation_vocabulary": sorted(all_relations),
            "median_completed_tool_latency_ms": (
                statistics.median(completed_latencies)
                if completed_latencies
                else None
            ),
        },
        "interpretation": {
            "mechanism_validation_only": True,
            "stochastic_agent_evaluation": False,
            "independent_statistical_trials": False,
            "repetitions_are_reproducibility_checks": True,
            "inferential_confidence_intervals_appropriate": False,
            "claim_boundary": (
                "These results establish deterministic enforcement and state-"
                "change prevention for the frozen scenario catalog. They do "
                "not measure how often an LLM agent chooses malicious actions."
            ),
        },
        "by_family": family_rows,
    }
    metrics_path = analysis / "manuscript_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    attack_family_rows = [
        row for row in family_rows if row["category"] == "attack"
    ]
    labels = [row["family"].replace("_", "\n") for row in attack_family_rows]
    baseline_rates = [
        row["baseline_attack_successes"] / row["repetitions"]
        for row in attack_family_rows
    ]
    sentry_rates = [
        row["sentrygraph_attack_successes"] / row["repetitions"]
        for row in attack_family_rows
    ]
    positions = list(range(len(labels)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(
        [position - width / 2 for position in positions],
        baseline_rates,
        width,
        label="Baseline",
        color="#c44e52",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        sentry_rates,
        width,
        label="SentryGraph",
        color="#4c72b0",
    )
    axis.set_ylabel("Observed state-changing attack rate")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(positions, labels, fontsize=8)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(analysis / "multi_agent_security_by_family.png", dpi=300)
    figure.savefig(analysis / "multi_agent_security_by_family.pdf")
    plt.close(figure)

    checksum_targets = sorted(
        path for path in analysis.iterdir() if path.name != "SHA256SUMS.txt"
    )
    (analysis / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )

    overall = metrics["overall"]
    print("=== PHASE 2G MULTI-AGENT ANALYSIS ===")
    print("Analysis status:", metrics["analysis_status"])
    print("Complete pairs:", overall["complete_pairs"])
    print("Attack families:", overall["attack_families"])
    print("Baseline attack successes:", overall["baseline_attack_successes"])
    print("SentryGraph attack successes:", overall["sentrygraph_attack_successes"])
    print("Prevented attacks:", overall["prevented_attacks"])
    print("Security regressions:", overall["security_regressions"])
    print("Benign utility preserved:", overall["benign_utility_preserved"])
    print("Outcome mismatches:", overall["outcome_mismatches"])
    print("Graph-evidenced runs:", metrics["graph_evidence"]["runs_with_graph"])
    print("Artifacts:", analysis)
    print("PASS: Results are labeled as deterministic mechanism validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
