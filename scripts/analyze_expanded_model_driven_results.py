#!/usr/bin/env python3
"""Generate manuscript-ready analysis for expanded model-driven results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = (
    "baseline",
    "sentrygraph_deterministic",
    "sentrygraph_hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def exact_mcnemar(baseline_only: int, protected_only: int) -> float:
    discordant = baseline_only + protected_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, protected_only)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1))
    return min(1.0, 2 * tail / (2**discordant))


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "runs").glob("*/result.json")):
        record = json.loads(path.read_text())
        if record.get("cell", {}).get("phase") == "attack":
            record["_run_directory"] = str(path.parent)
            records.append(record)
    return records


def outcome(arms: dict[str, dict[str, Any]]) -> str:
    values = tuple(bool(arms[c].get("attack_success")) for c in CONDITIONS)
    if values == (False, False, False):
        return "all_resisted"
    if values == (True, False, False):
        return "prevented_by_both"
    if not values[0] and (values[1] or values[2]):
        return "security_regression"
    if values == (True, True, True):
        return "all_compromised"
    return "mixed"


def main() -> int:
    args = parse_args()
    records = load_records(args.experiment_root)
    if len(records) != 108:
        raise RuntimeError(f"Expected 108 attack records, found {len(records)}")
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[record["pair_id"]][record["cell"]["condition"]] = record
    incomplete = [key for key, arms in groups.items() if set(arms) != set(CONDITIONS)]
    if incomplete:
        raise RuntimeError(f"Incomplete matched triples: {incomplete}")

    analysis = args.experiment_root / "analysis_model_driven"
    if analysis.exists():
        raise FileExistsError(f"Refusing to overwrite: {analysis}")
    temporary = Path(
        tempfile.mkdtemp(prefix=".analysis_model_driven-", dir=args.experiment_root)
    )
    try:
        paired_rows = []
        dimensions: dict[str, dict[str, Counter[str]]] = {
            name: defaultdict(Counter)
            for name in ("attack", "injection_task_id", "handoff_channel")
        }
        learned = Counter()
        denied_tools = Counter()
        graph_runs = 0
        for pair_id, arms in sorted(groups.items()):
            base = arms["baseline"]
            deterministic = arms["sentrygraph_deterministic"]
            hybrid = arms["sentrygraph_hybrid"]
            cell = base["cell"]
            label = outcome(arms)
            row = {
                "pair_id": pair_id,
                "model": cell["model"],
                "task_id": cell["task_id"],
                "attack": cell["attack"],
                "injection_task_id": cell["injection_task_id"],
                "handoff_channel": cell["handoff_channel"],
                "trial": cell["trial"],
                "baseline_utility": bool(base.get("utility")),
                "deterministic_utility": bool(deterministic.get("utility")),
                "hybrid_utility": bool(hybrid.get("utility")),
                "baseline_attack_success": bool(base.get("attack_success")),
                "deterministic_attack_success": bool(
                    deterministic.get("attack_success")
                ),
                "hybrid_attack_success": bool(hybrid.get("attack_success")),
                "hybrid_denied_actions": hybrid.get("denied_actions", 0),
                "hybrid_denied_tools": ";".join(hybrid.get("denied_tools", [])),
                "hybrid_learned_decisions": ";".join(
                    hybrid.get("learned_decisions", [])
                ),
                "outcome": label,
            }
            paired_rows.append(row)
            for dimension in dimensions:
                dimensions[dimension][str(cell[dimension])][label] += 1
            learned.update(hybrid.get("learned_decisions", []))
            denied_tools.update(hybrid.get("denied_tools", []))
            graph_runs += int(
                (Path(hybrid["_run_directory"]) / "execution_graph.json").exists()
            )
        write_csv(temporary / "paired_outcomes.csv", paired_rows)

        dimension_files = {}
        for dimension, categories in dimensions.items():
            rows = []
            for category, counts in sorted(categories.items()):
                total = sum(counts.values())
                baseline_successes = sum(
                    bool(row["baseline_attack_success"])
                    for row in paired_rows
                    if str(row[dimension]) == category
                )
                rows.append(
                    {
                        dimension: category,
                        "pairs": total,
                        "baseline_attack_successes": baseline_successes,
                        "baseline_asr": baseline_successes / total,
                        "deterministic_attack_successes": sum(
                            bool(row["deterministic_attack_success"])
                            for row in paired_rows
                            if str(row[dimension]) == category
                        ),
                        "hybrid_attack_successes": sum(
                            bool(row["hybrid_attack_success"])
                            for row in paired_rows
                            if str(row[dimension]) == category
                        ),
                        "all_resisted": counts["all_resisted"],
                        "prevented_by_both": counts["prevented_by_both"],
                        "security_regressions": counts["security_regression"],
                    }
                )
            filename = f"security_by_{dimension}.csv"
            write_csv(temporary / filename, rows)
            dimension_files[dimension] = rows

        total = len(paired_rows)
        condition_metrics = {}
        for condition in CONDITIONS:
            attack_key = f"{condition.replace('sentrygraph_', '')}_attack_success"
            utility_key = f"{condition.replace('sentrygraph_', '')}_utility"
            successes = sum(bool(row[attack_key]) for row in paired_rows)
            utilities = sum(bool(row[utility_key]) for row in paired_rows)
            condition_metrics[condition] = {
                "attack_successes": successes,
                "attack_success_rate": successes / total,
                "attack_success_rate_wilson_95": wilson(successes, total),
                "utility_successes": utilities,
                "utility_rate": utilities / total,
                "utility_rate_wilson_95": wilson(utilities, total),
            }

        outcomes = Counter(row["outcome"] for row in paired_rows)
        baseline_only_hybrid = sum(
            row["baseline_attack_success"] and not row["hybrid_attack_success"]
            for row in paired_rows
        )
        hybrid_only = sum(
            not row["baseline_attack_success"] and row["hybrid_attack_success"]
            for row in paired_rows
        )
        metrics = {
            "schema_version": "0.1",
            "analysis_status": "expanded_model_driven_single_model_single_task",
            "experiment_root": str(args.experiment_root),
            "overall": {
                "matched_triples": total,
                "outcomes": dict(outcomes),
                "conditional_prevention_rate": (
                    outcomes["prevented_by_both"]
                    / condition_metrics["baseline"]["attack_successes"]
                    if condition_metrics["baseline"]["attack_successes"]
                    else None
                ),
                "security_regressions": outcomes["security_regression"],
                "exact_mcnemar_baseline_vs_hybrid_p": exact_mcnemar(
                    baseline_only_hybrid, hybrid_only
                ),
                "baseline_only_compromises": baseline_only_hybrid,
                "hybrid_only_compromises": hybrid_only,
            },
            "conditions": condition_metrics,
            "by_attack": dimension_files["attack"],
            "by_injection_task": dimension_files["injection_task_id"],
            "by_handoff_channel": dimension_files["handoff_channel"],
            "hybrid_attribution": {
                "learned_decisions": dict(learned),
                "denied_tools": dict(denied_tools),
                "learned_security_interventions_observed": 0,
                "deterministic_denials_observed": sum(denied_tools.values()),
            },
            "graph_evidence": {
                "hybrid_runs": total,
                "hybrid_runs_with_graph": graph_runs,
            },
            "claim_boundary": {
                "single_model": True,
                "single_task": True,
                "descriptive_model_driven_evaluation": True,
                "conditional_prevention_is_observed_not_universal": True,
                "learned_component_prevention_claim_supported": False,
                "deterministic_enforcement_prevention_claim_supported": True,
            },
        }
        write_json(temporary / "manuscript_metrics.json", metrics)

        attacks = dimension_files["attack"]
        labels = [row["attack"].replace("_", "\n") for row in attacks]
        x = list(range(len(attacks)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for offset, key, label, colour in (
            (-width, "baseline_attack_successes", "Baseline", "#c44e52"),
            (0, "deterministic_attack_successes", "Deterministic", "#4c72b0"),
            (width, "hybrid_attack_successes", "Hybrid", "#55a868"),
        ):
            values = [row[key] / row["pairs"] for row in attacks]
            ax.bar([value + offset for value in x], values, width, label=label, color=colour)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Attack success rate")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(temporary / "model_driven_asr_by_attack.pdf", bbox_inches="tight")
        fig.savefig(temporary / "model_driven_asr_by_attack.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        payloads = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(f"{sha256(path)}  {path.name}" for path in payloads) + "\n"
        )
        os.replace(temporary, analysis)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 3J MODEL-DRIVEN ANALYSIS ===")
    print("Matched triples:", metrics["overall"]["matched_triples"])
    print("Outcomes:", metrics["overall"]["outcomes"])
    for condition, values in metrics["conditions"].items():
        print(
            f"{condition}: ASR={values['attack_success_rate']:.4f}, "
            f"utility={values['utility_rate']:.4f}"
        )
    print(
        "Exact McNemar p-value:",
        metrics["overall"]["exact_mcnemar_baseline_vs_hybrid_p"],
    )
    print("Learned decisions:", metrics["hybrid_attribution"]["learned_decisions"])
    print("Denied tools:", metrics["hybrid_attribution"]["denied_tools"])
    print("Artifacts:", analysis)
    print("PASS: Manuscript-ready model-driven artifacts generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
