"""Generate manuscript-ready analysis for a SentryGraph experiment batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTCOME_ORDER = [
    "both_resisted",
    "prevented",
    "both_compromised",
    "regression",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse matched baseline and SentryGraph AgentDojo results."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Experiment directory containing summary.json and paired CSV.",
    )
    parser.add_argument(
        "--analysis-status",
        choices=("exploratory", "confirmatory"),
        default="exploratory",
        help=(
            "Protocol status recorded in manuscript-ready "
            "analysis metadata."
        ),
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no", "", "none"}:
        return False

    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"none", "nan", "null"}:
        return None

    return float(text)


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def format_rate(value: float | None) -> str:
    if value is None:
        return ""

    return f"{value:.6f}"


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Wilson score interval."""

    if total == 0:
        return None, None

    proportion = successes / total
    denominator = 1.0 + (z * z / total)

    centre = (
        proportion
        + (z * z / (2.0 * total))
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                proportion * (1.0 - proportion) / total
                + z * z / (4.0 * total * total)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def exact_mcnemar_p(
    baseline_only_successes: int,
    sentrygraph_only_successes: int,
) -> float:
    """
    Exact two-sided McNemar test.

    baseline_only_successes:
        Baseline compromised while SentryGraph resisted.

    sentrygraph_only_successes:
        Baseline resisted while SentryGraph was compromised.
    """

    discordant = (
        baseline_only_successes
        + sentrygraph_only_successes
    )

    if discordant == 0:
        return 1.0

    smaller = min(
        baseline_only_successes,
        sentrygraph_only_successes,
    )

    tail_probability = sum(
        math.comb(discordant, index)
        for index in range(smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2.0 * tail_probability)


def median_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return statistics.median(values)


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return statistics.fmean(values)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pairs: list[dict[str, Any]] = []

    for row in rows:
        record = dict(row)

        for field in [
            "baseline_utility",
            "sentrygraph_utility",
            "baseline_attack_success",
            "sentrygraph_attack_success",
            "sentrygraph_contains_blocks",
            "sentrygraph_contains_governs",
        ]:
            record[field] = parse_bool(record[field])

        record["trial"] = int(record["trial"])
        record["sentrygraph_denied_actions"] = int(
            record["sentrygraph_denied_actions"]
        )

        for field in [
            "baseline_duration_seconds",
            "sentrygraph_duration_seconds",
            "sentrygraph_tool_latency_p50_ms",
            "sentrygraph_tool_latency_p95_ms",
        ]:
            record[field] = parse_float(record[field])

        record["baseline_only_compromise"] = (
            record["baseline_attack_success"]
            and not record["sentrygraph_attack_success"]
        )

        record["sentrygraph_only_compromise"] = (
            not record["baseline_attack_success"]
            and record["sentrygraph_attack_success"]
        )

        record["both_compromised"] = (
            record["baseline_attack_success"]
            and record["sentrygraph_attack_success"]
        )

        record["both_resisted"] = (
            not record["baseline_attack_success"]
            and not record["sentrygraph_attack_success"]
        )

        record["utility_preserved"] = (
            record["baseline_utility"]
            and record["sentrygraph_utility"]
        )

        record["security_and_utility_success"] = (
            not record["sentrygraph_attack_success"]
            and record["sentrygraph_utility"]
        )

        pairs.append(record)

    return pairs


def summarize_group(
    name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_count = len(records)

    baseline_successes = sum(
        record["baseline_attack_success"]
        for record in records
    )

    sentrygraph_successes = sum(
        record["sentrygraph_attack_success"]
        for record in records
    )

    prevented = sum(
        record["baseline_only_compromise"]
        for record in records
    )

    regressions = sum(
        record["sentrygraph_only_compromise"]
        for record in records
    )

    both_compromised = sum(
        record["both_compromised"]
        for record in records
    )

    both_resisted = sum(
        record["both_resisted"]
        for record in records
    )

    baseline_utility = sum(
        record["baseline_utility"]
        for record in records
    )

    sentrygraph_utility = sum(
        record["sentrygraph_utility"]
        for record in records
    )

    preserved_utility = sum(
        record["utility_preserved"]
        for record in records
    )

    secure_and_useful = sum(
        record["security_and_utility_success"]
        for record in records
    )

    denial_pairs = sum(
        record["sentrygraph_denied_actions"] > 0
        for record in records
    )

    blocking_pairs = sum(
        record["sentrygraph_contains_blocks"]
        for record in records
    )

    baseline_rate = safe_rate(
        baseline_successes,
        pair_count,
    )

    sentrygraph_rate = safe_rate(
        sentrygraph_successes,
        pair_count,
    )

    baseline_ci = wilson_interval(
        baseline_successes,
        pair_count,
    )

    sentrygraph_ci = wilson_interval(
        sentrygraph_successes,
        pair_count,
    )

    conditional_prevention = safe_rate(
        prevented,
        baseline_successes,
    )

    baseline_durations = [
        record["baseline_duration_seconds"]
        for record in records
        if record["baseline_duration_seconds"] is not None
    ]

    sentrygraph_durations = [
        record["sentrygraph_duration_seconds"]
        for record in records
        if record["sentrygraph_duration_seconds"] is not None
    ]

    duration_differences = [
        (
            record["sentrygraph_duration_seconds"]
            - record["baseline_duration_seconds"]
        )
        for record in records
        if (
            record["baseline_duration_seconds"] is not None
            and record["sentrygraph_duration_seconds"] is not None
        )
    ]

    p50_latencies = [
        record["sentrygraph_tool_latency_p50_ms"]
        for record in records
        if record["sentrygraph_tool_latency_p50_ms"] is not None
    ]

    p95_latencies = [
        record["sentrygraph_tool_latency_p95_ms"]
        for record in records
        if record["sentrygraph_tool_latency_p95_ms"] is not None
    ]

    return {
        "group": name,
        "pairs": pair_count,
        "baseline_attack_successes": baseline_successes,
        "baseline_attack_success_rate": baseline_rate,
        "baseline_asr_ci95_low": baseline_ci[0],
        "baseline_asr_ci95_high": baseline_ci[1],
        "sentrygraph_attack_successes": sentrygraph_successes,
        "sentrygraph_attack_success_rate": sentrygraph_rate,
        "sentrygraph_asr_ci95_low": sentrygraph_ci[0],
        "sentrygraph_asr_ci95_high": sentrygraph_ci[1],
        "absolute_asr_reduction": (
            None
            if baseline_rate is None or sentrygraph_rate is None
            else baseline_rate - sentrygraph_rate
        ),
        "prevented_attacks": prevented,
        "conditional_prevention_rate": conditional_prevention,
        "regressions": regressions,
        "both_compromised": both_compromised,
        "both_resisted": both_resisted,
        "exact_mcnemar_p": exact_mcnemar_p(
            prevented,
            regressions,
        ),
        "baseline_utility_passes": baseline_utility,
        "baseline_utility_rate": safe_rate(
            baseline_utility,
            pair_count,
        ),
        "sentrygraph_utility_passes": sentrygraph_utility,
        "sentrygraph_utility_rate": safe_rate(
            sentrygraph_utility,
            pair_count,
        ),
        "utility_preserved_pairs": preserved_utility,
        "secure_and_useful_sentrygraph_pairs": secure_and_useful,
        "secure_and_useful_sentrygraph_rate": safe_rate(
            secure_and_useful,
            pair_count,
        ),
        "sentrygraph_pairs_with_denial": denial_pairs,
        "sentrygraph_pairs_with_blocks": blocking_pairs,
        "mean_baseline_duration_seconds": mean_or_none(
            baseline_durations
        ),
        "mean_sentrygraph_duration_seconds": mean_or_none(
            sentrygraph_durations
        ),
        "median_duration_difference_seconds": median_or_none(
            duration_differences
        ),
        "median_tool_latency_p50_ms": median_or_none(
            p50_latencies
        ),
        "median_tool_latency_p95_ms": median_or_none(
            p95_latencies
        ),
    }


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    fieldnames = list(records[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            serializable = {}

            for key, value in record.items():
                if isinstance(value, bool):
                    serializable[key] = value
                elif isinstance(value, float):
                    serializable[key] = f"{value:.10f}"
                elif value is None:
                    serializable[key] = ""
                elif isinstance(value, (list, dict)):
                    serializable[key] = json.dumps(value)
                else:
                    serializable[key] = value

            writer.writerow(serializable)


def grouped_summaries(
    pairs: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for pair in pairs:
        groups[str(pair[field])].append(pair)

    return [
        summarize_group(group_name, groups[group_name])
        for group_name in sorted(groups)
    ]


def create_asr_plot(
    attack_rows: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    labels = [
        row["group"].replace("_", "\n")
        for row in attack_rows
    ]

    baseline = [
        100.0 * row["baseline_attack_success_rate"]
        for row in attack_rows
    ]

    sentrygraph = [
        100.0 * row["sentrygraph_attack_success_rate"]
        for row in attack_rows
    ]

    positions = list(range(len(labels)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.2, 4.8))

    baseline_bars = axis.bar(
        [position - width / 2 for position in positions],
        baseline,
        width,
        label="Unprotected baseline",
        color="#C84A4A",
    )

    sentrygraph_bars = axis.bar(
        [position + width / 2 for position in positions],
        sentrygraph,
        width,
        label="SentryGraph",
        color="#2878B5",
    )

    axis.set_ylabel("Attack success rate (%)")
    axis.set_xlabel("Attack family")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 105)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    axis.legend(frameon=False)

    axis.bar_label(
        baseline_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    axis.bar_label(
        sentrygraph_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    figure.tight_layout()

    for extension in ["pdf", "png"]:
        figure.savefig(
            output_directory
            / f"attack_success_by_family.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(figure)


def create_outcome_plot(
    attack_rows: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    labels = [
        row["group"].replace("_", "\n")
        for row in attack_rows
    ]

    colours = {
        "both_resisted": "#5AAE61",
        "prevented": "#2878B5",
        "both_compromised": "#C84A4A",
        "regression": "#8E5EA2",
    }

    display_names = {
        "both_resisted": "Both resisted",
        "prevented": "Prevented by SentryGraph",
        "both_compromised": "Both compromised",
        "regression": "SentryGraph-only compromise",
    }

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bottoms = [0] * len(attack_rows)

    outcome_fields = {
        "both_resisted": "both_resisted",
        "prevented": "prevented_attacks",
        "both_compromised": "both_compromised",
        "regression": "regressions",
    }

    for outcome in OUTCOME_ORDER:
        field = outcome_fields[outcome]

        values = [
            row[field]
            for row in attack_rows
        ]

        axis.bar(
            labels,
            values,
            bottom=bottoms,
            label=display_names[outcome],
            color=colours[outcome],
        )

        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, values)
        ]

    axis.set_ylabel("Number of matched attack pairs")
    axis.set_xlabel("Attack family")
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    axis.legend(
        frameon=False,
        fontsize=8,
    )

    figure.tight_layout()

    for extension in ["pdf", "png"]:
        figure.savefig(
            output_directory
            / f"paired_outcomes_by_family.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(figure)


def create_utility_plot(
    attack_rows: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    labels = [
        row["group"].replace("_", "\n")
        for row in attack_rows
    ]

    baseline = [
        100.0 * row["baseline_utility_rate"]
        for row in attack_rows
    ]

    sentrygraph = [
        100.0 * row["sentrygraph_utility_rate"]
        for row in attack_rows
    ]

    positions = list(range(len(labels)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.2, 4.8))

    baseline_bars = axis.bar(
        [position - width / 2 for position in positions],
        baseline,
        width,
        label="Unprotected baseline",
        color="#E69F00",
    )

    sentrygraph_bars = axis.bar(
        [position + width / 2 for position in positions],
        sentrygraph,
        width,
        label="SentryGraph",
        color="#009E73",
    )

    axis.set_ylabel("Task utility rate (%)")
    axis.set_xlabel("Attack family")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 105)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    axis.legend(frameon=False)

    axis.bar_label(
        baseline_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    axis.bar_label(
        sentrygraph_bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

    figure.tight_layout()

    for extension in ["pdf", "png"]:
        figure.savefig(
            output_directory
            / f"utility_by_attack_family.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()

    summary_path = root / "summary.json"
    paired_path = root / "paired_attack_results.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not paired_path.exists():
        raise FileNotFoundError(paired_path)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    pairs = load_pairs(paired_path)

    if not pairs:
        raise RuntimeError("No matched attack pairs were found.")

    analysis_directory = root / "analysis"
    analysis_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall = summarize_group(
        "overall",
        pairs,
    )

    attack_rows = grouped_summaries(
        pairs,
        "attack",
    )

    task_rows = grouped_summaries(
        pairs,
        "task_id",
    )

    model_rows = grouped_summaries(
        pairs,
        "model",
    )

    write_csv(
        analysis_directory / "security_by_attack.csv",
        attack_rows,
    )

    write_csv(
        analysis_directory / "security_by_task.csv",
        task_rows,
    )

    write_csv(
        analysis_directory / "security_by_model.csv",
        model_rows,
    )

    utility_rows = []

    for row in attack_rows:
        utility_rows.append(
            {
                "attack": row["group"],
                "pairs": row["pairs"],
                "baseline_utility_passes": (
                    row["baseline_utility_passes"]
                ),
                "baseline_utility_rate": (
                    row["baseline_utility_rate"]
                ),
                "sentrygraph_utility_passes": (
                    row["sentrygraph_utility_passes"]
                ),
                "sentrygraph_utility_rate": (
                    row["sentrygraph_utility_rate"]
                ),
                "utility_rate_difference": (
                    row["sentrygraph_utility_rate"]
                    - row["baseline_utility_rate"]
                ),
                "utility_preserved_pairs": (
                    row["utility_preserved_pairs"]
                ),
                "secure_and_useful_sentrygraph_pairs": (
                    row[
                        "secure_and_useful_sentrygraph_pairs"
                    ]
                ),
                "secure_and_useful_sentrygraph_rate": (
                    row[
                        "secure_and_useful_sentrygraph_rate"
                    ]
                ),
            }
        )

    write_csv(
        analysis_directory / "utility_by_attack.csv",
        utility_rows,
    )

    latency_rows = []

    for row in [overall, *attack_rows]:
        latency_rows.append(
            {
                "group": row["group"],
                "pairs": row["pairs"],
                "mean_baseline_duration_seconds": (
                    row["mean_baseline_duration_seconds"]
                ),
                "mean_sentrygraph_duration_seconds": (
                    row["mean_sentrygraph_duration_seconds"]
                ),
                "median_duration_difference_seconds": (
                    row["median_duration_difference_seconds"]
                ),
                "median_tool_latency_p50_ms": (
                    row["median_tool_latency_p50_ms"]
                ),
                "median_tool_latency_p95_ms": (
                    row["median_tool_latency_p95_ms"]
                ),
            }
        )

    write_csv(
        analysis_directory / "latency_summary.csv",
        latency_rows,
    )

    paired_export = []

    for pair in pairs:
        paired_export.append(
            {
                "pair_id": pair["pair_id"],
                "model": pair["model"],
                "task_id": pair["task_id"],
                "attack": pair["attack"],
                "injection_task_id": (
                    pair["injection_task_id"]
                ),
                "trial": pair["trial"],
                "baseline_utility": (
                    pair["baseline_utility"]
                ),
                "sentrygraph_utility": (
                    pair["sentrygraph_utility"]
                ),
                "baseline_attack_success": (
                    pair["baseline_attack_success"]
                ),
                "sentrygraph_attack_success": (
                    pair["sentrygraph_attack_success"]
                ),
                "prevented": (
                    pair["baseline_only_compromise"]
                ),
                "regression": (
                    pair["sentrygraph_only_compromise"]
                ),
                "utility_preserved": (
                    pair["utility_preserved"]
                ),
                "sentrygraph_denied_actions": (
                    pair["sentrygraph_denied_actions"]
                ),
                "sentrygraph_denied_tools": (
                    pair["sentrygraph_denied_tools"]
                ),
                "sentrygraph_contains_blocks": (
                    pair["sentrygraph_contains_blocks"]
                ),
                "block_attributed_prevention": (
                    pair["baseline_only_compromise"]
                    and pair["sentrygraph_contains_blocks"]
                ),
                "outcome": pair["outcome"],
            }
        )

    write_csv(
        analysis_directory / "paired_outcomes.csv",
        paired_export,
    )

    def block_attribution(
        selected_pairs: list[dict[str, Any]],
    ) -> dict[str, int]:
        prevented_with_block = sum(
            pair["baseline_only_compromise"]
            and pair["sentrygraph_contains_blocks"]
            for pair in selected_pairs
        )

        prevented_without_block = sum(
            pair["baseline_only_compromise"]
            and not pair["sentrygraph_contains_blocks"]
            for pair in selected_pairs
        )

        proactive_blocks = sum(
            pair["outcome"] == "both_resisted"
            and pair["sentrygraph_contains_blocks"]
            for pair in selected_pairs
        )

        all_blocked_pairs = sum(
            pair["sentrygraph_contains_blocks"]
            for pair in selected_pairs
        )

        return {
            "prevented_with_block": (
                prevented_with_block
            ),
            "prevented_without_block": (
                prevented_without_block
            ),
            "proactive_blocks_both_resisted": (
                proactive_blocks
            ),
            "all_pairs_with_block": (
                all_blocked_pairs
            ),
        }

    block_attribution_metrics = {
        "overall": block_attribution(pairs),
        "by_task": {
            task_id: block_attribution(
                [
                    pair
                    for pair in pairs
                    if pair["task_id"] == task_id
                ]
            )
            for task_id in sorted(
                {
                    pair["task_id"]
                    for pair in pairs
                }
            )
        },
        "by_attack": {
            attack: block_attribution(
                [
                    pair
                    for pair in pairs
                    if pair["attack"] == attack
                ]
            )
            for attack in sorted(
                {
                    pair["attack"]
                    for pair in pairs
                }
            )
        },
    }

    manuscript_metrics = {
        "schema_version": "0.2",
        "experiment_root": str(root),
        "analysis_status": args.analysis_status,
        "source_summary": summary,
        "overall": overall,
        "by_attack": attack_rows,
        "by_task": task_rows,
        "by_model": model_rows,
        "block_attribution": (
            block_attribution_metrics
        ),
        "interpretation": {
            "analysis_status": args.analysis_status,
            "exploratory_status": (
                args.analysis_status == "exploratory"
            ),
            "confirmatory_status": (
                args.analysis_status == "confirmatory"
            ),
            "single_model": len(model_rows) == 1,
            "single_trial": (
                len(
                    {
                        pair["trial"]
                        for pair in pairs
                    }
                )
                == 1
            ),
            "no_security_regressions": (
                overall["regressions"] == 0
            ),
            "all_observed_baseline_compromises_prevented": (
                overall["prevented_attacks"]
                == overall["baseline_attack_successes"]
            ),
        },
    }

    (
        analysis_directory / "manuscript_metrics.json"
    ).write_text(
        json.dumps(
            manuscript_metrics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    create_asr_plot(
        attack_rows,
        analysis_directory,
    )

    create_outcome_plot(
        attack_rows,
        analysis_directory,
    )

    create_utility_plot(
        attack_rows,
        analysis_directory,
    )

    print("=== SENTRYGRAPH BATCH ANALYSIS ===")
    print(f"Experiment: {root}")
    print(f"Matched pairs: {overall['pairs']}")
    print(
        "Baseline attack successes:",
        overall["baseline_attack_successes"],
    )
    print(
        "SentryGraph attack successes:",
        overall["sentrygraph_attack_successes"],
    )
    print(
        "Prevented attacks:",
        overall["prevented_attacks"],
    )
    print(
        "Security regressions:",
        overall["regressions"],
    )
    print(
        "Baseline ASR:",
        f"{100 * overall['baseline_attack_success_rate']:.2f}%",
    )
    print(
        "SentryGraph ASR:",
        f"{100 * overall['sentrygraph_attack_success_rate']:.2f}%",
    )
    print(
        "Conditional prevention:",
        (
            f"{100 * overall['conditional_prevention_rate']:.2f}%"
            if overall["conditional_prevention_rate"] is not None
            else "N/A"
        ),
    )
    print(
        "Exact McNemar p-value:",
        f"{overall['exact_mcnemar_p']:.8f}",
    )
    print(
        "Baseline attack-time utility:",
        (
            f"{overall['baseline_utility_passes']}/"
            f"{overall['pairs']}"
        ),
    )
    print(
        "SentryGraph attack-time utility:",
        (
            f"{overall['sentrygraph_utility_passes']}/"
            f"{overall['pairs']}"
        ),
    )
    print(
        "Median SentryGraph p95 tool latency:",
        (
            f"{overall['median_tool_latency_p95_ms']:.6f} ms"
            if overall["median_tool_latency_p95_ms"] is not None
            else "N/A"
        ),
    )
    print(f"Artifacts: {analysis_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
