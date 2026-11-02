"""Build a leakage-resistant graph-conformance learning dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from sentrygraph.prefix_labeler import (
    label_prefixes,
    load_contract,
    load_graph,
    load_jsonl,
    proposed_actions,
    sanitize_execution_prefix,
)


SCHEMA_VERSION = "0.1"
SPLIT_RATIOS = {
    "train": 0.60,
    "validation": 0.15,
    "calibration": 0.10,
    "test": 0.15,
}
DEFAULT_ROOTS = (
    Path("outputs/experiments/c0dcdcb291672e3c"),
    Path("outputs/experiments/827f97129ce30ba1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build graph-prefix examples from completed SentryGraph runs."
        )
    )
    parser.add_argument(
        "--experiment-roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_ROOTS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/learning/graph_conformance_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def inner_contract(artifact: Mapping[str, Any]) -> dict[str, Any]:
    value = artifact.get("contract", artifact)
    if not isinstance(value, Mapping):
        raise TypeError("Contract payload must be a mapping.")
    return dict(value)


def add_graph_node(
    nodes: list[dict[str, Any]],
    seen: set[str],
    node_id: str,
    node_type: str,
    attributes: Mapping[str, Any],
) -> None:
    if node_id in seen:
        return
    seen.add(node_id)
    nodes.append(
        {
            "node_id": node_id,
            "node_type": node_type,
            "attributes": dict(attributes),
        }
    )


def contract_to_graph(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Convert serialized contract semantics to a heterogeneous graph."""

    contract = inner_contract(artifact)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    contract_node = "contract:0"
    principal_node = "principal:0"

    add_graph_node(
        nodes,
        seen,
        contract_node,
        "task_contract",
        {
            "objective": contract.get("objective"),
            "delegation_allowed": contract.get("delegation_allowed", False),
            "compiler": contract.get("compiler"),
            "compiler_version": contract.get("compiler_version"),
        },
    )
    add_graph_node(
        nodes,
        seen,
        principal_node,
        "principal",
        {"name": contract.get("principal")},
    )
    edges.append(
        {
            "source": principal_node,
            "target": contract_node,
            "relation": "BOUND_BY",
            "attributes": {},
        }
    )

    for authorization_index, authorization in enumerate(
        contract.get("permitted_tools", [])
    ):
        if not isinstance(authorization, Mapping):
            raise TypeError("Tool authorization must be a mapping.")
        tool = str(authorization["tool"])
        tool_node = f"tool:permitted:{authorization_index}"
        add_graph_node(
            nodes,
            seen,
            tool_node,
            "tool",
            {
                "name": tool,
                "authorization": "permitted",
                "allow_additional_arguments": authorization.get(
                    "allow_additional_arguments", True
                ),
                "approval_required": authorization.get(
                    "approval_required", False
                ),
            },
        )
        edges.append(
            {
                "source": contract_node,
                "target": tool_node,
                "relation": "PERMITS",
                "attributes": {},
            }
        )

        effect = str(authorization.get("effect", "unknown"))
        effect_node = f"effect:authorized:{authorization_index}"
        add_graph_node(
            nodes,
            seen,
            effect_node,
            "effect",
            {"name": effect, "authorization": "permitted"},
        )
        edges.append(
            {
                "source": tool_node,
                "target": effect_node,
                "relation": "HAS_EFFECT",
                "attributes": {},
            }
        )

        for constraint_index, constraint in enumerate(
            authorization.get("constraints", [])
        ):
            if not isinstance(constraint, Mapping):
                raise TypeError("Argument constraint must be a mapping.")
            constraint_node = (
                f"constraint:{authorization_index}:{constraint_index}"
            )
            add_graph_node(
                nodes,
                seen,
                constraint_node,
                "argument_constraint",
                {
                    "argument": constraint.get("argument"),
                    "operator": constraint.get("operator"),
                    "value": constraint.get("value"),
                },
            )
            edges.append(
                {
                    "source": tool_node,
                    "target": constraint_node,
                    "relation": "CONSTRAINS",
                    "attributes": {},
                }
            )

    for index, tool in enumerate(sorted(contract.get("prohibited_tools", []))):
        tool_node = f"tool:prohibited:{index}"
        add_graph_node(
            nodes,
            seen,
            tool_node,
            "tool",
            {"name": tool, "authorization": "prohibited"},
        )
        edges.append(
            {
                "source": contract_node,
                "target": tool_node,
                "relation": "PROHIBITS",
                "attributes": {},
            }
        )

    for index, effect in enumerate(
        sorted(contract.get("prohibited_effects", []))
    ):
        effect_node = f"effect:prohibited:{index}"
        add_graph_node(
            nodes,
            seen,
            effect_node,
            "effect",
            {"name": effect, "authorization": "prohibited"},
        )
        edges.append(
            {
                "source": contract_node,
                "target": effect_node,
                "relation": "PROHIBITS_EFFECT",
                "attributes": {},
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }


def canonicalize_execution_prefix(
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove run identifiers and timestamps while preserving topology."""

    identifier_map: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    nodes: list[dict[str, Any]] = []

    for node in prefix.get("nodes", []):
        node_type = str(node.get("node_type", "unknown"))
        local_id = f"{node_type}:{type_counts[node_type]}"
        type_counts[node_type] += 1
        identifier_map[str(node["node_id"])] = local_id
        attributes = dict(node.get("attributes", {}))
        attributes.pop("action_id", None)
        nodes.append(
            {
                "node_id": local_id,
                "node_type": node_type,
                "attributes": attributes,
            }
        )

    edges: list[dict[str, Any]] = []
    for edge in prefix.get("edges", []):
        source = identifier_map.get(str(edge.get("source")))
        target = identifier_map.get(str(edge.get("target")))
        if source is None or target is None:
            raise ValueError("Prefix contains an edge to a removed node.")
        edges.append(
            {
                "source": source,
                "target": target,
                "relation": edge.get("relation"),
                "step": edge.get("step"),
                "attributes": dict(edge.get("attributes", {})),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "prefix_through_step": prefix.get("prefix_through_step"),
        "nodes": nodes,
        "edges": edges,
    }


def scenario_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    cell = result.get("cell", {})
    if not isinstance(cell, Mapping):
        raise TypeError("Result cell must be a mapping.")
    return {
        "suite": result.get("suite"),
        "model": cell.get("model", result.get("model")),
        "task_id": cell.get("task_id", result.get("task_id")),
        "phase": cell.get("phase"),
        "attack": cell.get("attack"),
        "injection_task_id": cell.get("injection_task_id"),
    }


def build_run_examples(
    experiment_root: Path,
    run_directory: Path,
) -> list[dict[str, Any]]:
    result = read_json(run_directory / "result.json")
    contract = load_contract(run_directory / "contract.json")
    events = load_jsonl(run_directory / "monitor_trace.jsonl")
    execution_graph = load_graph(run_directory / "execution_graph.json")
    proposals = proposed_actions(events)
    labels = label_prefixes(contract, proposals)
    contract_graph = contract_to_graph(contract)
    scenario = scenario_payload(result)
    group_id = stable_hash(scenario, 20)
    cell = result.get("cell", {})
    examples: list[dict[str, Any]] = []

    for proposal, label in zip(proposals, labels, strict=True):
        step = int(proposal["step"])
        sanitized = sanitize_execution_prefix(
            execution_graph,
            through_step=step,
        )
        prefix_graph = canonicalize_execution_prefix(sanitized)
        identity = {
            "experiment": experiment_root.name,
            "run": run_directory.name,
            "step": step,
            "tool": proposal["tool"],
        }
        examples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "example_id": stable_hash(identity, 24),
                "group_id": group_id,
                "split": None,
                "source": {
                    "experiment_id": experiment_root.name,
                    "run_id": run_directory.name,
                    "model": scenario["model"],
                    "suite": scenario["suite"],
                    "task_id": scenario["task_id"],
                    "phase": scenario["phase"],
                    "attack": scenario["attack"],
                    "injection_task_id": scenario["injection_task_id"],
                    "trial": cell.get("trial"),
                    "step": step,
                    "contract_fingerprint": result.get(
                        "contract_fingerprint",
                        contract.get("contract_fingerprint"),
                    ),
                },
                "features": {
                    "contract_graph": contract_graph,
                    "execution_prefix": prefix_graph,
                    "candidate_action": {
                        "tool": proposal["tool"],
                        "arguments": proposal["arguments"],
                        "step": step,
                    },
                },
                "labels": {
                    "action_permitted": label.action.permitted,
                    "action_label": label.action.label,
                    "violation_rule": label.action.rule_id,
                    "trajectory_permitted": label.trajectory_permitted,
                    "trajectory_label": label.trajectory_label,
                },
            }
        )

    return examples


def discover_examples(
    roots: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    discovery: dict[str, Any] = {}

    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Experiment root not found: {root}")
        runs_root = root / "runs"
        if not runs_root.is_dir():
            raise FileNotFoundError(f"Runs directory not found: {runs_root}")

        run_count = 0
        eligible_count = 0
        root_examples = 0

        for run_directory in sorted(runs_root.iterdir()):
            if not run_directory.is_dir():
                continue
            run_count += 1
            required = (
                run_directory / "result.json",
                run_directory / "contract.json",
                run_directory / "monitor_trace.jsonl",
                run_directory / "execution_graph.json",
            )
            if not all(path.exists() for path in required):
                continue
            eligible_count += 1
            run_examples = build_run_examples(root, run_directory)
            examples.extend(run_examples)
            root_examples += len(run_examples)

        discovery[root.name] = {
            "path": str(root),
            "run_directories": run_count,
            "eligible_sentrygraph_runs": eligible_count,
            "examples": root_examples,
        }

    identifiers = [example["example_id"] for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate dataset example identifiers detected.")
    return examples, discovery


def allocation_counts(total: int) -> dict[str, int]:
    if total == 0:
        return {name: 0 for name in SPLIT_RATIOS}

    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())

    order = sorted(
        SPLIT_RATIOS,
        key=lambda name: (-(raw[name] - counts[name]), name),
    )
    for name in order[:remainder]:
        counts[name] += 1

    if total >= len(SPLIT_RATIOS):
        for empty_name in [name for name, count in counts.items() if count == 0]:
            donor = max(counts, key=lambda name: counts[name])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[empty_name] += 1

    assert sum(counts.values()) == total
    return counts


def assign_group_splits(
    examples: list[dict[str, Any]],
    seed: int,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["group_id"]].append(example)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, group_examples in grouped.items():
        has_violation = any(
            not example["labels"]["action_permitted"]
            for example in group_examples
        )
        strata["violation_present" if has_violation else "conformant_only"].append(
            group_id
        )

    assignments: dict[str, str] = {}
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(
            group_ids,
            key=lambda group_id: stable_hash(
                {"seed": seed, "stratum": stratum, "group_id": group_id}
            ),
        )
        counts = allocation_counts(len(ordered))
        offset = 0
        for split_name in SPLIT_RATIOS:
            for group_id in ordered[offset : offset + counts[split_name]]:
                assignments[group_id] = split_name
            offset += counts[split_name]

    if set(assignments) != set(grouped):
        raise RuntimeError("Not every scenario group received a split.")

    for example in examples:
        example["split"] = assignments[example["group_id"]]
    return assignments


def audit_dataset(examples: list[dict[str, Any]]) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for example in examples:
        split = example["split"]
        group_splits[example["group_id"]].add(split)
        labels = example["labels"]
        split_counts[split]["examples"] += 1
        split_counts[split][
            "action_allow" if labels["action_permitted"] else "action_deny"
        ] += 1
        split_counts[split][
            "trajectory_allow"
            if labels["trajectory_permitted"]
            else "trajectory_deny"
        ] += 1

        features = example["features"]
        execution_prefix = features["execution_prefix"]
        contract_graph = features["contract_graph"]

        if "trajectory_id" in execution_prefix:
            raise RuntimeError(
                f"Trajectory identifier leaked into {example['example_id']}."
            )

        for graph_name, graph in (
            ("execution_prefix", execution_prefix),
            ("contract_graph", contract_graph),
        ):
            for node in graph["nodes"]:
                if node["node_type"] == "policy_decision":
                    raise RuntimeError(
                        f"Policy-decision node leaked into {graph_name} "
                        f"for {example['example_id']}."
                    )

                attributes = node.get("attributes", {})

                # These keys are forbidden only on runtime action/decision
                # nodes. The same words may be legitimate task argument or
                # contract-constraint names and must not trigger rejection.
                if node["node_type"] in {
                    "proposed_action",
                    "policy_decision",
                }:
                    leaked_keys = {
                        "decision",
                        "decision_source",
                        "reason",
                        "rule_id",
                        "status",
                    } & set(attributes)
                    if leaked_keys:
                        raise RuntimeError(
                            f"Runtime policy fields {sorted(leaked_keys)} "
                            f"leaked into {example['example_id']}."
                        )

                if "timestamp" in attributes:
                    raise RuntimeError(
                        f"Timestamp leaked into {example['example_id']}."
                    )

            for edge in graph["edges"]:
                if edge["relation"] in {
                    "EVALUATED_AS",
                    "GOVERNS",
                    "BLOCKS",
                }:
                    raise RuntimeError(
                        f"Policy relation {edge['relation']} leaked into "
                        f"{example['example_id']}."
                    )
                if "timestamp" in edge:
                    raise RuntimeError(
                        f"Edge timestamp leaked into {example['example_id']}."
                    )

    leaking_groups = {
        group_id: sorted(splits)
        for group_id, splits in group_splits.items()
        if len(splits) != 1
    }
    if leaking_groups:
        raise RuntimeError(f"Scenario groups cross splits: {leaking_groups}")

    return {
        "examples": len(examples),
        "groups": len(group_splits),
        "split_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_counts.items())
        },
        "group_overlap": 0,
        "feature_leakage_findings": 0,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_dataset(
    output_dir: Path,
    examples: list[dict[str, Any]],
    discovery: Mapping[str, Any],
    assignments: Mapping[str, str],
    audit: Mapping[str, Any],
    seed: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}-",
            dir=output_dir.parent,
        )
    )

    try:
        ordered = sorted(examples, key=lambda example: example["example_id"])
        write_jsonl(temporary / "all_examples.jsonl", ordered)

        for split in SPLIT_RATIOS:
            write_jsonl(
                temporary / f"{split}.jsonl",
                [example for example in ordered if example["split"] == split],
            )

        group_manifest = [
            {"group_id": group_id, "split": split}
            for group_id, split in sorted(assignments.items())
        ]
        write_json(temporary / "group_assignments.json", group_manifest)

        schema = {
            "schema_version": SCHEMA_VERSION,
            "learning_instance": {
                "features": [
                    "contract_graph",
                    "execution_prefix",
                    "candidate_action",
                ],
                "targets": [
                    "action_permitted",
                    "action_label",
                    "violation_rule",
                    "trajectory_permitted",
                    "trajectory_label",
                ],
            },
            "split_unit": (
                "suite + model + task + phase + attack + injection task; "
                "trials remain in the same split"
            ),
            "excluded_feature_evidence": [
                "recorded policy decision",
                "recorded rule identifier",
                "recorded reason",
                "policy-decision nodes",
                "EVALUATED_AS edges",
                "GOVERNS edges",
                "BLOCKS edges",
                "current-step execution outcomes",
                "future nodes and edges",
                "timestamps and run identifiers",
            ],
        }
        write_json(temporary / "schema.json", schema)

        dataset_id = stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "assignments": assignments,
                "example_ids": [item["example_id"] for item in ordered],
            },
            20,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "seed": seed,
            "split_ratios": SPLIT_RATIOS,
            "source_discovery": discovery,
            "audit": audit,
            "files": {},
        }
        write_json(temporary / "manifest.json", manifest)

        data_files = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file()
            and path.name not in {"manifest.json", "SHA256SUMS.txt"}
        )
        checksums = {
            path.name: file_sha256(path)
            for path in data_files
        }
        manifest["files"] = checksums
        write_json(temporary / "manifest.json", manifest)

        # Recompute after the final manifest is written.
        data_files = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        checksum_lines = [
            f"{file_sha256(path)}  {path.name}"
            for path in data_files
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    examples, discovery = discover_examples(args.experiment_roots)

    if not examples:
        raise RuntimeError("No eligible SentryGraph proposals were found.")

    assignments = assign_group_splits(examples, args.seed)
    audit = audit_dataset(examples)
    materialize_dataset(
        args.output_dir,
        examples,
        discovery,
        assignments,
        audit,
        args.seed,
    )

    print("=== PHASE 1C GRAPH-CONFORMANCE DATASET ===")
    print("Dataset:", args.output_dir)
    print("Examples:", audit["examples"])
    print("Scenario groups:", audit["groups"])
    print("Group overlap:", audit["group_overlap"])
    print("Feature leakage findings:", audit["feature_leakage_findings"])
    print("Split counts:")
    for split, counts in audit["split_counts"].items():
        print(f"  {split}: {counts}")
    print("PASS: Dataset created atomically with grouped splits and checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
