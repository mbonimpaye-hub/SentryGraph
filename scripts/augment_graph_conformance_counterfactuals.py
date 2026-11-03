"""Create train-only counterfactual contracts for task-held-out learning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROLES = ("train", "validation", "calibration", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("deduplicated", "full"), default="deduplicated"
    )
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def identifier(value: Any, length: int = 24) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def matching_tools(example: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = example["features"]["candidate_action"]["tool"]
    return [
        node
        for node in example["features"]["contract_graph"]["nodes"]
        if node.get("node_type") == "tool"
        and node.get("attributes", {}).get("name") == candidate
    ]


def set_labels(example: dict[str, Any], permitted: bool, rule: str) -> None:
    example["labels"] = {
        **example.get("labels", {}),
        "action_permitted": permitted,
        "action_label": "conformant" if permitted else "violation",
        "trajectory_permitted": permitted,
        "trajectory_label": (
            "conformant_prefix" if permitted else "violating_prefix"
        ),
        "violation_rule": rule,
    }


def finalize(
    original: dict[str, Any],
    synthetic: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    payload = {
        "parent": original["example_id"],
        "kind": kind,
        "features": synthetic["features"],
        "labels": synthetic["labels"],
    }
    synthetic["example_id"] = identifier(payload)
    synthetic["group_id"] = "counterfactual-" + identifier(payload, 16)
    synthetic["source"] = {
        **original.get("source", {}),
        "synthetic": True,
        "counterfactual_kind": kind,
        "parent_example_id": original["example_id"],
    }
    synthetic["split"] = "train"
    return synthetic


def authorization_flip(example: dict[str, Any]) -> dict[str, Any] | None:
    synthetic = copy.deepcopy(example)
    tools = matching_tools(synthetic)
    if len(tools) != 1:
        return None
    tool = tools[0]
    attributes = tool.setdefault("attributes", {})
    current = attributes.get("authorization")
    if current not in {"permitted", "prohibited"}:
        return None
    replacement = "prohibited" if current == "permitted" else "permitted"
    attributes["authorization"] = replacement
    graph = synthetic["features"]["contract_graph"]
    tool_id = tool["node_id"]
    for edge in graph["edges"]:
        if edge.get("target") != tool_id:
            continue
        if edge.get("relation") == "PERMITS":
            edge["relation"] = "PROHIBITS"
        elif edge.get("relation") == "PROHIBITS":
            edge["relation"] = "PERMITS"
    permitted = replacement == "permitted"
    set_labels(
        synthetic,
        permitted,
        "COUNTERFACTUAL_CONTRACT_ALLOW" if permitted else "COUNTERFACTUAL_PROHIBIT",
    )
    return finalize(example, synthetic, "authorization_flip")


def constraint_violation(example: dict[str, Any]) -> dict[str, Any] | None:
    synthetic = copy.deepcopy(example)
    tools = matching_tools(synthetic)
    if len(tools) != 1:
        return None
    tool = tools[0]
    if tool.get("attributes", {}).get("authorization") != "permitted":
        return None
    graph = synthetic["features"]["contract_graph"]
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    constraints = [
        nodes.get(edge.get("target"))
        for edge in graph["edges"]
        if edge.get("source") == tool["node_id"]
        and edge.get("relation") == "CONSTRAINS"
    ]
    constraints = [item for item in constraints if item is not None]
    if not constraints:
        return None
    constraint = constraints[0]["attributes"]
    argument = str(constraint["argument"])
    expected = constraint.get("value")
    replacement: Any
    if isinstance(expected, str):
        replacement = expected + "__counterfactual_violation__"
    elif isinstance(expected, bool):
        replacement = not expected
    elif isinstance(expected, (int, float)):
        replacement = expected + 1
    elif isinstance(expected, list):
        replacement = ["__counterfactual_violation__"]
    else:
        replacement = "__counterfactual_violation__"
    synthetic["features"]["candidate_action"].setdefault("arguments", {})[
        argument
    ] = replacement
    set_labels(synthetic, False, "COUNTERFACTUAL_ARGUMENT_CONSTRAINT_VIOLATION")
    return finalize(example, synthetic, "constraint_violation")


def unexpected_argument_violation(
    example: dict[str, Any],
) -> dict[str, Any] | None:
    synthetic = copy.deepcopy(example)
    tools = matching_tools(synthetic)
    if len(tools) != 1:
        return None
    attributes = tools[0].get("attributes", {})
    if attributes.get("authorization") != "permitted" or attributes.get(
        "allow_additional_arguments", True
    ):
        return None
    arguments = synthetic["features"]["candidate_action"].setdefault(
        "arguments", {}
    )
    arguments["__counterfactual_unexpected_argument__"] = "non-null"
    set_labels(synthetic, False, "COUNTERFACTUAL_ADDITIONAL_ARGUMENT_VIOLATION")
    return finalize(example, synthetic, "unexpected_argument_violation")


def augment_train(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = list(rows)
    counts = {
        "original": len(rows),
        "authorization_flip": 0,
        "constraint_violation": 0,
        "unexpected_argument_violation": 0,
    }
    seen = {identifier(row["features"], 64) for row in rows}
    builders = (
        ("authorization_flip", authorization_flip),
        ("constraint_violation", constraint_violation),
        ("unexpected_argument_violation", unexpected_argument_violation),
    )
    for row in rows:
        for name, builder in builders:
            synthetic = builder(row)
            if synthetic is None:
                continue
            digest = identifier(synthetic["features"], 64)
            if digest in seen:
                continue
            seen.add(digest)
            output.append(synthetic)
            counts[name] += 1
    counts["augmented_total"] = len(output)
    return output, counts


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    manifest = json.loads(
        (args.protocol_root / "manifest.json").read_text(encoding="utf-8")
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent
        )
    )
    audit: dict[str, Any] = {}
    try:
        for fold in manifest["folds"]:
            name = fold["fold"]
            source = args.protocol_root / "folds" / name / args.variant
            destination = temporary / "folds" / name / args.variant
            audit[name] = {}
            for role in ROLES:
                source_path = source / f"{role}.jsonl"
                destination_path = destination / f"{role}.jsonl"
                rows = load_jsonl(source_path)
                if role == "train":
                    rows, counts = augment_train(rows)
                    audit[name] = counts
                    write_jsonl(destination_path, rows)
                else:
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                    if file_sha256(source_path) != file_sha256(destination_path):
                        raise RuntimeError(
                            f"Non-training partition changed: {name}/{role}"
                        )
        output_manifest = {
            **manifest,
            "schema_version": "0.3",
            "parent_protocol_id": manifest.get("protocol_id"),
            "augmentation": {
                "name": "train_only_contract_counterfactuals",
                "test_observed_for_selection": False,
                "validation_observed_for_generation": False,
                "calibration_observed_for_generation": False,
                "audit": audit,
            },
        }
        output_manifest["protocol_id"] = identifier(output_manifest, 20)
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary)
            checksum_lines.append(f"{file_sha256(path)}  {relative}\n")
        (temporary / "SHA256SUMS.txt").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("=== PHASE 3D TRAIN-ONLY COUNTERFACTUAL AUGMENTATION ===")
    print("Protocol ID:", output_manifest["protocol_id"])
    print("Parent protocol:", output_manifest["parent_protocol_id"])
    print("Output:", args.output_dir)
    for fold, counts in sorted(audit.items()):
        print(fold, counts)
    print("PASS: Only training partitions were augmented.")
    print("PASS: Validation, calibration, and test records were copied unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

