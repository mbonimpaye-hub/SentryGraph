#!/usr/bin/env python3
"""Encode task-held-out SentryGraph folds under three feature views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sentrygraph.graph_features import example_to_feature_dict


ROLES = ("train", "validation", "calibration", "test")
VIEWS = (
    "full_fidelity",
    "structural_generalization",
    "semantic_ablated",
    "invariant_relational",
)

STRUCTURAL_FORBIDDEN = (
    "candidate.contract_authorization::",
    "candidate.matching_contract_tool_count",
)
SEMANTIC_FORBIDDEN = STRUCTURAL_FORBIDDEN + (
    "candidate.constraint.expected_value::",
    "candidate.constraint.observed_value::",
)

INVARIANT_ALLOWED = (
    "candidate.argument_count",
    "candidate.null_argument_count",
    "candidate.contract_authorization::",
    "candidate.approval_required",
    "candidate.allow_additional_arguments",
    "candidate.matching_contract_tool_count",
    "candidate.constraint_count",
    "candidate.constraint_satisfied_count",
    "candidate.constraint_violated_count",
    "candidate.constraint_missing_count",
    "candidate.all_constraints_satisfied",
    "candidate.constraint.satisfied::",
    "candidate.constraint.violated::",
    "candidate.unexpected_argument_count",
    "candidate.unexpected_non_null_argument_count",
    "candidate.argument_surface_valid",
    "candidate.invariant_contract_conformant",
    "contract.node_type::",
    "contract.relation::",
    "prefix.node_count",
    "prefix.edge_count",
    "prefix.node_type::",
    "prefix.relation::",
    "prefix.completed_actions",
    "prefix.same_tool_actions",
    "prefix.resource_kind::",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode task-held-out graph-conformance folds."
    )
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("deduplicated", "full"),
        default="deduplicated",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"Empty partition: {path}")
    return records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def view_forbidden(view: str) -> tuple[str, ...]:
    if view == "full_fidelity":
        return ()
    if view == "structural_generalization":
        return STRUCTURAL_FORBIDDEN
    if view == "semantic_ablated":
        return SEMANTIC_FORBIDDEN
    if view == "invariant_relational":
        return ()
    raise ValueError(view)


def filtered_feature_dict(
    example: Mapping[str, Any],
    view: str,
) -> dict[str, Any]:
    extracted = example_to_feature_dict(example)
    if view == "invariant_relational":
        return {
            key: value
            for key, value in extracted.items()
            if any(
                key == prefix or key.startswith(prefix)
                for prefix in INVARIANT_ALLOWED
            )
        }
    forbidden = view_forbidden(view)
    return {
        key: value
        for key, value in extracted.items()
        if not any(
            key == prefix or key.startswith(prefix)
            for prefix in forbidden
        )
    }


def assert_label_independence(example: dict[str, Any], view: str) -> None:
    original = filtered_feature_dict(example, view)
    probe = dict(example)
    probe["labels"] = {
        "action_permitted": not example["labels"]["action_permitted"],
        "trajectory_permitted": not example["labels"]["trajectory_permitted"],
        "secret": "must_not_appear",
    }
    probe["source"] = {"task_id": "must_not_appear"}
    probe["split"] = "must_not_appear"
    probe["group_id"] = "must_not_appear"
    if filtered_feature_dict(probe, view) != original:
        raise RuntimeError(f"{view} observes labels or source metadata")


def labels(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [int(record["labels"][key]) for record in records],
        dtype=np.int64,
    )


def matrix_digest(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def encoded_overlap(
    matrices: dict[str, np.ndarray],
    action_labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left_index, left in enumerate(ROLES):
        left_map: dict[str, set[int]] = {}
        for row, label in zip(matrices[left], action_labels[left], strict=True):
            left_map.setdefault(matrix_digest(row), set()).add(int(label))
        for right in ROLES[left_index + 1:]:
            right_map: dict[str, set[int]] = {}
            for row, label in zip(
                matrices[right], action_labels[right], strict=True
            ):
                right_map.setdefault(matrix_digest(row), set()).add(int(label))
            shared = set(left_map) & set(right_map)
            conflicting = sum(
                len(left_map[item] | right_map[item]) > 1
                for item in shared
            )
            result[f"{left}__{right}"] = {
                "shared_unique_vectors": len(shared),
                "conflicting_label_vectors": conflicting,
                "left_rows_matching_right": sum(
                    matrix_digest(row) in right_map for row in matrices[left]
                ),
                "right_rows_matching_left": sum(
                    matrix_digest(row) in left_map for row in matrices[right]
                ),
            }
    return result


def encode_view(
    records: dict[str, list[dict[str, Any]]],
    view: str,
    output: Path,
) -> dict[str, Any]:
    assert_label_independence(records["train"][0], view)
    train_dicts = [filtered_feature_dict(item, view) for item in records["train"]]
    vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
    vectorizer.fit(train_dicts)

    matrices = {
        role: np.asarray(
            vectorizer.transform(
                [filtered_feature_dict(item, view) for item in records[role]]
            ),
            dtype=np.float32,
        )
        for role in ROLES
    }
    names = list(vectorizer.get_feature_names_out())
    forbidden = view_forbidden(view)
    leaked_names = [
        name for name in names
        if any(name == prefix or name.startswith(prefix) for prefix in forbidden)
    ]
    if leaked_names:
        raise RuntimeError(f"Forbidden features survived in {view}: {leaked_names}")
    if len(names) == 0:
        raise RuntimeError(f"No features remain for {view}")
    if any(matrix.shape[1] != len(names) for matrix in matrices.values()):
        raise RuntimeError(f"Feature dimension mismatch in {view}")

    action = {role: labels(records[role], "action_permitted") for role in ROLES}
    trajectory = {
        role: labels(records[role], "trajectory_permitted") for role in ROLES
    }

    output.mkdir(parents=True)
    write_json(output / "feature_names.json", names)
    write_json(
        output / "vocabulary.json",
        dict(sorted(vectorizer.vocabulary_.items())),
    )
    split_summary: dict[str, Any] = {}
    for role in ROLES:
        matrix = matrices[role]
        np.save(output / f"X_{role}.npy", matrix, allow_pickle=False)
        np.save(output / f"y_action_{role}.npy", action[role], allow_pickle=False)
        np.save(
            output / f"y_trajectory_{role}.npy",
            trajectory[role],
            allow_pickle=False,
        )
        write_jsonl(
            output / f"metadata_{role}.jsonl",
            [
                {
                    "example_id": item["example_id"],
                    "group_id": item["group_id"],
                    "source": item["source"],
                }
                for item in records[role]
            ],
        )
        nonzero = int(np.count_nonzero(matrix))
        split_summary[role] = {
            "examples": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "nonzero_values": nonzero,
            "density": float(nonzero / matrix.size),
            "action_allow": int(action[role].sum()),
            "action_deny": int((1 - action[role]).sum()),
            "trajectory_allow": int(trajectory[role].sum()),
            "trajectory_deny": int((1 - trajectory[role]).sum()),
        }

    return {
        "view": view,
        "fit_split": "train",
        "feature_count": len(names),
        "removed_feature_prefixes": list(forbidden),
        "label_independence_check": True,
        "forbidden_feature_check": True,
        "split_summary": split_summary,
        "encoded_vector_overlap": encoded_overlap(matrices, action),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    protocol = json.loads(
        (args.protocol_root / "manifest.json").read_text(encoding="utf-8")
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}-",
            dir=args.output_dir.parent,
        )
    )
    fold_results: list[dict[str, Any]] = []
    try:
        for fold in protocol["folds"]:
            fold_name = fold["fold"]
            source = args.protocol_root / "folds" / fold_name / args.variant
            records = {
                role: load_jsonl(source / f"{role}.jsonl") for role in ROLES
            }
            view_results = []
            for view in VIEWS:
                destination = temporary / "folds" / fold_name / view
                result = encode_view(records, view, destination)
                write_json(destination / "manifest.json", result)
                view_results.append(result)
            fold_result = {
                "fold": fold_name,
                "assignment": fold["assignment"],
                "test_evaluation": fold["test_evaluation"],
                "views": view_results,
            }
            write_json(temporary / "folds" / fold_name / "manifest.json", fold_result)
            fold_results.append(fold_result)

        manifest = {
            "schema_version": "0.1",
            "encoder": "TaskHeldOutGraphFeatureEncoder/0.1",
            "source_protocol_id": protocol["protocol_id"],
            "source_protocol_root": str(args.protocol_root),
            "source_variant": args.variant,
            "views": list(VIEWS),
            "folds": fold_results,
            "fit_policy": "independent train-only vocabulary per fold and view",
        }
        manifest["encoding_id"] = hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest()[:20]
        write_json(temporary / "manifest.json", manifest)

        payloads = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary)}"
                for path in payloads
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 1G TASK-HELD-OUT FEATURE ENCODING ===")
    print("Protocol:", protocol["protocol_id"])
    print("Encoding:", manifest["encoding_id"])
    print("Variant:", args.variant)
    print("Output:", args.output_dir)
    for fold in fold_results:
        print(f"\n{fold['fold']} ({fold['test_evaluation']})")
        for view in fold["views"]:
            test = view["split_summary"]["test"]
            overlap = view["encoded_vector_overlap"]["train__test"]
            print(
                f"  {view['view']}: features={view['feature_count']}, "
                f"test={test['examples']}, "
                f"train-test-shared-vectors={overlap['shared_unique_vectors']}, "
                f"conflicting={overlap['conflicting_label_vectors']}"
            )
    print("PASS: Every encoder was fitted on its fold's training partition only.")
    print("PASS: Feature views passed label and forbidden-feature audits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
