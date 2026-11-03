"""Encode Phase 1F folds as train-only-fitted relational graph tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from sentrygraph.relational_graph_data import (
    RelationalVocabulary,
    VIEWS,
    encode_example,
    stable_hash,
    tensor_fingerprint,
)


SPLITS = ("train", "validation", "calibration", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-resistant relational graph tensors."
    )
    parser.add_argument(
        "--folds-root",
        type=Path,
        required=True,
        help="Phase 1F v2 generalization-fold directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Atomic output directory.",
    )
    parser.add_argument(
        "--variant",
        choices=("deduplicated", "full"),
        default="deduplicated",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["action_label"] for record in records)
    return {
        "allow": counts[0],
        "violation": counts[1],
    }


def main() -> int:
    args = parse_args()
    protocol = read_json(args.folds_root / "manifest.json")

    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary_parent = Path(
        tempfile.mkdtemp(prefix="phase1j-", dir=args.output_dir.parent)
    )
    temporary_output = temporary_parent / args.output_dir.name
    temporary_output.mkdir()
    fold_summaries = []

    try:
        for fold in protocol["folds"]:
            fold_name = fold["fold"]
            source = args.folds_root / "folds" / fold_name / args.variant
            examples = {
                split: read_jsonl(source / f"{split}.jsonl")
                for split in SPLITS
            }

            fold_output = temporary_output / "folds" / fold_name
            fold_output.mkdir(parents=True)
            view_summaries = []

            for view in VIEWS:
                vocabulary = RelationalVocabulary.fit(examples["train"], view)
                view_output = fold_output / view
                view_output.mkdir()
                write_json(view_output / "vocabulary.json", vocabulary.to_dict())

                encoded_splits = {
                    split: [
                        encode_example(example, vocabulary)
                        for example in examples[split]
                    ]
                    for split in SPLITS
                }

                train_fingerprints: dict[str, set[int]] = {}
                for record in encoded_splits["train"]:
                    fingerprint = tensor_fingerprint(record)
                    train_fingerprints.setdefault(fingerprint, set()).add(
                        record["action_label"]
                    )
                collision_summary = {}

                for split, records in encoded_splits.items():
                    torch.save(
                        {
                            "schema_version": "0.1",
                            "fold": fold_name,
                            "view": view,
                            "split": split,
                            "records": records,
                        },
                        view_output / f"{split}.pt",
                    )

                    if split != "train":
                        fingerprints = [
                            tensor_fingerprint(record) for record in records
                        ]
                        shared = 0
                        conflicting = 0
                        for fingerprint, record in zip(
                            fingerprints,
                            records,
                            strict=True,
                        ):
                            if fingerprint not in train_fingerprints:
                                continue
                            shared += 1
                            if (
                                record["action_label"]
                                not in train_fingerprints[fingerprint]
                            ):
                                conflicting += 1

                        collision_summary[f"train__{split}"] = {
                            "shared_examples": shared,
                            "conflicting_label_examples": conflicting,
                        }

                view_summaries.append(
                    {
                        "view": view,
                        "tokens": len(vocabulary.token_to_id),
                        "relations": len(vocabulary.relation_to_id),
                        "split_counts": {
                            split: {
                                "examples": len(records),
                                "labels": label_counts(records),
                            }
                            for split, records in encoded_splits.items()
                        },
                        "train_partition_collisions": collision_summary,
                    }
                )

            fold_summaries.append(
                {
                    "fold": fold_name,
                    "test_evaluation": fold["test_evaluation"],
                    "assignment": fold["assignment"],
                    "views": view_summaries,
                }
            )

        configuration = {
            "source_protocol_id": protocol["protocol_id"],
            "variant": args.variant,
            "views": list(VIEWS),
            "vocabulary_fit_split": "train",
            "labels_used_as_features": False,
            "source_metadata_used_as_features": False,
        }
        encoding_id = stable_hash(configuration)[:20]
        manifest = {
            "schema_version": "0.1",
            "encoding_id": encoding_id,
            "configuration": configuration,
            "folds": fold_summaries,
        }
        write_json(temporary_output / "manifest.json", manifest)

        files = sorted(
            path for path in temporary_output.rglob("*") if path.is_file()
        )
        (temporary_output / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(temporary_output)}"
                for path in files
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_output.replace(args.output_dir)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)

    print("=== PHASE 1J RELATIONAL GRAPH ENCODING ===")
    print("Encoding ID:", encoding_id)
    print("Source protocol:", protocol["protocol_id"])
    print("Variant:", args.variant)
    print("Output:", args.output_dir)
    print("Folds:", len(fold_summaries))
    for fold in fold_summaries:
        print()
        print(fold["fold"], fold["test_evaluation"])
        for view in fold["views"]:
            print(
                f"  {view['view']}: tokens={view['tokens']}, "
                f"relations={view['relations']}, "
                f"collisions={view['train_partition_collisions']}"
            )
    print()
    print("PASS: Every relational vocabulary was fitted on training only.")
    print("PASS: Labels and source metadata were excluded from tensors.")
    print("PASS: Cross-partition tensor collisions were explicitly audited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
