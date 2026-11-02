"""Encode frozen graph-conformance splits as deterministic dense matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from sentrygraph.graph_features import GraphFeatureEncoder, example_to_feature_dict


SPLITS = ("train", "validation", "calibration", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")

    source_manifest = json.loads(
        (args.dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    records = {
        split: load_jsonl(args.dataset_root / f"{split}.jsonl")
        for split in SPLITS
    }

    # Explicitly verify that feature extraction cannot observe labels or source.
    probe = records["train"][0]
    original_features = example_to_feature_dict(probe)
    adversarial_probe = dict(probe)
    adversarial_probe["labels"] = {
        "action_permitted": not probe["labels"]["action_permitted"],
        "arbitrary_secret_label": "must_not_appear",
    }
    adversarial_probe["source"] = {"task_id": "must_not_appear"}
    if example_to_feature_dict(adversarial_probe) != original_features:
        raise RuntimeError("Feature extraction depends on labels or source metadata.")

    encoder = GraphFeatureEncoder().fit(records["train"])
    matrices = {split: encoder.transform(records[split]) for split in SPLITS}

    feature_count = len(encoder.feature_names)
    if any(matrix.shape[1] != feature_count for matrix in matrices.values()):
        raise RuntimeError("Feature dimensions differ across splits.")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )

    try:
        write_json(temporary / "feature_names.json", encoder.feature_names)
        write_json(temporary / "vocabulary.json", encoder.vocabulary)

        split_summary: dict[str, Any] = {}
        for split in SPLITS:
            matrix = matrices[split]
            examples = records[split]
            action_labels = np.asarray(
                [int(item["labels"]["action_permitted"]) for item in examples],
                dtype=np.int64,
            )
            trajectory_labels = np.asarray(
                [int(item["labels"]["trajectory_permitted"]) for item in examples],
                dtype=np.int64,
            )
            np.save(temporary / f"X_{split}.npy", matrix, allow_pickle=False)
            np.save(
                temporary / f"y_action_{split}.npy",
                action_labels,
                allow_pickle=False,
            )
            np.save(
                temporary / f"y_trajectory_{split}.npy",
                trajectory_labels,
                allow_pickle=False,
            )
            write_jsonl(
                temporary / f"metadata_{split}.jsonl",
                [
                    {
                        "example_id": item["example_id"],
                        "group_id": item["group_id"],
                        "source": item["source"],
                    }
                    for item in examples
                ],
            )
            split_summary[split] = {
                "examples": int(matrix.shape[0]),
                "features": int(matrix.shape[1]),
                "nonzero_values": int(np.count_nonzero(matrix)),
                "density": float(np.count_nonzero(matrix) / matrix.size),
                "action_permitted": int(action_labels.sum()),
                "action_violations": int((1 - action_labels).sum()),
                "trajectory_permitted": int(trajectory_labels.sum()),
                "trajectory_violations": int((1 - trajectory_labels).sum()),
            }

        manifest = {
            "schema_version": "0.1",
            "source_dataset_id": source_manifest["dataset_id"],
            "encoder": "GraphFeatureEncoder/0.1",
            "fit_split": "train",
            "feature_count": feature_count,
            "label_independence_check": True,
            "split_summary": split_summary,
            "files": {},
        }
        write_json(temporary / "manifest.json", manifest)

        payload_files = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "manifest.json"
        )
        manifest["files"] = {path.name: sha256(path) for path in payload_files}
        write_json(temporary / "manifest.json", manifest)

        checksum_files = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_files)
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print("=== PHASE 1D GRAPH FEATURE ENCODING ===")
    print("Source dataset:", source_manifest["dataset_id"])
    print("Output:", args.output_dir)
    print("Features:", feature_count)
    for split, summary in split_summary.items():
        print(f"  {split}: {summary}")
    print("PASS: Encoder was fitted on training data only.")
    print("PASS: Feature extraction is independent of labels and source metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
