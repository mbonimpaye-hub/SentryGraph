"""Unit tests for Phase 1J relational graph tensorization."""

from __future__ import annotations

from copy import deepcopy

from sentrygraph.relational_graph_data import (
    RelationalVocabulary,
    VIEWS,
    encode_example,
    tensor_fingerprint,
)


def example() -> dict:
    return {
        "example_id": "example-a",
        "group_id": "group-a",
        "split": "train",
        "source": {
            "task_id": "task-secret",
            "contract_fingerprint": "fingerprint-secret",
        },
        "labels": {
            "action_permitted": False,
            "trajectory_permitted": False,
        },
        "features": {
            "candidate_action": {
                "tool": "delete_file",
                "arguments": {"file_id": "13"},
            },
            "contract_graph": {
                "nodes": [
                    {
                        "id": "contract:0",
                        "type": "task_contract",
                        "attributes": {"compiler": "explicit_policy"},
                    },
                    {
                        "id": "tool:0",
                        "type": "tool",
                        "attributes": {"name": "delete_file"},
                    },
                ],
                "edges": [
                    {
                        "source": "contract:0",
                        "target": "tool:0",
                        "relation": "PERMITS",
                    }
                ],
            },
            "execution_prefix": {"nodes": [], "edges": []},
        },
    }


def main() -> None:
    original = example()

    for view in VIEWS:
        vocabulary = RelationalVocabulary.fit([original], view)
        encoded = encode_example(original, vocabulary)

        changed = deepcopy(original)
        changed["example_id"] = "different-example"
        changed["group_id"] = "different-group"
        changed["split"] = "test"
        changed["source"]["task_id"] = "different-task"
        changed["source"]["contract_fingerprint"] = "different-contract"
        changed["labels"]["action_permitted"] = True
        changed["labels"]["trajectory_permitted"] = True

        changed_encoded = encode_example(changed, vocabulary)

        assert tensor_fingerprint(encoded) == tensor_fingerprint(
            changed_encoded
        )
        assert encoded["action_label"] != changed_encoded["action_label"]
        assert encoded["trajectory_label"] != changed_encoded[
            "trajectory_label"
        ]

        unseen = deepcopy(original)
        unseen["features"]["candidate_action"]["tool"] = "unseen_tool"
        unseen_encoded = encode_example(unseen, vocabulary)
        assert unseen_encoded["num_nodes"] == encoded["num_nodes"]

    print("PASS: Labels and source metadata cannot affect graph tensors.")
    print("PASS: Unseen held-out symbols map through train-only vocabularies.")


if __name__ == "__main__":
    main()

