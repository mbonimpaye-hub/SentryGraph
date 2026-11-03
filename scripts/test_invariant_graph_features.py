"""Regression tests for task-invariant contract-conformance features."""

from __future__ import annotations

import copy

from sentrygraph.graph_features import example_to_feature_dict


def example(tool_name: str = "delete_file") -> dict:
    return {
        "features": {
            "candidate_action": {
                "tool": tool_name,
                "arguments": {"file_id": "11"},
                "step": 2,
            },
            "contract_graph": {
                "nodes": [
                    {
                        "node_id": "tool:0",
                        "node_type": "tool",
                        "attributes": {
                            "name": tool_name,
                            "authorization": "permitted",
                            "allow_additional_arguments": False,
                            "approval_required": False,
                        },
                    },
                    {
                        "node_id": "constraint:0",
                        "node_type": "argument_constraint",
                        "attributes": {
                            "argument": "file_id",
                            "operator": "equals",
                            "value": "11",
                        },
                    },
                ],
                "edges": [
                    {
                        "source": "tool:0",
                        "target": "constraint:0",
                        "relation": "CONSTRAINS",
                    }
                ],
            },
            "execution_prefix": {"nodes": [], "edges": []},
        }
    }


def main() -> None:
    allowed = example_to_feature_dict(example())
    assert allowed["candidate.constraint_satisfied_count"] == 1.0
    assert allowed["candidate.constraint_violated_count"] == 0.0
    assert allowed["candidate.invariant_contract_conformant"] == 1.0

    wrong_value = example()
    wrong_value["features"]["candidate_action"]["arguments"]["file_id"] = "13"
    denied = example_to_feature_dict(wrong_value)
    assert denied["candidate.constraint_violated_count"] == 1.0
    assert denied["candidate.invariant_contract_conformant"] == 0.0

    null_optional = example()
    null_optional["features"]["candidate_action"]["arguments"]["location"] = None
    normalized = example_to_feature_dict(null_optional)
    assert normalized["candidate.unexpected_non_null_argument_count"] == 0.0
    assert normalized["candidate.invariant_contract_conformant"] == 1.0

    unexpected = copy.deepcopy(null_optional)
    unexpected["features"]["candidate_action"]["arguments"]["location"] = "room"
    rejected = example_to_feature_dict(unexpected)
    assert rejected["candidate.unexpected_non_null_argument_count"] == 1.0
    assert rejected["candidate.invariant_contract_conformant"] == 0.0

    renamed = example_to_feature_dict(example("erase_resource"))
    invariant_keys = {
        key: value
        for key, value in allowed.items()
        if "tool" not in key and "value" not in key
    }
    renamed_invariant = {
        key: value
        for key, value in renamed.items()
        if "tool" not in key and "value" not in key
    }
    assert invariant_keys == renamed_invariant
    print("PASS: Exact constraint satisfaction is represented explicitly.")
    print("PASS: Unexpected non-null arguments fail invariant conformance.")
    print("PASS: Optional null arguments remain normalized as absent.")
    print("PASS: Invariant signals survive tool renaming.")


if __name__ == "__main__":
    main()

