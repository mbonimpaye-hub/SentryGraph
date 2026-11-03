"""Focused Phase 3C tests that require no paid model calls."""

from sentrygraph.contract_compiler import (
    ArgumentConstraint,
    ConstraintOperator,
    EffectClass,
    ToolAuthorization,
    compile_explicit_contract,
)
from sentrygraph.model_driven_multi_agent_runtime import (
    contract_graph,
    delegated_root_contract,
    prefix_graph,
)


def main() -> None:
    authorization = ToolAuthorization(
        tool="delete_file",
        effect=EffectClass.DELETE,
        constraints=(
            ArgumentConstraint(
                argument="file_id",
                operator=ConstraintOperator.EQUALS,
                value="11",
            ),
        ),
        allow_additional_arguments=False,
        justification="Test exact authority.",
    )
    original = compile_explicit_contract(
        principal="agentdojo-workspace-user",
        objective="Delete only file 11.",
        permitted_tools=(authorization,),
        prohibited_tools=frozenset({"send_email"}),
    )
    root = delegated_root_contract(original)
    assert root.principal == "orchestrator"
    assert root.delegation_allowed
    assert root.permitted_tools == original.permitted_tools
    assert EffectClass.DELEGATE not in root.prohibited_effects
    graph = contract_graph(original)
    relations = {edge["relation"] for edge in graph["edges"]}
    assert {"BOUND_BY", "PERMITS", "HAS_EFFECT", "CONSTRAINS", "PROHIBITS"} <= relations
    events = [
        {"event_id": "a", "step": 1, "phase": "proposed", "tool": "list_files"},
        {"event_id": "a", "step": 1, "phase": "completed", "tool": "list_files"},
        {"event_id": "b", "step": 2, "phase": "denied", "tool": "send_email"},
    ]
    prefix = prefix_graph(events, before_step=2)
    assert all(node.get("attributes", {}).get("step", 0) < 2 for node in prefix["nodes"])
    assert "BLOCKS" not in {edge["relation"] for edge in prefix["edges"]}
    print("PASS: Orchestrator authority preserves exact task constraints.")
    print("PASS: Learned features exclude the candidate and policy outcomes.")
    print("PASS: Phase 3C tests require no API calls.")


if __name__ == "__main__":
    main()

