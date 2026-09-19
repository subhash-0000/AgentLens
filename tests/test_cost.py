"""Self-checks for token pricing and aggregation."""

from agentlens.cost import calculate_cost, cost_by_step, total_cost_for_run
from agentlens.trace_schema import TraceEvent


def main() -> None:
    """Run known-model, unknown-model, and sum assertions."""
    assert calculate_cost("gpt-4", 1000, 500) > 0
    assert calculate_cost("some-made-up-model-xyz", 100, 100) == 0.0
    events = [TraceEvent("r", "t", "llm_call", "gpt-4", "", "", 1000, 500, 100, None), TraceEvent("r", "t", "llm_call", "gpt-4", "", "", 500, 250, 100, None)]
    assert total_cost_for_run(events) == calculate_cost("gpt-4", 1000, 500) + calculate_cost("gpt-4", 500, 250)
    assert calculate_cost("openai/gpt-oss-20b", 1000, 1000) > 0
    assert calculate_cost("qwen/qwen3.8-27b", 1000, 1000) > 0
    assert cost_by_step(events)[0][1] >= cost_by_step(events)[-1][1]
    print("ALL COST TESTS PASSED")


if __name__ == "__main__":
    main()
