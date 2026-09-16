"""End-to-end checks for happy, error, and LLM-disabled report scenarios."""

import os

from agentlens.capture import AgentLensCallback
from agentlens.report import generate_report
from agentlens.trace_schema import load_trace


def main() -> None:
    """Exercise clean capture, tool error capture, and narrative opt-out behavior."""
    os.makedirs("traces", exist_ok=True)
    happy = AgentLensCallback("integration_happy")
    happy.on_llm_start({"name": "gpt-4"}, ["hello"], run_id="llm")
    happy.on_llm_end(type("Response", (), {"llm_output": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}})(), run_id="llm")
    happy.on_tool_start({"name": "calculator"}, "2 + 3", run_id="tool")
    happy.on_tool_end("5", run_id="tool")
    assert len(load_trace("integration_happy")) == 2
    assert "$" in generate_report("integration_happy", use_llm=False)

    broken = AgentLensCallback("integration_error")
    broken.on_tool_start({"name": "broken_tool"}, "bad input", run_id="bad-tool")
    broken.on_chain_error(RuntimeError("tool failed"), run_id="bad-tool")
    error_report = generate_report("integration_error", use_llm=False)
    assert "tool failed" in error_report and "Anomalies" in error_report

    os.environ.pop("ANTHROPIC_API_KEY", None)
    assert "What Happened" not in generate_report("integration_happy", use_llm=False)
    for run_id in ("integration_happy", "integration_error"):
        os.remove(f"traces/{run_id}.jsonl")
    print("ALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
