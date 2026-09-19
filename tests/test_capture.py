"""Self-checks for the LangChain callback handler."""

import os

from agentlens.capture import AgentLensCallback
from agentlens.trace_schema import load_trace


def main() -> None:
    """Run callback persistence, truncation, and malformed-input assertions."""
    callback = AgentLensCallback("capture_test_001", "traces")
    callback.on_llm_start({"name": "test-model"}, ["hello world"], run_id="llm-1")
    callback.on_llm_end(type("Response", (), {"llm_output": {"token_usage": {"total_tokens": 10}}})(), run_id="llm-1")
    assert len(load_trace("capture_test_001")) >= 1
    callback = AgentLensCallback("capture_test_002", "traces")
    callback.on_llm_start({"name": "test"}, ["x" * 1000], run_id="llm-2")
    callback.on_llm_end(None, run_id="llm-2")
    assert all(len(event.input_summary) <= 200 for event in load_trace("capture_test_002"))
    callback = AgentLensCallback("capture_test_003", "traces")
    callback.on_llm_end(None, run_id="capture_test_003")
    assert callback._llm_name({"name": "ChatGroq", "kwargs": {"model_name": "openai/gpt-oss-20b"}}, {}) == "openai/gpt-oss-20b"
    assert callback._llm_name({"name": "ChatGroq"}, {"metadata": {"ls_model_name": "qwen/qwen3.8-27b"}}) == "qwen/qwen3.8-27b"
    callback.on_chain_start(None, "", run_id="chain-1", name="QuestionGenerator")
    callback.on_chain_error(RuntimeError("chain failed"), run_id="chain-1")
    assert load_trace("capture_test_003")[-1].name == "QuestionGenerator"
    for run_id in ("capture_test_001", "capture_test_002", "capture_test_003"):
        path = f"traces/{run_id}.jsonl"
        if os.path.exists(path):
            os.remove(path)
    print("ALL CAPTURE TESTS PASSED")


if __name__ == "__main__":
    main()
