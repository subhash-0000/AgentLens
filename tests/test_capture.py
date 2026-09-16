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
    for run_id in ("capture_test_001", "capture_test_002", "capture_test_003"):
        path = f"traces/{run_id}.jsonl"
        if os.path.exists(path):
            os.remove(path)
    print("ALL CAPTURE TESTS PASSED")


if __name__ == "__main__":
    main()
