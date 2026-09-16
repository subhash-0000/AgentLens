"""Self-checks for trace schema behavior."""

from agentlens.trace_schema import TraceEvent, from_jsonl_line, load_trace, to_jsonl_line


def main() -> None:
    """Run schema round-trip, missing-file, and truncation assertions."""
    event = TraceEvent("test123", "2026-01-01T00:00:00", "llm_call", "gpt-4", "hello", "hi there", 5, 3, 200, None)
    restored = from_jsonl_line(to_jsonl_line(event))
    assert restored.run_id == event.run_id
    assert restored.tokens_in == 5
    assert load_trace("nonexistent_run_id_xyz") == []
    long_event = TraceEvent("t", "x", "tool_call", "search", "x" * 500, "y" * 500, None, None, None, None)
    assert len(long_event.input_summary) <= 200
    print("ALL TRACE_SCHEMA TESTS PASSED")


if __name__ == "__main__":
    main()
