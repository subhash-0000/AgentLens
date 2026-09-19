"""Self-checks for template report generation."""

import os

from agentlens.report import generate_report
from agentlens.trace_schema import TraceEvent, to_jsonl_line


def main() -> None:
    """Write a fixture trace and assert summary, timeline, anomaly, and missing-run output."""
    os.makedirs("traces", exist_ok=True)
    run_id = "report_test_001"
    events = [TraceEvent(run_id, "2026-01-01T00:00:00", "llm_call", "gpt-4", "plan", "plan", 200, 100, 800, None), TraceEvent(run_id, "2026-01-01T00:00:02", "tool_call", "web_search", "query", "results", None, None, 6000, None)]
    with open(f"traces/{run_id}.jsonl", "w", encoding="utf-8") as trace_file:
        for event in events:
            trace_file.write(to_jsonl_line(event) + "\n")
    report = generate_report(run_id, use_llm=False)
    assert "Summary" in report
    assert "Step-by-step" in report or "timeline" in report.lower()
    assert "Anomal" in report and "web_search" in report
    assert "no trace" in generate_report("run_id_that_does_not_exist", use_llm=False).lower()
    os.remove(f"traces/{run_id}.jsonl")
    close_costs = [0.001, 0.0012, 0.0009, 0.0011]
    lopsided_costs = [0.001, 0.001, 0.001, 0.005]
    for name, costs, expected in (("close", close_costs, 0), ("lopsided", lopsided_costs, 1)):
        synthetic_events = [TraceEvent(name, str(index), "llm_call", "openai/gpt-oss-20b", "", "", 0, int(cost / 0.0003 * 1000), 10, None) for index, cost in enumerate(costs)]
        from agentlens.report import _anomalies, total_cost_for_run
        anomalies = [item for item in _anomalies(synthetic_events, total_cost_for_run(synthetic_events)) if "High-cost" in item]
        assert len(anomalies) == expected, (name, anomalies)
        print(f"{name} cost anomaly test: PASS ({len(anomalies)} high-cost anomalies)")
    print("ALL REPORT TESTS PASSED")


if __name__ == "__main__":
    main()
