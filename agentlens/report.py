"""Markdown report generation for local AgentLens traces."""

from __future__ import annotations

import os
import sys
from typing import Any

from .cost import cost_by_step, total_cost_for_run
from .trace_schema import TraceEvent, load_trace

HIGH_COST_MULTIPLIER = 2.5


def generate_report(run_id: str, use_llm: bool = True, traces_dir: str = "traces") -> str:
    """Load a run and return a complete Markdown audit, gracefully handling missing traces."""
    events = load_trace(run_id, traces_dir)
    if not events:
        return f"No trace found for run_id `{run_id}`."

    total_cost = total_cost_for_run(events)
    total_duration = sum(event.duration_ms or 0 for event in events)
    errors = [event for event in events if event.error or event.event_type == "error"]
    lines = [f"# AgentLens Audit: {run_id}", "", "## Summary", f"- Run ID: `{run_id}`", f"- Steps: {len(events)}", f"- Total cost: ${total_cost:.6f}", f"- Total duration: {total_duration}ms", f"- Errors: {len(errors)}", ""]

    narrative = _narrative(events, use_llm)
    if narrative:
        lines.extend(["## What Happened", narrative, ""])

    lines.extend(["## Step-by-step timeline", ""])
    for event in events:
        cost = _event_cost(event)
        duration = "?" if event.duration_ms is None else str(event.duration_ms)
        detail = f"; error: {event.error}" if event.error else ""
        lines.append(f"- [{event.timestamp}] {event.event_type}: {event.name} - {duration}ms, ${cost:.6f}{detail}")
    lines.append("")

    lines.extend(["## Most expensive steps", ""])
    expensive = cost_by_step(events)[:3]
    if expensive:
        lines.extend(f"- {name}: ${cost:.6f}" for name, cost in expensive)
    else:
        lines.append("- No LLM steps with recorded token usage.")
    lines.append("")

    lines.extend(["## Anomalies", ""])
    anomalies = _anomalies(events, total_cost)
    lines.extend(f"- {item}" for item in anomalies) if anomalies else lines.append("- None detected.")
    lines.append("")
    return "\n".join(lines)


def _event_cost(event: TraceEvent) -> float:
    """Calculate one event's cost without charging non-LLM events."""
    if event.event_type != "llm_call":
        return 0.0
    from .cost import calculate_cost
    return calculate_cost(event.name, event.tokens_in or 0, event.tokens_out or 0)


def _anomalies(events: list[TraceEvent], total_cost: float) -> list[str]:
    """Find slow, expensive, or failed steps in a trace."""
    anomalies: list[str] = []
    cost_events = [event for event in events if event.event_type == "llm_call" and _event_cost(event) > 0]
    average_cost = total_cost / len(cost_events) if len(cost_events) >= 2 else None
    for event in events:
        if event.duration_ms is not None and event.duration_ms > 5000:
            anomalies.append(f"Slow step: {event.name} took {event.duration_ms}ms.")
        if average_cost is not None and _event_cost(event) > average_cost * HIGH_COST_MULTIPLIER:
            anomalies.append(f"High-cost step: {event.name} cost more than {HIGH_COST_MULTIPLIER}x the average step cost.")
        if event.error:
            anomalies.append(f"Error in {event.name}: {event.error}")
    return anomalies


def _narrative(events: list[TraceEvent], use_llm: bool) -> str | None:
    """Optionally ask Anthropic for a short narrative, falling back silently to no narrative."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not use_llm or not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        summary = [{"type": event.event_type, "name": event.name, "input": event.input_summary, "output": event.output_summary, "error": event.error} for event in events]
        response: Any = client.messages.create(model="claude-sonnet-4-6", max_tokens=180, messages=[{"role": "user", "content": f"Write 2-3 plain-English sentences explaining what this agent was trying to do and whether anything looks wrong. Trace summary: {summary}"}])
        return str(response.content[0].text).strip()
    except Exception as exc:
        print(f"AgentLens warning: narrative generation failed: {exc}", file=sys.stderr)
        return None
