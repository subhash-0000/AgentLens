"""Token pricing and cost aggregation for trace events."""

from __future__ import annotations

import sys
import re

from .trace_schema import TraceEvent

# Prices as of 2026-09-19 - verify against https://groq.com/pricing/ before billing.
# Groq public rates converted from USD per million tokens to USD per 1K tokens.
PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4": (0.03, 0.06),
    "gpt-4o": (0.0025, 0.01),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "openai/gpt-oss-20b": (0.000075, 0.0003),
    "qwen/qwen3.8-27b": (0.00029, 0.00059),
    "llama-3.1-8b-instant": (0.00005, 0.00008),
}


def calculate_cost(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for known model token counts, warning and returning zero otherwise."""
    pricing = _pricing_for_name(model_name)
    if pricing is None:
        print(f"AgentLens warning: unknown model pricing for {model_name!r}", file=sys.stderr)
        return 0.0
    return (max(tokens_in, 0) / 1000 * pricing[0]) + (max(tokens_out, 0) / 1000 * pricing[1])


def _pricing_for_name(model_name: str) -> tuple[float, float] | None:
    """Resolve direct model names and AgentLens names formatted as `Agent (model)`."""
    normalized = model_name.lower()
    pricing = PRICING_PER_1K.get(normalized)
    if pricing is not None:
        return pricing
    match = re.search(r"\(([^()]+)\)$", normalized)
    return PRICING_PER_1K.get(match.group(1)) if match else None


def total_cost_for_run(events: list[TraceEvent]) -> float:
    """Sum known-model input and output costs across all LLM events."""
    return sum(calculate_cost(event.name, event.tokens_in or 0, event.tokens_out or 0) for event in events if event.event_type == "llm_call")


def cost_by_step(events: list[TraceEvent]) -> list[tuple[str, float]]:
    """Return LLM step costs sorted from most expensive to least expensive."""
    costs = [(event.name, calculate_cost(event.name, event.tokens_in or 0, event.tokens_out or 0)) for event in events if event.event_type == "llm_call"]
    return sorted(costs, key=lambda item: item[1], reverse=True)
