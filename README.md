# AgentLens

## What this is

AgentLens is a local Python CLI for observing LangChain agent runs and auditing them afterward. It writes append-only JSONL traces and produces plain-English Markdown reports without a database or hosted service.

## Install

```bash
pip install -e .
```

## Quick start

Wire the callback into an agent run:

```python
from agentlens.capture import AgentLensCallback

callback = AgentLensCallback(run_id="my_run")
agent.invoke(input, config={"callbacks": [callback]})
```

For multi-agent systems, pass a `tags` list so each agent shows up as a distinct step in the report instead of being grouped under one generic model name:

```python
agent.invoke(input, config={"callbacks": [callback], "tags": ["QuestionGenerator"]})
```

Then generate the report:

```bash
agentlens audit --run my_run --no-llm
```

Use `python examples/sample_agent.py` to create a fully local sample trace.

## What's NOT in scope

This Phase 1 CLI has no database, no dashboard, no multi-run comparison, and no cloud storage. It observes LangChain callbacks only and does not orchestrate agents.

## Privacy note

No trace data leaves the local machine. Only input and output summaries truncated to 200 characters maximum are stored; full input/output payloads are never written. The optional narrative uses Anthropic only when explicitly enabled and `ANTHROPIC_API_KEY` is present.

## Optional: enabling LLM narratives

Set `ANTHROPIC_API_KEY` in `.env` or the environment, then run `agentlens audit --run my_run`. Omit `--no-llm` to enable the narrative summary, or use `--no-llm` to force a template-only report.

## Cost tracking

AgentLens calculates real per-call cost using a static, hardcoded pricing table (`cost.py`) keyed by exact model name. This is a known limitation, not a live lookup: rates are not fetched from any provider API and can go stale if a provider changes pricing. Currently priced models include OpenAI, Anthropic, and Groq's `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`. An unpriced model shows `$0.00` with a console warning rather than failing — add new models to the pricing table in `cost.py` as needed.

## Anomaly detection

A step is flagged as a high-cost anomaly if it exceeds `2.5x` the average cost per step in that run (not a flat percentage of total cost, which produced false positives on runs with only a few similarly-sized steps). Cost anomalies are skipped entirely on runs with fewer than 2 priced steps. Steps taking longer than 5 seconds are separately flagged as slow.

## How agent identification works

By default, AgentLens shows the real model name for each LLM call (e.g. `openai/gpt-oss-20b`) rather than the LangChain wrapper class name. In a multi-agent system where every agent uses the same model, this alone won't distinguish agents from each other — pass a `tags=["YourAgentName"]` list in the invocation config (see Quick Start above) to get a distinct, human-readable name per agent in the report, e.g. `QuestionGenerator (openai/gpt-oss-20b)`.

## Tested providers

The capture, timing, and token pipeline has been validated with the local `FakeListLLM` mock and a real Groq provider through `ChatGroq` using the currently available `openai/gpt-oss-20b` model, including a real multi-agent production pipeline with 4 distinct LLM-powered agents, retry/fallback logic, and batch-optimized calls.