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

## Tested providers

The capture, timing, and token pipeline has been validated with the local `FakeListLLM` mock and a real Groq provider through `ChatGroq` using the currently available `openai/gpt-oss-20b` model.
