"""LangChain callback handler that writes privacy-preserving local traces."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .trace_schema import TraceEvent, to_jsonl_line


class AgentLensCallback(BaseCallbackHandler):
    """Capture LangChain LLM and tool lifecycle callbacks in an append-only JSONL file."""

    def __init__(self, run_id: str, traces_dir: str = "traces") -> None:
        """Take a run identifier and local directory, creating no external resources."""
        self.run_id = run_id
        self.traces_dir = Path(traces_dir)
        self._started_at: dict[str, float] = {}
        self._pending_inputs: dict[str, str] = {}
        self._pending_names: dict[str, str] = {}

    @staticmethod
    def _run_key(run_id: UUID | str | None) -> str:
        """Convert a callback run identifier to a stable dictionary key."""
        return str(run_id) if run_id is not None else "unknown"

    @staticmethod
    def _summary(value: object) -> str:
        """Convert callback payloads to a bounded, non-sensitive summary."""
        return str(value)[:200]

    def _append(self, event: TraceEvent) -> None:
        """Append one event locally, creating its parent directory when needed."""
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.traces_dir / f"{self.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(to_jsonl_line(event) + "\n")

    def _record_safely(self, event: TraceEvent) -> None:
        """Write an event and report callback failures without interrupting an agent."""
        try:
            self._append(event)
        except Exception as exc:  # Callback failures must never escape to user code.
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def _event(self, event_type: str, name: str, input_summary: str = "", output_summary: str = "", tokens_in: int | None = None, tokens_out: int | None = None, duration_ms: int | None = None, error: str | None = None) -> TraceEvent:
        """Build a trace event with the current UTC timestamp and bounded fields."""
        return TraceEvent(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            name=name,
            input_summary=input_summary,
            output_summary=output_summary,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            error=error,
        )

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], run_id: UUID, **kwargs: Any) -> None:
        """Record an LLM start; malformed callback data is logged and ignored."""
        try:
            key = self._run_key(run_id)
            self._started_at[key] = time.perf_counter()
            self._pending_inputs[key] = self._summary(prompts)
            self._pending_names[key] = self._llm_name(serialized, kwargs)
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[Any], run_id: UUID, **kwargs: Any) -> None:
        """Record a chat-model start using the same timer and state as an LLM start."""
        self.on_llm_start(serialized, messages, run_id, **kwargs)

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, run_id: UUID, **kwargs: Any) -> None:
        """Inspect chain metadata; chain-level names may be limited by LangChain callbacks."""
        try:
            name = self._chain_name(serialized, kwargs)
            if name:
                self._pending_names[self._run_key(run_id)] = name
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def on_llm_end(self, response: Any, run_id: UUID, **kwargs: Any) -> None:
        """Record an LLM completion with token usage when available."""
        try:
            key = self._run_key(run_id)
            llm_output = getattr(response, "llm_output", None) or {}
            usage = self._token_usage(response, llm_output)
            name = self._pending_names.get(key, "unknown_model")
            input_summary = self._pending_inputs.get(key, "")
            duration = self._duration(key)
            output = self._summary(response)
            self._record_safely(self._event("llm_call", name, input_summary, output, _int_or_none(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count")), _int_or_none(usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("completion_token_count")), duration))
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, run_id: UUID, **kwargs: Any) -> None:
        """Record the start of a tool invocation without retaining its full input."""
        try:
            key = self._run_key(run_id)
            self._started_at[key] = time.perf_counter()
            self._pending_inputs[key] = self._summary(input_str)
            self._pending_names[key] = str(serialized.get("name") or serialized.get("id") or "tool")
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def on_tool_end(self, output: Any, run_id: UUID, **kwargs: Any) -> None:
        """Record a tool result and elapsed duration without raising callback errors."""
        try:
            key = self._run_key(run_id)
            name = self._pending_names.get(key, "tool")
            input_summary = self._pending_inputs.get(key, "")
            self._record_safely(self._event("tool_call", name, input_summary, self._summary(output), duration_ms=self._duration(key)))
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def on_chain_error(self, error: BaseException, run_id: UUID, **kwargs: Any) -> None:
        """Record a chain error as an error event while allowing the chain to fail normally."""
        try:
            key = self._run_key(run_id)
            name = self._pending_names.get(key) or self._chain_name(None, kwargs) or "chain"
            self._record_safely(self._event("error", name, self._pending_inputs.get(key, ""), error=str(error), duration_ms=self._duration(key)))
        except Exception as exc:
            print(f"AgentLens callback warning: {exc}", file=sys.stderr)

    def _duration(self, key: str) -> int | None:
        """Return elapsed milliseconds for a callback key and remove its temporary state."""
        started = self._started_at.pop(key, None)
        self._pending_inputs.pop(key, None)
        self._pending_names.pop(key, None)
        return None if started is None else int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _llm_name(serialized: dict[str, Any] | None, callback_kwargs: dict[str, Any]) -> str:
        """Extract an agent tag plus provider model ID, falling back to model metadata."""
        serialized = serialized or {}
        serialized_kwargs = serialized.get("kwargs") or {}
        metadata = callback_kwargs.get("metadata") or {}
        invocation_params = callback_kwargs.get("invocation_params") or {}
        tags = callback_kwargs.get("tags") or []
        custom_tags = [str(tag) for tag in tags if not str(tag).startswith("seq:")]
        agent_name = custom_tags[-1] if custom_tags else None
        model_name = None
        for source in (serialized_kwargs, metadata, invocation_params):
            for key in ("model", "model_name", "ls_model_name"):
                value = source.get(key) if isinstance(source, dict) else None
                if value:
                    model_name = str(value)
                    break
            if model_name:
                break
        if agent_name and model_name:
            return f"{agent_name} ({model_name})"
        if agent_name:
            return agent_name
        if model_name:
            return model_name
        name = serialized.get("name")
        if name:
            return str(name)
        identifiers = serialized.get("id")
        if isinstance(identifiers, list) and identifiers:
            return str(identifiers[-1])
        if identifiers:
            return str(identifiers)
        return "unknown_model"

    @staticmethod
    def _chain_name(serialized: dict[str, Any] | None, callback_kwargs: dict[str, Any]) -> str | None:
        """Extract a chain name or tag; LangChain may provide no richer chain identity."""
        serialized = serialized or {}
        name = serialized.get("name") or callback_kwargs.get("name")
        if name:
            return str(name)
        # LangChain may expose only generic sequence metadata here; custom invocation tags are the useful fallback.
        tags = callback_kwargs.get("tags") or []
        custom_tags = [str(tag) for tag in tags if not str(tag).startswith("seq:")]
        return custom_tags[-1] if custom_tags else None

    @staticmethod
    def _token_usage(response: Any, llm_output: object) -> dict[str, Any]:
        """Extract token usage from LLM metadata or chat-generation response metadata."""
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if isinstance(usage, dict):
                return usage
        generations = getattr(response, "generations", []) or []
        for generation_group in generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", {}) or {}
                usage = metadata.get("token_usage") or metadata.get("usage")
                if isinstance(usage, dict):
                    return usage
                usage_metadata = getattr(message, "usage_metadata", {}) or {}
                if isinstance(usage_metadata, dict):
                    return usage_metadata
        return {}


def _int_or_none(value: object) -> int | None:
    """Convert numeric token metadata to an integer, returning None for bad input."""
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
