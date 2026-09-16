"""Data structures and JSONL persistence for local agent traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_MAX_SUMMARY_LENGTH = 200


def _truncate(value: object) -> str:
    """Convert a value to text and truncate it to 200 characters."""
    text = "" if value is None else str(value)
    return text[:_MAX_SUMMARY_LENGTH]


@dataclass
class TraceEvent:
    """Represent one local agent event with privacy-preserving summaries."""

    run_id: str
    timestamp: str
    event_type: str
    name: str
    input_summary: str
    output_summary: str
    tokens_in: int | None
    tokens_out: int | None
    duration_ms: int | None
    error: str | None

    def __post_init__(self) -> None:
        """Truncate summaries and normalize optional error text after construction."""
        self.input_summary = _truncate(self.input_summary)
        self.output_summary = _truncate(self.output_summary)
        if self.error is not None:
            self.error = _truncate(self.error)


def to_jsonl_line(event: TraceEvent) -> str:
    """Serialize a trace event to one JSON line; invalid values raise naturally."""
    return json.dumps(asdict(event), ensure_ascii=True)


def from_jsonl_line(line: str) -> TraceEvent:
    """Parse one JSON line into a TraceEvent; malformed input raises ValueError."""
    try:
        payload = json.loads(line)
        return TraceEvent(**payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid trace JSONL line") from exc


def load_trace(run_id: str, traces_dir: str = "traces") -> list[TraceEvent]:
    """Load a run's events, returning an empty list when its file is absent."""
    path = Path(traces_dir) / f"{run_id}.jsonl"
    if not path.exists():
        return []
    events: list[TraceEvent] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            try:
                events.append(from_jsonl_line(line))
            except ValueError as exc:
                raise ValueError(f"Invalid trace line {line_number} in {path}") from exc
    return events
