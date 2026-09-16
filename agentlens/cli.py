"""Command-line entrypoint for AgentLens capture and audit commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import generate_report


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with capture and audit subcommands."""
    parser = argparse.ArgumentParser(prog="agentlens", description="Locally capture and audit LangChain agent traces.")
    subparsers = parser.add_subparsers(dest="command")
    capture = subparsers.add_parser("capture", help="show callback wiring instructions")
    capture.add_argument("--run", required=True, help="trace run identifier")
    audit = subparsers.add_parser("audit", help="generate a Markdown audit report")
    audit.add_argument("--run", required=True, help="trace run identifier")
    audit.add_argument("--no-llm", action="store_true", help="disable optional narrative generation")
    audit.add_argument("--output", help="report output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, execute a command, and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "capture":
        print(f"from agentlens.capture import AgentLensCallback\n\ncallback = AgentLensCallback(run_id={args.run!r})\nagent.invoke(input, config={{'callbacks': [callback]}})")
        return 0
    report = generate_report(args.run, use_llm=not args.no_llm)
    output_path = Path(args.output) if args.output else Path("reports") / f"{args.run}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
