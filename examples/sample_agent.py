"""Run a dependency-local sample LLM and tool flow to create a trace."""

import time

from langchain_community.llms.fake import FakeListLLM

from agentlens.capture import AgentLensCallback


def calculate(expression: str) -> int:
    """Calculate the two-number addition used by the sample, rejecting bad input."""
    left, right = (int(part.strip()) for part in expression.split("+"))
    return left + right


def main() -> None:
    """Run one fake LLM call and one local calculator tool call."""
    callback = AgentLensCallback(run_id="sample_run")
    model = FakeListLLM(responses=["I will add 2 and 3."])
    model.invoke("Add 2 and 3", config={"callbacks": [callback]})
    callback.on_tool_start({"name": "calculator"}, "2 + 3", run_id="sample-tool")
    time.sleep(0.002)
    result = calculate("2 + 3")
    callback.on_tool_end(str(result), run_id="sample-tool")
    print(f"Sample agent completed: {result}")


if __name__ == "__main__":
    main()
