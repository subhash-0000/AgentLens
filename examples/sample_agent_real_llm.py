"""Run a real Groq chat-model call and local calculator tool call."""

import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agentlens.capture import AgentLensCallback


def calculate(expression: str) -> int:
    """Calculate the two-number addition used by the sample, rejecting bad input."""
    left, right = (int(part.strip()) for part in expression.split("+"))
    return left + right


def main() -> None:
    """Call Groq, capture its response, and capture one local calculator call."""
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required; set it in .env or the environment.")
    callback = AgentLensCallback(run_id="sample_run_real_llm")
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    model.invoke("Explain briefly that you will add 2 and 3.", config={"callbacks": [callback]})
    callback.on_tool_start({"name": "calculator"}, "2 + 3", run_id="real-sample-tool")
    time.sleep(0.002)
    result = calculate("2 + 3")
    callback.on_tool_end(str(result), run_id="real-sample-tool")
    print(f"Real Groq sample completed: {result}")


if __name__ == "__main__":
    main()