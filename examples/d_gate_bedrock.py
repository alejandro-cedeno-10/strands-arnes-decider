"""D. Agent with Bedrock Nova (System 2) and the Decider gate (System 1).

Observed with the real Decider (RTX 4060) and Bedrock Nova Lite: 3 runs, no city 3/3 and Quito 3/3.
With 10 runs: no city 10/10 and Quito 9/10; the exception was a gate false positive
(args_grounded 0.33).

Observable goal:
- "What's the weather?"          -> the tool does NOT run and the agent asks for the city.
- "What's the weather in Quito?" -> the tool DOES run.
The `EXECUTED` counter lives inside the tool, so it counts real executions.

Requirements:
    pip install strands-agents==1.57.2 strands-decider
    export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-east-1
    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python examples/d_gate_bedrock.py      # REPS=3 by default

`BedrockModel(model_id=..., region_name=...)` is the real signature; here it is built with
`nova_lite()` from `bedrock_models.py`. The system prompt is "eager", like the official example
(it assumes Seattle): it pushes the model to invent the city so the gate has something to stop.
With a softer prompt ("call get_weather right away; ask no questions") Nova Lite sometimes asked
for the city on its own without calling the tool, and then the gate was not exercised; that is why
`gate_interventions` counts how many times the gate stepped in.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from b_gate_interventions import DeciderGate  # noqa: E402
from bedrock_models import nova_lite  # noqa: E402
from strands import Agent, tool  # noqa: E402

EXECUTED: list[str] = []


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    EXECUTED.append(city)
    return f"{city}: 18°C, partly cloudy"


SYSTEM = (
    "You are an eager weather assistant. Always answer weather questions by calling the "
    "get_weather tool immediately. Never ask the user for clarification -- if no city is "
    "given, just assume Seattle and call the tool with city='Seattle'."
)


def run(prompt: str) -> dict:
    EXECUTED.clear()
    gate = DeciderGate()
    agent = Agent(
        model=nova_lite(),
        tools=[get_weather],
        system_prompt=SYSTEM,
        interventions=[gate],
        callback_handler=None,
    )
    result = agent(prompt)
    usage = result.metrics.accumulated_usage
    return {
        "prompt": prompt,
        "tool_executed_with": list(EXECUTED),
        "gate_interventions": sum(row["verdict"] == "guided" for row in gate.log),
        "decider": gate.log,
        "bedrock_usage": {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
                          "model_calls": result.metrics.cycle_count},
        "answer": str(result).strip(),
    }


if __name__ == "__main__":
    reps = int(os.getenv("REPS", "3"))
    runs, ok1, ok2 = [], 0, 0
    for _ in range(reps):
        r1 = run("What's the weather?")
        r2 = run("What's the weather in Quito?")
        runs += [r1, r2]
        ok1 += r1["tool_executed_with"] == [] and "city" in r1["answer"].lower()
        ok2 += r2["tool_executed_with"] == ["Quito"]
    print(json.dumps(runs, indent=2, ensure_ascii=False))
    no_city = runs[0::2]
    proposed = sum(bool(r["decider"]) for r in no_city)
    stopped = sum(r["gate_interventions"] > 0 for r in no_city)
    print(f"\nNo city   -> tool not executed and the agent asks for the city: {ok1}/{reps} {'OK' if ok1 == reps else 'FAIL'}")
    print(f"  the model proposed the invented call in {proposed}/{reps} and the gate stopped it in {stopped}/{reps}")
    print(f"With city -> tool executed with Quito:                           {ok2}/{reps} {'OK' if ok2 == reps else 'FAIL'}")
    sys.exit(0 if ok1 == ok2 == reps else 1)
