"""F. Strands harness + the Decider gate.

`create_harness` is the real one from strands-harness 0.1.2 (PyPI); the model is Nova Lite on
Amazon Bedrock. Run it from the harness virtualenv (env/requirements-lock-harness.txt).

Verified in the strands-harness 0.1.2 code:
- The PyPI package is `strands-harness` (`strands-agents-harness` does not exist on PyPI);
  on npm it is `@strands-agents/harness`.
- `create_harness(..., interventions=None)` defaults to "off — every call runs".
- `interventions` accepts presets ("ask", "smart"), a natural-language policy, a .cedar file, or
  an InterventionHandler instance, which is passed through unchanged. That is why DeciderGate
  plugs in directly.
- `model` accepts a `Model` instance, a "provider/name" string or a Bedrock id. Here a prebuilt
  `BedrockModel` (Nova Lite) is passed, with `caching=False` because the harness prompt cache does
  not apply to an already built instance.

The instructions prompt is the "eager" one from d_gate_bedrock.py (it assumes Seattle when there
is no city). Without it, the harness default prompt makes Nova Lite ask for the city on its own
and the gate is not exercised.

For the demo the built-in tools, memory, skills and session are disabled so the gate is the only
observable effect.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from b_gate_interventions import DeciderGate  # noqa: E402
from bedrock_models import nova_lite  # noqa: E402
from d_gate_bedrock import SYSTEM as EAGER_PROMPT  # noqa: E402
from strands import tool  # noqa: E402
from strands_harness import create_harness  # noqa: E402

EXECUTED: list[str] = []


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    EXECUTED.append(city)
    return f"{city}: 18°C"


def run(prompt: str, gated: bool) -> dict:
    EXECUTED.clear()
    agent = create_harness(
        model=nova_lite(),
        caching=False,
        instructions=EAGER_PROMPT,
        tools=[get_weather],
        builtin_tools=[], builtin_plugins=[], memory=False, skills=False, session=False,
        interventions=DeciderGate() if gated else None,
        callback_handler=None,
    )
    result = agent(prompt)
    usage = result.metrics.accumulated_usage
    return {"prompt": prompt, "gated": gated, "executed": list(EXECUTED), "answer": str(result).strip(),
            "bedrock_usage": {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
                              "model_calls": result.metrics.cycle_count}}


if __name__ == "__main__":
    reps = int(os.getenv("REPS", "3"))
    runs = []
    for _ in range(reps):
        runs += [run("What's the weather?", gated=False), run("What's the weather?", gated=True)]
    print(json.dumps(runs, indent=2, ensure_ascii=False))
    free = [r for r in runs if not r["gated"]]
    gated = [r for r in runs if r["gated"]]
    invented = sum(bool(r["executed"]) for r in free)
    blocked = sum(not r["executed"] for r in gated)
    print(f"\nWithout the gate the harness runs the tool with an invented city: {invented}/{reps}")
    print(f"With DeciderGate the tool does not run:                          {blocked}/{reps}")
    ok = blocked == reps
    print(f"Result: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
