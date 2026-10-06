"""C. Human-in-the-loop with Decider as the risk classifier.

HumanInTheLoop, ClassifierResult, Agent and BedrockModel (Nova Lite) are the real ones from
strands-agents 1.57.2.

API verified (strands/vended_interventions/hitl/):
- `from strands.vended_interventions.hitl import HumanInTheLoop`
- `ClassifierResult` is NOT exported from `strands.vended_interventions.hitl`; import it from
  `strands.vended_interventions.hitl.classifier` (fields `requires_human_in_the_loop: bool`,
  `reason: str | None = None`).
- `HumanInTheLoop(*, allowed_tools=None, classifier=None, enable_trust=False,
  evaluate_trust=None, evaluate=None, ask=None)`. `classifier` accepts True,
  `LLMClassifierConfig` or a callable (sync or async) `(event, **kwargs) -> ClassifierResult`.
- Fail-closed: if the classifier raises, returns another type or a non-boolean value, approval
  is requested.
- Tools in `allowed_tools` do NOT go through the classifier.
- `ask="stdio"` asks in the terminal; `ask=None` pauses the agent with an interrupt;
  `ask=callable` receives the prompt and returns "y"/"n"/True.

Design: async classifier (does not block the event loop); structured state with the call; if
Decider fails, approval is requested (explicit fail-closed); for the automated demo `ask` is a
callable that records the prompt and answers "n".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from a_http_client import ask, noul  # noqa: E402
from bedrock_models import nova_lite  # noqa: E402
from strands import Agent, tool  # noqa: E402
from strands.vended_interventions.hitl import HumanInTheLoop  # noqa: E402
from strands.vended_interventions.hitl.classifier import ClassifierResult  # noqa: E402

RISKY = noul(
    "Does this tool call modify data or send anything outside the company?",
    {"true": "the call writes, deletes, pays or sends something to someone",
     "false": "the call only reads information"},
)


async def decider_risk(event, **kwargs) -> ClassifierResult:
    """If Decider does not answer, human approval is requested (fail-closed)."""
    state = {"tool_call": {"name": event.tool_use["name"], "input": event.tool_use["input"]}}
    try:
        p = (await asyncio.to_thread(ask, state, {"risky": RISKY}))["answers"]["risky"]["noul"]
    except Exception as e:
        return ClassifierResult(requires_human_in_the_loop=True, reason=f"decider error: {e}")
    return ClassifierResult(requires_human_in_the_loop=p >= 0.5, reason=f"decider risky={p:.2f}")


EXECUTED: list[str] = []
PROMPTS: list[str] = []


@tool
def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient address.
        body: Message body.
    """
    EXECUTED.append(f"send_email:{to}")
    return "sent"


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city: City name.
    """
    EXECUTED.append(f"get_weather:{city}")
    return f"{city}: 18°C"


def human(prompt: str, **kwargs) -> str:
    """Simulated human for the demo: rejects every approval request."""
    PROMPTS.append(prompt)
    return "n"


def make_agent() -> Agent:
    return Agent(
        model=nova_lite(),
        tools=[send_email, get_weather],
        interventions=[HumanInTheLoop(classifier=decider_risk, ask=human)],
        callback_handler=None,
    )


if __name__ == "__main__":
    results = []
    for prompt in ["Please email Ana that the report is ready", "What's the weather in Quito?"]:
        EXECUTED.clear(); PROMPTS.clear()
        out = make_agent()(prompt)
        results.append({"prompt": prompt, "executed": list(EXECUTED), "human_prompts": list(PROMPTS),
                        "answer": str(out).strip()})
    print(json.dumps(results, indent=2, ensure_ascii=False))
    ok1 = results[0]["executed"] == [] and len(results[0]["human_prompts"]) == 1
    ok2 = results[1]["executed"] == ["get_weather:Quito"] and results[1]["human_prompts"] == []
    print(f"\nsend_email -> approval requested and the rejection blocked it: {'OK' if ok1 else 'FAIL'}")
    print(f"get_weather -> ran without asking for approval:               {'OK' if ok2 else 'FAIL'}")
    sys.exit(0 if ok1 and ok2 else 1)
