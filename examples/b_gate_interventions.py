"""B. Decider gate before every tool call, using interventions (strands-agents 1.57.2).

Running this file directly only prints the probabilities with and without a city. The full gate
runs with the Strands agent and Bedrock Nova Lite in d_gate_bedrock.py and f_harness_decider.py.

API verified against the installed code (strands-agents 1.57.2):
- `from strands.interventions import InterventionHandler, Proceed, Guide, Deny`.
- `name` is an abstract property; it is overridden with a class attribute.
- The Python error-handling attribute is `on_error` (snake_case), with values "throw" (default),
  "proceed" (fail-open) and "deny" (fail-closed). `onError` is the TypeScript name.
- Methods may be `async def`.
- `event.tool_use["name"]`, `event.tool_use["input"]`, `event.tool_use["toolUseId"]`, and the
  conversation in `event.agent.messages`.
- `Guide(feedback)` in before_tool_call cancels the tool with "GUIDANCE: <feedback>";
  `Deny(reason=...)` cancels it with "DENIED: <reason>" and stops the handler chain.

Design:
- `state` includes the conversation (user and assistant text) plus the call, as in the official
  example.
- Two noul questions with `criteria` (args_grounded and premature) in a single request, so the
  state is read once.
- The HTTP call runs in a thread (`asyncio.to_thread`) so it does not block the event loop.
- A single threshold (0.45), as in the official example, and `Guide` instead of `Deny`. A first
  version used three zones (act at 0.9, reject at 0.1) and with the real model it also blocked
  the well-grounded "Quito" request.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from a_http_client import ask, noul  # noqa: E402
from strands.interventions import Guide, InterventionHandler, Proceed  # noqa: E402

QUESTIONS = {
    "args_grounded": noul(
        "Are the tool's argument values grounded in facts the user actually provided?",
        {
            "true": "every argument value traces back to something the user said",
            "false": "an argument value was guessed or invented, not stated by the user",
        },
    ),
    "premature": noul(
        "Is it premature to call this tool now, before clarifying with the user?",
        {
            "true": "the assistant should ask a clarifying question before calling the tool",
            "false": "there is nothing left to clarify; calling now is appropriate",
        },
    ),
}


def conversation(messages: list[dict[str, Any]], max_turns: int = 12) -> list[dict[str, str]]:
    """Plain text of the last turns. Tool results are excluded: they do not come from the user
    and could carry prompt injection."""
    out = []
    for m in messages[-max_turns:]:
        text = " ".join(b["text"] for b in m.get("content", []) if "text" in b).strip()
        if text:
            out.append({"role": m["role"], "text": text})
    return out


class DeciderGate(InterventionHandler):
    """Tool gate backed by Decider. It is fail-closed: with `on_error = "deny"`, if Decider does
    not answer the tool does not run. `gated=None` watches every tool.

    The policy follows the official example (`examples/strands/tool_call_intervention.py`): a
    single `yes` threshold. If P(args_grounded) < yes, or P(premature) >= yes, it returns `Guide`
    and the model asks the user again; otherwise `Proceed`. The 0.45 threshold is illustrative:
    calibrate it with your own traffic. With a 0.9 threshold to act, well-grounded calls
    (args_grounded 0.55 to 0.70 with "Quito") were blocked too.
    """

    name = "decider-gate"
    on_error = "deny"

    def __init__(self, gated: set[str] | None = None, yes: float = 0.45):
        self.gated = gated
        self.yes = yes
        self.log: list[dict[str, Any]] = []

    async def before_tool_call(self, event, **kwargs):
        tool = event.tool_use["name"]
        if self.gated is not None and tool not in self.gated:
            return Proceed(reason="not gated")
        state = {
            "conversation": conversation(event.agent.messages),
            "tool_call": {"name": tool, "input": event.tool_use["input"]},
        }
        resp = await asyncio.to_thread(ask, state, QUESTIONS)
        answers = resp["answers"]
        grounded, premature = answers["args_grounded"]["noul"], answers["premature"]["noul"]
        reason = f"grounded={grounded:.2f} premature={premature:.2f}"
        verdict = "proceeded"
        if grounded < self.yes:
            verdict = "guided"
            action = Guide(
                "The arguments are not grounded in anything the user said; they look guessed. "
                "Ask the user to provide the missing values (for example, which city) instead "
                "of inventing them.",
                reason=reason,
            )
        elif premature >= self.yes:
            verdict = "guided"
            action = Guide(
                "It is too early to call this tool. Clarify with the user first, then try again.",
                reason=reason,
            )
        else:
            action = Proceed(reason=reason)
        self.log.append({"tool": tool, "input": event.tool_use["input"], "args_grounded": grounded,
                         "premature": premature, "verdict": verdict, "latency_ms": resp.get("latency_ms")})
        return action


if __name__ == "__main__":
    for user, city in [("What's the weather?", "Seattle"), ("What's the weather in Quito?", "Quito")]:
        st = {"conversation": [{"role": "user", "text": user}],
              "tool_call": {"name": "get_weather", "input": {"city": city}}}
        print(user, "->", {k: v["noul"] for k, v in ask(st, QUESTIONS)["answers"].items()})
