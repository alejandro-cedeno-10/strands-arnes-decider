"""E. Model routing with Decider (a custom RoutingStrategy for ModelRouter).

ModelRouter and RoutingCandidate are the real ones from strands-agents 1.57.2; the candidates are
Nova Lite ("routine") and Nova Pro ("advanced") on Amazon Bedrock.

API verified (strands/models/routing/):
- `from strands.models import ModelRouter, RoutingCandidate, RoutingContext` (also
  `ClassifierStrategy`, `FallbackStrategy`, `RoutingStrategy`).
- `ModelRouter(models, *, strategy=None, max_switches=None)`; the first one is the default.
- `RoutingCandidate(model, name=None, description=None, metadata=None)`.
- Contract: `async def select(self, context, **kwargs) -> RoutingCandidate | None`. `select`
  MUST be a coroutine (the router raises TypeError otherwise). It must return an instance from
  `context.candidates` (compared by identity) or None (default at start; after a failure, routing
  ends).
- `RoutingContext` (frozen dataclass): messages, system_prompt, tool_specs, candidates,
  invocation_state, attempts. `messages` is a deep copy.

Design: the text comes from the last user message with `text` blocks; after a failure it
delegates to another candidate instead of returning None (failover); it falls back to "routine"
if Decider fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from a_http_client import ask, noul  # noqa: E402
from bedrock_models import NOVA_LITE, NOVA_PRO, nova_lite, nova_pro  # noqa: E402
from strands import Agent  # noqa: E402
from strands.models import ModelRouter, RoutingCandidate  # noqa: E402

HARD = noul(
    "Does this request need multi-step reasoning, planning or design work?",
    {"true": "a short factual or routine answer would not be enough",
     "false": "a routine, short answer is enough"},
)


def last_user_text(messages) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            text = " ".join(b["text"] for b in m["content"] if "text" in b).strip()
            if text:
                return text
    return ""


class DeciderStrategy:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.log: list[dict] = []

    async def select(self, context, **kwargs):
        """Pick by Decider's `hard` probability. After a failure, try the unused candidate.
        If Decider fails, stay with the cheap model."""
        by_name = {c.name: c for c in context.candidates}
        if context.attempts:
            used = {id(a.candidate) for a in context.attempts}
            return next((c for c in context.candidates if id(c) not in used), None)
        text = last_user_text(context.messages)
        try:
            p = (await asyncio.to_thread(ask, text, {"hard": HARD}))["answers"]["hard"]["noul"]
        except Exception:
            p = 0.0
        name = "advanced" if p >= self.threshold else "routine"
        chosen = by_name[name]
        self.log.append({"text": text, "hard": p, "route": name, "model_id": chosen.model.config["model_id"]})
        return chosen


def make_router(strategy: DeciderStrategy) -> ModelRouter:
    return ModelRouter(
        [RoutingCandidate(nova_lite(), name="routine"), RoutingCandidate(nova_pro(), name="advanced")],
        strategy=strategy,
    )


if __name__ == "__main__":
    strategy = DeciderStrategy()
    out = []
    for prompt in ["What time zone is Quito in?",
                   "Design a migration plan and architecture to move our billing system to AWS"]:
        agent = Agent(model=make_router(strategy), callback_handler=None)
        result = agent(prompt)
        usage = result.metrics.accumulated_usage
        out.append({"prompt": prompt, "answer": str(result).strip(),
                    "bedrock_usage": {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"]}})
    print(json.dumps({"models": {"routine": NOVA_LITE, "advanced": NOVA_PRO}, "routes": strategy.log,
                      "answers": out}, indent=2, ensure_ascii=False))
    ok = [r["route"] for r in strategy.log] == ["routine", "advanced"]
    print(f"\nSimple question -> routine (Nova Lite); design -> advanced (Nova Pro): {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
