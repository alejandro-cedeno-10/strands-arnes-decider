"""A. Minimal HTTP client for the Strands Decider server (/v1/systemone).

Tested against the real strands-decider 0.1.0 server (RTX 4060, CUDA).

Notes:
- Supports the three primitives: `noul`, `choice` and `score`.
- `choice` ALWAYS sends `criteria` as a dict {option: description}. A list is converted to
  {option: ""}, as the CLI does. Sending `criteria` as a list or null returns 422.
- `noul` accepts optional `criteria` {"true": ..., "false": ...}, which is what
  `Decider.noul(instruction, {...})` does in the official example.
- `score` sends `criteria` as an ascending list of 2 to 10 levels.
- Uses only the standard library (no `requests`) so it runs in any virtualenv.

Usage:
    STRANDS_DECIDER_URL=http://127.0.0.1:8099 python examples/a_http_client.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DECIDER_URL = os.getenv("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")


class DeciderError(RuntimeError):
    """The server did not answer or rejected the request (e.g. 422 for an invalid schema)."""


def noul(instructions: str, criteria: dict[str, str] | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return q


def choice(instructions: str, options: dict[str, str] | list[str]) -> dict[str, Any]:
    if isinstance(options, list):
        options = {o: "" for o in options}
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def ask(state: str | dict | list, questions: dict[str, dict[str, Any]], *,
        url: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """POST /v1/systemone. Returns the full JSON (answers, usage, latency_ms, model)."""
    body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        f"{url or DECIDER_URL}/v1/systemone",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise DeciderError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise DeciderError(f"Decider unavailable at {url or DECIDER_URL}: {e}") from e


def ask_noul(state: str | dict | list, questions: dict[str, str | dict[str, Any]], **kw: Any) -> dict[str, float]:
    """Shortcut: {name: question} -> {name: P(true)}."""
    qs = {k: (noul(v) if isinstance(v, str) else v) for k, v in questions.items()}
    answers = ask(state, qs, **kw)["answers"]
    return {k: float(v["noul"]) for k, v in answers.items()}


if __name__ == "__main__":
    state = "Help! My payouts have been failing for 3 days!"
    out = ask(state, {
        "team": choice("Which team should handle this?", ["billing", "sales", "retail"]),
        "urgent": noul("Does this convey urgency?"),
        "frustration": score("How frustrated is the writer?", ["calm", "frustrated", "depressed"]),
    })
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("ask_noul ->", ask_noul(state, {"is_urgent": "Does this convey urgency?"}))
