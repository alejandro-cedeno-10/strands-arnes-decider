"""B (variant). Decider down + `on_error = "deny"`: the tool does NOT run (fail-closed).

Points the client at a port with no server (STRANDS_DECIDER_URL=http://127.0.0.1:8197) and reuses
the d_gate_bedrock.py agent with Nova Lite. With a well-formed request ("Quito") the tool would run
without a gate; with the gate down and `on_error = "deny"` the SDK cancels it.

Usage:
    STRANDS_DECIDER_URL=http://127.0.0.1:8197 python examples/b_fail_closed.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("STRANDS_DECIDER_URL", "http://127.0.0.1:8197")
sys.path.insert(0, os.path.dirname(__file__))

import d_gate_bedrock as demo  # noqa: E402
from b_gate_interventions import DeciderGate  # noqa: E402

if __name__ == "__main__":
    print("Decider URL:", os.environ["STRANDS_DECIDER_URL"])
    print("DeciderGate on_error:", DeciderGate().on_error)
    result = demo.run("What's the weather in Quito?")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = result["tool_executed_with"] == []
    print(f"\nDecider down + on_error='deny' -> tool NOT executed: {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
