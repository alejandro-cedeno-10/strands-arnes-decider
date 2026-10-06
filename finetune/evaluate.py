"""Evaluate a Decider checkpoint on the DevOps alerts with the "Decider only" policy.

Policy: if the chosen runbook is not "escalate" and its confidence is at least the threshold, the
runbook runs automatically; everything else goes to a person. No LLM is called. For each threshold
it reports how many alerts are resolved automatically, how many automatic actions are wrong (the
expensive error), and the share sent to a person. Runs on CPU in bf16.

Usage:
    python finetune/evaluate.py --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 --name v19
    python finetune/evaluate.py --checkpoint results/finetune/checkpoint --name tuned

Writes results/finetune/eval_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "results" / "finetune"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load_alerts(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{name}_alerts.jsonl").read_text(encoding="utf-8").splitlines()]


def predict(engine, alerts: list[dict]) -> list[dict]:
    from build_examples import RUNBOOK_QUESTION, RUNBOOKS
    from strands_decider.schema import SystemOneRequest

    questions = {"runbook": {"type": "choice", "instructions": RUNBOOK_QUESTION, "criteria": RUNBOOKS}}
    rows = []
    for alert in alerts:
        request = SystemOneRequest.model_validate(
            {"state": {"service": alert["service"], "alert": alert["text"]}, "questions": questions})
        started = time.perf_counter()
        answer = engine.evaluate(request).model_dump()["answers"]["runbook"]
        rows.append({"expected": alert["runbook"], "choice": answer["choice"], "confidence": answer["confidence"],
                     "ms": (time.perf_counter() - started) * 1000})
    return rows


def policy(rows: list[dict], threshold: float) -> dict:
    auto = [r for r in rows if r["choice"] != "escalate" and r["confidence"] >= threshold]
    wrong = [r for r in auto if r["choice"] != r["expected"]]
    return {"threshold": threshold, "auto_resolved": len(auto), "auto_wrong": len(wrong),
            "to_person": len(rows) - len(auto), "n": len(rows)}


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from strands_decider.infer import SystemOneEngine, load_engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    SystemOneEngine._upcast_torso_for_cpu = lambda self: None  # type: ignore[method-assign]
    engine = load_engine(args.checkpoint, device="cpu")
    report = {"checkpoint": args.name}
    for split in ("test", "holdout"):
        rows = predict(engine, load_alerts(split))
        report[split] = {
            "accuracy": round(sum(r["choice"] == r["expected"] for r in rows) / len(rows), 3),
            "median_ms": round(statistics.median(r["ms"] for r in rows)),
            "policy": [policy(rows, t) for t in THRESHOLDS],
            "rows": rows,
        }
        print(json.dumps({"checkpoint": args.name, "split": split, "accuracy": report[split]["accuracy"],
                          "median_ms": report[split]["median_ms"], "policy": report[split]["policy"]}), flush=True)
    (DATA / f"eval_{args.name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
