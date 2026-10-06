"""Generate a labelled synthetic dataset of production alerts with Amazon Bedrock (Nova Pro).

Each alert is labelled with the runbook that should handle it and whether its fix could be
destructive. The 12 hand-written alerts in examples/g_devops_self_healing.py are never generated
or used for training: they stay as the final held-out test.

Usage:
    python finetune/generate_alerts.py [--per-label 80] [--batch 20]

Writes results/finetune/alerts.jsonl. Costs a few US cents with Nova Pro.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import boto3

MODEL_ID = os.getenv("GENERATOR_MODEL_ID", "us.amazon.nova-pro-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")
OUT = Path(__file__).resolve().parents[1] / "results" / "finetune" / "alerts.jsonl"

LABELS = {
    "restart_service": "the process crashed, hung, was OOMKilled, deadlocked or its health check keeps failing; a restart fixes it",
    "scale_out": "traffic spike, CPU or memory saturation from load, queue backlog, or latency growing with load; more replicas fix it",
    "rollback_deploy": "errors, crashes or regressions that started right after a new deployment, release or config rollout",
    "clear_disk": "disk, volume or inode exhaustion, no space left on device, logs or temp files filling the disk",
    "escalate": ("anything a runbook must not touch: unclear cause, data or database problems, failed migrations, "
                 "security incidents, third-party outages, or several plausible causes at once "
                 "(for example a deploy AND a traffic spike together)"),
}
DESTRUCTIVE_HINT = ("destructive=true only when the safe fix could lose or corrupt data or cannot be undone "
                    "(databases, migrations, deleting data that is not a log or temp file); restarts, "
                    "scaling and rollbacks of stateless services are not destructive")

PROMPT = """You write realistic production monitoring alerts for a synthetic dataset.
Write {n} different alerts whose correct handling is the runbook "{label}": {meaning}.
Vary services (APIs, workers, databases, queues, caches, proxies, batch jobs), wording, metrics,
units, tone (pager, Slack, Prometheus/CloudWatch style) and length (one to three sentences).
{extra}
Label each alert: {destructive}.
Never mention the runbook name or the words "runbook", "restart", "scale out", "rollback" or "escalate".
Return ONLY a JSON array of objects: {{"service": "...", "text": "...", "destructive": true|false}}."""

ESCALATE_EXTRA = ("At least a third must be deliberately ambiguous, mixing signals of two runbooks "
                  "(deploy plus traffic, disk plus database, crash plus security), so no single runbook is safe.")


def generate_batch(client, label: str, n: int, seen: set[str]) -> list[dict]:
    prompt = PROMPT.format(n=n, label=label, meaning=LABELS[label], destructive=DESTRUCTIVE_HINT,
                           extra=ESCALATE_EXTRA if label == "escalate" else "")
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4000, "temperature": 0.9},
    )
    text = response["output"]["message"]["content"][0]["text"]
    match = re.search(r"\[.*\]", text, re.S)
    rows = json.loads(match.group(0)) if match else []
    fresh = []
    for row in rows:
        key = row.get("text", "").strip().lower()
        if key and key not in seen and isinstance(row.get("destructive"), bool):
            seen.add(key)
            fresh.append({"service": row.get("service", "unknown"), "text": row["text"].strip(),
                          "runbook": label, "destructive": row["destructive"]})
    usage = response["usage"]
    print(json.dumps({"label": label, "kept": len(fresh), "input_tokens": usage["inputTokens"],
                      "output_tokens": usage["outputTokens"]}), flush=True)
    return fresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=80)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    client = boto3.Session(region_name=REGION).client("bedrock-runtime")
    seen: set[str] = set()
    alerts: list[dict] = []
    for label in LABELS:
        rows: list[dict] = []
        for _ in range(args.per_label // args.batch * 2):
            if len(rows) >= args.per_label:
                break
            rows += generate_batch(client, label, args.batch, seen)
        alerts += rows[: args.per_label]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in alerts), encoding="utf-8")
    print(f"{len(alerts)} alerts -> {OUT}")


if __name__ == "__main__":
    main()
