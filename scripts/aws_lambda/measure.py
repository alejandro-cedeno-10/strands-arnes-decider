"""Measure the Decider Lambda function: 1 cold invocation, N sequential warm ones, and 1 with three questions.

    python scripts/aws_lambda/measure.py fp32
    python scripts/aws_lambda/measure.py bf16

Changing DECIDER_KEEP_BF16 in the function configuration forces a new execution environment (a cold
start). Each invocation uses LogType=Tail; the client duration and the REPORT line are saved per call.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import time

import botocore.config

import common as c

COLD, WARM, THREE_QUESTIONS = "cold", "warm", "three_questions"
NOUL_PAYLOAD = {
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
}
THREE_QUESTIONS_PAYLOAD = {
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "", "sales": "", "retail": ""},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["calm", "slightly annoyed", "frustrated", "angry", "furious"],
        },
    },
}
REPORT_FIELDS = {
    "duration_ms": r"Duration: ([\d.]+) ms",
    "billed_ms": r"Billed Duration: (\d+) ms",
    "memory_size_mb": r"Memory Size: (\d+) MB",
    "max_memory_mb": r"Max Memory Used: (\d+) MB",
    "init_ms": r"Init Duration: ([\d.]+) ms",
}
CSV_COLUMNS = [
    "i", "kind", "client_ms", "duration_ms", "billed_ms", "init_ms", "max_memory_mb", "memory_size_mb",
    "function_error", "decider_latency_ms", "cold_start_load_ms", "urgent_noul", "team", "team_confidence",
    "frustration_score", "report",
]


def lambda_client():
    config = botocore.config.Config(read_timeout=960, connect_timeout=10, retries={"max_attempts": 1})
    return c.session().client("lambda", config=config)


def parse_report(log_text: str) -> dict:
    line = next((ln for ln in log_text.splitlines() if ln.startswith("REPORT")), "")
    parsed = {"report": line.strip()}
    for key, pattern in REPORT_FIELDS.items():
        match = re.search(pattern, line)
        parsed[key] = float(match.group(1)) if match else None
    return parsed


def set_variant(lam, variant: str) -> None:
    value = "1" if variant == "bf16" else "0"
    lam.update_function_configuration(
        FunctionName=c.FUNCTION_NAME, Environment={"Variables": {"DECIDER_KEEP_BF16": value}}
    )
    lam.get_waiter("function_updated_v2").wait(FunctionName=c.FUNCTION_NAME, WaiterConfig={"Delay": 3, "MaxAttempts": 100})


def invoke(lam, payload: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    response = lam.invoke(
        FunctionName=c.FUNCTION_NAME, Payload=json.dumps(payload), LogType="Tail", InvocationType="RequestResponse"
    )
    body = response["Payload"].read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - started) * 1000
    log_text = base64.b64decode(response.get("LogResult", "")).decode("utf-8", errors="replace")
    return {"status": response["StatusCode"], "function_error": response.get("FunctionError"), "body": body, "log": log_text}, elapsed_ms


def build_row(index: int, kind: str, result: dict, elapsed_ms: float) -> dict:
    row = {"i": index, "kind": kind, "client_ms": round(elapsed_ms, 1), "function_error": result["function_error"] or ""}
    row.update(parse_report(result["log"]))
    try:
        data = json.loads(result["body"])
        answers = data.get("answers", {})
        row["decider_latency_ms"] = data.get("latency_ms")
        row["cold_start_load_ms"] = data.get("cold_start_load_ms")
        row["urgent_noul"] = answers.get("urgent", {}).get("noul")
        row["team"] = answers.get("team", {}).get("choice")
        row["team_confidence"] = answers.get("team", {}).get("confidence")
        row["frustration_score"] = answers.get("frustration", {}).get("score")
    except (json.JSONDecodeError, AttributeError):
        pass
    return row


def rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run(variant: str, warm: int) -> None:
    lam = lambda_client()
    set_variant(lam, variant)
    plan = [(COLD, NOUL_PAYLOAD)] + [(WARM, NOUL_PAYLOAD)] * warm + [(THREE_QUESTIONS, THREE_QUESTIONS_PAYLOAD)]
    rows: list[dict] = []
    raw_path = c.RESULTS_DIR / f"invocations_{variant}.jsonl"
    raw_path.unlink(missing_ok=True)
    for index, (kind, payload) in enumerate(plan):
        result, elapsed_ms = invoke(lam, payload)
        row = build_row(index, kind, result, elapsed_ms)
        rows.append(row)
        c.append_line(raw_path, json.dumps({"i": index, "kind": kind, "client_ms": round(elapsed_ms, 1), **result}, ensure_ascii=False))
        print(index, kind, row["client_ms"], row.get("duration_ms"), row.get("init_ms"), row.get("max_memory_mb"), row["function_error"], flush=True)
        if index == 0 and result["function_error"]:
            print("cold invocation failed:", result["body"][:500])
            break
        if kind == WARM and index == 1:
            c.write_text(c.RESULTS_DIR / f"sample_response_{variant}.json", result["body"])
        if kind == THREE_QUESTIONS:
            c.write_text(c.RESULTS_DIR / f"sample_response_three_questions_{variant}.json", result["body"])
    c.write_text(c.RESULTS_DIR / f"lambda_{variant}.csv", rows_to_csv(rows, CSV_COLUMNS))
    report_lines = [r["report"] for r in rows if r.get("report")]
    c.write_text(c.RESULTS_DIR / f"report_{variant}.txt", "\n".join(report_lines) + "\n")
    billed_gb_s = sum((r.get("billed_ms") or 0) for r in rows) / 1000 * 10
    print(f"{variant}: {len(rows)} invocations, {billed_gb_s:.1f} GB-s billed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=["fp32", "bf16"])
    parser.add_argument("--warm", type=int, default=20, help="number of sequential warm invocations")
    args = parser.parse_args()
    run(args.variant, args.warm)
