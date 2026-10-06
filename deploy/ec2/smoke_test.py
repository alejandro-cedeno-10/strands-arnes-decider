"""Smoke test for deploy/ec2/serve.py: latency and answers under concurrent clients.

Sends the DevOps three-question request sequentially, then from N concurrent clients, and checks
that every concurrent answer matches its sequential one (upstream issue #17 makes them differ
without the lock in serve.py).

Usage:
    python deploy/ec2/smoke_test.py [--url http://127.0.0.1:8099] [--clients 8] [--requests 24]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ALERTS = [
    "Pod checkout-api-7d9 was OOMKilled 4 times in 10 minutes; restarts keep failing the health check.",
    "search-api p95 latency 4.1 s; CPU saturation at 95% on all replicas; traffic spike 3x normal load.",
    "HTTP 5xx errors at 31% starting 3 minutes after deployment of release v2.14.0.",
    "orders-db replication lag 40 minutes and checksum mismatches after last night's schema migration.",
]
QUESTIONS = {
    "runbook": {"type": "choice", "instructions": "Which runbook should handle this production alert?",
                "criteria": {"restart_service": "crash, hang, OOMKilled or failing health check",
                             "scale_out": "load, CPU saturation, backlog or latency growing with traffic",
                             "rollback_deploy": "errors right after a deployment",
                             "clear_disk": "disk or volume full",
                             "escalate": "unclear cause, data, database or security problems"}},
    "destructive": {"type": "noul", "instructions": "Could fixing this alert be destructive or irreversible?"},
    "severity": {"type": "score", "instructions": "How severe is this alert for customers?",
                 "criteria": ["low", "medium", "high", "critical"]},
}


def post(url: str, alert: str) -> tuple[dict, float]:
    body = json.dumps({"state": {"alert": alert}, "questions": QUESTIONS}).encode()
    request = urllib.request.Request(f"{url}/v1/systemone", data=body, headers={"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        answers = json.loads(response.read())["answers"]
    return answers, (time.perf_counter() - started) * 1000


def summary(answers: dict) -> tuple:
    return (answers["runbook"]["choice"], round(answers["runbook"]["confidence"], 3), round(answers["destructive"]["noul"], 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--requests", type=int, default=24)
    args = parser.parse_args()

    post(args.url, ALERTS[0])
    sequential, latencies = {}, []
    for alert in ALERTS:
        answers, ms = post(args.url, alert)
        sequential[alert] = summary(answers)
        latencies.append(ms)
    jobs = [ALERTS[i % len(ALERTS)] for i in range(args.requests)]
    with ThreadPoolExecutor(args.clients) as pool:
        results = list(pool.map(lambda alert: (alert, post(args.url, alert)), jobs))
    mismatches = sum(summary(answers) != sequential[alert] for alert, (answers, _) in results)
    print(json.dumps({
        "sequential_median_ms": round(statistics.median(latencies)),
        "concurrent_clients": args.clients,
        "concurrent_requests": len(results),
        "concurrent_median_ms": round(statistics.median(ms for _, (_, ms) in results)),
        "mismatches_vs_sequential": mismatches,
        "answers": {alert[:40]: value for alert, value in sequential.items()},
    }))


if __name__ == "__main__":
    main()
