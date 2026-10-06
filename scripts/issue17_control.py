"""Controls for the issue #17 concurrency test, against a local Decider server on 127.0.0.1.

1. Server as shipped by the package: a sequential control (the same requests, 3 rounds one at a time,
   must yield 0 differences) and concurrency with 2, 4, 8 and 16 clients.
2. Server with a lock around `SystemOneEngine.evaluate` (only inside this test process; the package is
   not modified): tests the hypothesis that `self._last_offsets`, state shared between `_fit` and
   `_option_idx`, is the cause. With 8 and 16 clients.

Usage:
    python scripts/issue17_control.py                      (starts and stops both servers)
    python scripts/issue17_control.py --serve-locked PORT  (internal mode: locked server)
Output: results/issue17_control.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import bench_decider as bench  # noqa: E402

PORT = 8099
CONCURRENT_ROUNDS = 5
SEQUENTIAL_ROUNDS = 3
PROBABILITY_TOLERANCE = 1e-3


def build_requests(workers: int) -> list[dict]:
    requests = []
    for i in range(workers):
        options = {f"team_{i}_{j}": "x" * (j * (i + 1)) for j in range(2 + i % 4)}
        requests.append({"state": f"Ticket {i}: " + ("payout failed. " * (1 + 3 * i)),
                         "questions": {"q": bench.choice("Which team should handle this? " + "Be careful. " * i, options)}})
    return requests


def compare(baseline: list[dict], got: list[dict]) -> tuple[int, int, int, float]:
    """Return (different choice, probability drift above tolerance, errors, max absolute drift)."""
    changed, drifted, errors, max_drift = 0, 0, 0, 0.0
    for base, answer in zip(baseline, got):
        if "error" in answer:
            errors += 1
            continue
        base_probs, probs = base["answers"]["q"]["probabilities"], answer["answers"]["q"]["probabilities"]
        drift = max(abs(base_probs[k] - probs.get(k, -1)) for k in base_probs)
        max_drift = max(max_drift, drift)
        changed += base["answers"]["q"]["choice"] != answer["answers"]["q"]["choice"]
        drifted += drift > PROBABILITY_TOLERANCE
    return changed, drifted, errors, max_drift


def run_rounds(url: str, workers: int, rounds: int, concurrent: bool) -> dict:
    requests = build_requests(workers)
    baseline = [bench.post(url, r)[0] for r in requests]
    total = changed = drifted = errors = 0
    max_drift = 0.0
    for _ in range(rounds):
        if concurrent:
            with ThreadPoolExecutor(workers) as executor:
                got = list(executor.map(lambda r: bench.post(url, r)[0], requests))
        else:
            got = [bench.post(url, r)[0] for r in requests]
        round_changed, round_drifted, round_errors, round_drift = compare(baseline, got)
        total += len(requests)
        changed += round_changed
        drifted += round_drifted
        errors += round_errors
        max_drift = max(max_drift, round_drift)
    return {"requests": total, "different_choice": changed, "probability_drift_gt_1e-3": drifted,
            "http_422_or_other_error": errors, "max_abs_probability_drift": round(max_drift, 6)}


def run_concurrent(url: str, workers: int) -> dict:
    return {"clients": workers, **run_rounds(url, workers, CONCURRENT_ROUNDS, concurrent=True)}


def run_sequential_control(url: str, workers: int = 16) -> dict:
    return {"mode": "sequential (control)", **run_rounds(url, workers, SEQUENTIAL_ROUNDS, concurrent=False)}


def serve_locked(port: int) -> None:
    """Serve Decider with a process-local lock around evaluate, without modifying the package."""
    import threading

    from strands_decider import infer, server

    lock = threading.Lock()
    original = infer.SystemOneEngine.evaluate

    def locked(self, request):
        with lock:
            return original(self, request)

    infer.SystemOneEngine.evaluate = locked
    server.serve(bench.CHECKPOINT, host="127.0.0.1", port=port, device="cuda")


def with_server(extra_args: list[str], log_name: str, measure):
    process = None
    try:
        if extra_args:
            command = [sys.executable, str(Path(__file__).resolve()), *extra_args]
            log = open(bench.RESULTS_DIR / log_name, "w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        else:
            args = types.SimpleNamespace(checkpoint=bench.CHECKPOINT, port=PORT, device="cuda")
            process, _ = bench.start_server(args, log_name)
        url = f"http://127.0.0.1:{PORT}"
        bench.wait_health(url, 600)
        return measure(url)
    finally:
        if process:
            bench.stop_server(process)
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve-locked", type=int, help="internal: serve with a lock on this port")
    args = parser.parse_args()
    if args.serve_locked:
        serve_locked(args.serve_locked)
        return
    if not bench.port_is_free(PORT):
        sys.exit(f"port {PORT} is busy")
    bench.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output: dict = {"date": time.strftime("%Y-%m-%d %H:%M")}
    output["original_server"] = with_server([], "server_issue17_control.log", lambda url: {
        "sequential_control": run_sequential_control(url),
        "concurrent": [run_concurrent(url, w) for w in (2, 4, 8, 16)]})
    output["locked_server"] = with_server(["--serve-locked", str(PORT)], "server_issue17_locked.log", lambda url: {
        "concurrent": [run_concurrent(url, w) for w in (8, 16)]})
    output["port_free_at_end"] = bench.port_is_free(PORT)
    (bench.RESULTS_DIR / "issue17_control.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
