"""Local benchmarks for Strands Decider against its HTTP server (built for a consumer GPU such as an RTX 4060).

What it runs (sequential except for the issue #17 test):
  0. Starts `strands-decider serve` on 127.0.0.1 (or uses one already running via --url) and records
     load time and peak memory (process RAM + GPU VRAM via NVML / nvidia-smi).
  1. Reproduces the README/blog examples (billing 0.845 / confidence 0.768, noul 0.8277, score 1.10).
  2. Latency: 1 cold request + N warm requests with a short state (<300 tokens) and a long one
     (~1000 tokens). Median, p90, p95, min, max, both client side (HTTP + inference) and the
     server-reported latency_ms (inference + serialization only).
  3. One question vs three questions on the same state.
  4. Determinism: the same request 20 times.
  5. Spanish vs English: 30 SYNTHETIC Spanish tickets (tickets.json) asked with English questions,
     Spanish questions, and English questions without option descriptions (like the CLI).
  6. A ticket that fits none of the options, with and without an "other" option.
  7. (Only with --issue17, 8 and 16 clients) concurrent requests to detect crossed answers,
     compared against the sequential answer to the same request.

Output: results/*.csv, results/results.json and results/summary.md at the repository root.

Usage (from the repository root, with the virtualenv active):
    source env/env.sh
    pip install psutil
    python benchmarks/bench_decider.py --device cuda                 # starts its own server
    python benchmarks/bench_decider.py --url http://127.0.0.1:8099   # uses a running server
    python benchmarks/bench_decider.py --issue17                     # adds the concurrency test

The server has no authentication: it only binds to 127.0.0.1 and is stopped at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
TICKETS_FILE = HERE / "tickets.json"
CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
HIGH_CONFIDENCE = 0.9
PROBABILITY_TOLERANCE = 1e-3

SHORT = "Help! My payouts have been failing for 3 days!"
LONG = (
    "Customer support conversation (ticket #48213).\n"
    + "\n".join(
        f"[{i:02d}] Customer: I run a small online store and since last Monday the payouts to my "
        f"bank account keep failing. The dashboard shows status 'returned' for transfer {1000 + i}. "
        f"I already checked my account number and it is correct. Agent: Thanks, I am checking the "
        f"payout logs and the bank response code for transfer {1000 + i}."
        for i in range(14)
    )
    + "\nCustomer: This is the third time I write. I need the money for payroll on Friday."
)
TEAM_QUESTION = "Which team should handle this?"
URGENCY_QUESTION = "Does this convey urgency?"
FRUSTRATION_QUESTION = "How frustrated is the writer?"
FRUSTRATION_LEVELS = ["calm", "frustrated", "depressed"]
TEAMS_WITHOUT_DESCRIPTIONS = {"billing": "", "sales": "", "retail": ""}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def describe(values: list[float]) -> dict:
    return {"n": len(values), "median": round(statistics.median(values), 1), "p90": round(percentile(values, .90), 1),
            "p95": round(percentile(values, .95), 1), "min": round(min(values), 1), "max": round(max(values), 1)}


def post(url: str, body: dict, timeout: float = 600) -> tuple[dict, float]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + "/v1/systemone", data, {"content-type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        payload = {"error": error.code, "detail": error.read().decode("utf-8", "replace")}
    return payload, (time.perf_counter() - started) * 1000


def noul(instructions: str, criteria: str | None = None) -> dict:
    question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(RESULTS_DIR / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_tickets() -> dict:
    return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))


def environment() -> dict:
    info = {"os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "cpu": platform.processor() or platform.machine(), "python": platform.python_version()}
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        info["ram_total_gb"] = "psutil not installed"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    except Exception as error:  # noqa: BLE001
        info["torch"] = f"unavailable ({error})"
    try:
        from importlib.metadata import version
        info["strands_decider"] = version("strands-decider")
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("nvidia-smi"):
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True)
        info["nvidia_smi"] = query.stdout.strip()
    try:
        import fla  # noqa: F401
        info["flash_linear_attention"] = "installed"
    except ImportError:
        info["flash_linear_attention"] = ("NOT installed: Gated DeltaNet layers use the torch reference path; "
                                          "the official latency (RTX 3090, WSL2) was measured with fla")
    return info


class MemorySampler(threading.Thread):
    """Samples the server process RSS (with its children) and the GPU VRAM every 0.1 s, keeping per-phase peaks."""

    def __init__(self, pid: int | None):
        super().__init__(daemon=True)
        self.pid, self.phase, self.stop_flag = pid, "load", False
        self.peaks: dict[str, dict[str, float]] = {}

    def _rss_gb(self) -> float | None:
        try:
            import psutil
            process = psutil.Process(self.pid)
            return sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]) / 2**30
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def vram_gb() -> float | None:
        """Whole-GPU used VRAM. Prefers NVML (microseconds) and falls back to nvidia-smi."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**30
        except Exception:  # noqa: BLE001
            pass
        if not shutil.which("nvidia-smi"):
            return None
        query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
        try:
            return float(query.stdout.strip().splitlines()[0]) / 1024
        except Exception:  # noqa: BLE001
            return None

    def run(self) -> None:
        while not self.stop_flag:
            peak = self.peaks.setdefault(self.phase, {})
            for key, value in (("ram_gb", self._rss_gb() if self.pid else None), ("vram_gb", self.vram_gb())):
                if value is not None:
                    peak[key] = round(max(peak.get(key, 0.0), value), 2)
            time.sleep(0.1)


def start_server(args, log_name: str):
    """Launches `strands_decider.cli serve` on 127.0.0.1 and returns (process, start time)."""
    command = [sys.executable, "-m", "strands_decider.cli", "serve", args.checkpoint, "--host", "127.0.0.1",
               "--port", str(args.port)] + (["--device", args.device] if args.device else [])
    log = open(RESULTS_DIR / log_name, "w", encoding="utf-8")
    started = time.time()
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), started


def stop_server(process) -> None:
    """Kills the process tree (on Windows the virtualenv python can be a launcher with a child)."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        tree = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for p in tree:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(tree, timeout=30)
    for p in alive:
        p.kill()


def port_is_free(port: int) -> bool:
    import socket
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def first_request(url: str) -> dict:
    """First request after /health is OK and before anything else: the real cold latency."""
    response, ms = post(url, {"state": SHORT, "questions": {"is_urgent": noul(URGENCY_QUESTION)}})
    return {"client_ms": round(ms, 1), "server_ms": response.get("latency_ms")}


def wait_health(url: str, timeout: float) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                return json.loads(response.read())
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise TimeoutError(f"the server did not answer /health within {timeout} s")


def reproduce_readme(url: str) -> dict:
    """Same requests the CLI builds (choice with empty option descriptions)."""
    cli_state = "Help! My payouts have been failing for 3 days! "
    body = {"state": cli_state, "questions": {
        "choice_0": choice(TEAM_QUESTION, TEAMS_WITHOUT_DESCRIPTIONS),
        "noul_0": noul(URGENCY_QUESTION),
        "score_0": score(FRUSTRATION_QUESTION, FRUSTRATION_LEVELS)}}
    combined, _ = post(url, body)
    separate = {key: post(url, {"state": cli_state, "questions": {key: q}})[0] for key, q in body["questions"].items()}
    curl, _ = post(url, {"state": SHORT, "questions": {"is_urgent": noul(URGENCY_QUESTION)}})
    return {"published": {"choice": "billing 0.845 / retail 0.091 / sales 0.064, confidence 0.768",
                          "noul": "0.828 (CLI) / 0.8277 (curl)", "score": "1.10, confidence 0.518"},
            "measured_separately": separate, "measured_combined": combined, "measured_curl": curl}


def latency(url: str, n: int, sampler: MemorySampler) -> tuple[dict, list[dict]]:
    rows, summary = [], {}
    questions = {"is_urgent": noul(URGENCY_QUESTION)}
    sampler.phase = "inference"
    for label, state in (("short", SHORT), ("long", LONG)):
        client, server = [], []
        for i in range(n + 1):
            response, ms = post(url, {"state": state, "questions": questions})
            rows.append({"state": label, "i": i, "cold": i == 0, "client_ms": round(ms, 2),
                         "server_ms": response.get("latency_ms"),
                         "input_tokens": response.get("usage", {}).get("input_tokens")})
            if i == 0:
                summary[f"{label}_cold_ms"] = round(ms, 1)
            else:
                client.append(ms)
                server.append(response.get("latency_ms", float("nan")))
        summary[label] = {"client": describe(client), "server": describe(server), "input_tokens": rows[-1]["input_tokens"]}
    return summary, rows


def multi_question(url: str, n: int = 30) -> tuple[dict, list[dict]]:
    """Alternates one-question and three-question requests so thermal drift affects both equally."""
    one = {"team": choice(TEAM_QUESTION, TEAMS_WITHOUT_DESCRIPTIONS)}
    three = {**one, "urgent": noul(URGENCY_QUESTION), "frustration": score(FRUSTRATION_QUESTION, FRUSTRATION_LEVELS)}
    rows = []
    for state_label, state in (("short", SHORT), ("long", LONG)):
        for i in range(n):
            for count, questions in (("1", one), ("3", three)):
                response, ms = post(url, {"state": state, "questions": questions})
                rows.append({"state": state_label, "questions": count, "i": i, "client_ms": round(ms, 2),
                             "server_ms": response.get("latency_ms")})
    summary = {}
    for state_label in ("short", "long"):
        for count in ("1", "3"):
            values = [r["client_ms"] for r in rows if r["state"] == state_label and r["questions"] == count]
            summary[f"{state_label}_{count}q"] = describe(values)
    return summary, rows


DETERMINISM_BODY = {"state": SHORT, "questions": {
    "team": choice(TEAM_QUESTION, TEAMS_WITHOUT_DESCRIPTIONS),
    "urgent": noul(URGENCY_QUESTION)}}


def determinism(url: str, n: int = 20) -> dict:
    seen = [json.dumps(post(url, DETERMINISM_BODY)[0]["answers"], sort_keys=True) for _ in range(n)]
    return {"repetitions": n, "distinct_answers": len(set(seen)), "example": json.loads(seen[0])}


def determinism_after_restart(url: str, reference: dict, n: int = 5) -> dict:
    """Same request after restarting the server, compared with the reference answer from before."""
    answers = [post(url, DETERMINISM_BODY)[0]["answers"] for _ in range(n)]
    expected = json.dumps(reference, sort_keys=True)
    identical = sum(json.dumps(a, sort_keys=True) == expected for a in answers)
    max_choice_delta = max(abs(a["team"]["probabilities"][k] - reference["team"]["probabilities"][k])
                           for a in answers for k in reference["team"]["probabilities"])
    max_noul_delta = max(abs(a["urgent"]["noul"] - reference["urgent"]["noul"]) for a in answers)
    return {"repetitions_after_restart": n, "identical_to_reference": identical,
            "max_choice_probability_delta": round(max_choice_delta, 6), "max_noul_delta": round(max_noul_delta, 6),
            "answer": answers[0]}


def spanish_vs_english(url: str) -> tuple[dict, list[dict]]:
    data = load_tickets()
    variants = {
        "en": (data["questions"]["en"], data["labels"]["en"]),
        "es": (data["questions"]["es"], data["labels"]["es"]),
        "en_without_descriptions": (data["questions"]["en"], {k: "" for k in data["labels"]["en"]}),
    }
    rows, summary = [], {}
    labels = list(data["labels"]["en"])
    total = len(data["tickets"])
    for variant, (question, criteria) in variants.items():
        hits, confidences, high, high_correct = 0, [], 0, 0
        confusion = {expected: {chosen: 0 for chosen in labels} for expected in labels}
        for ticket in data["tickets"]:
            response, ms = post(url, {"state": ticket["text"], "questions": {"team": choice(question, criteria)}})
            answer = response["answers"]["team"]
            correct = answer["choice"] == ticket["label"]
            confusion[ticket["label"]][answer["choice"]] += 1
            hits += correct
            confidences.append(answer["confidence"])
            if answer["confidence"] >= HIGH_CONFIDENCE:
                high += 1
                high_correct += correct
            rows.append({"variant": variant, "id": ticket["id"], "expected": ticket["label"], "chosen": answer["choice"],
                         "correct": correct, "confidence": answer["confidence"],
                         "p_chosen": answer["probabilities"][answer["choice"]], "client_ms": round(ms, 1)})
        summary[variant] = {"hits": f"{hits}/{total}", "accuracy": round(hits / total, 3),
                            "mean_confidence": round(statistics.mean(confidences), 3),
                            "confidence_ge_0.9": high, "confidence_ge_0.9_correct": high_correct,
                            "confusion_matrix": confusion}
    return summary, rows


def out_of_options(url: str) -> dict:
    data = load_tickets()
    state = data["out_of_options"]["text"]
    english = data["labels"]["en"]
    three = {k: english[k] for k in ("billing", "sales", "retail")}
    unclear = {**english, "unclear": "the ticket is ambiguous or incomplete and a person must clarify it first"}
    answers = {}
    for label, criteria in (("without_other", three), ("with_other", english), ("with_other_and_unclear", unclear)):
        response, _ = post(url, {"state": state, "questions": {"team": choice(data["questions"]["en"], criteria)}})
        answers[label] = response["answers"]["team"]
    return {"ticket": state, **answers}


def build_issue17_requests(workers: int) -> list[dict]:
    """One distinct request per client: different state, question and option lengths."""
    requests = []
    for i in range(workers):
        options = {f"team_{i}_{j}": "x" * (j * (i + 1)) for j in range(2 + i % 4)}
        requests.append({"state": f"Ticket {i}: " + ("payout failed. " * (1 + 3 * i)),
                         "questions": {"q": choice(TEAM_QUESTION + " " + "Be careful. " * i, options)}})
    return requests


def issue17(url: str, workers: int = 8, rounds: int = 5) -> tuple[dict, list[dict]]:
    """Compares concurrent answers with the sequential answers to the SAME requests."""
    requests = build_issue17_requests(workers)
    sequential = [post(url, body)[0] for body in requests]
    sequential_errors = sum("error" in answer for answer in sequential)
    changed_choice = changed_probabilities = http_422 = other_errors = total = 0
    max_delta, examples, rows = 0.0, [], []
    for round_index in range(rounds):
        with ThreadPoolExecutor(workers) as pool:
            concurrent = list(pool.map(lambda body: post(url, body), requests))
        for i, (expected, (got, ms)) in enumerate(zip(sequential, concurrent)):
            total += 1
            row = {"workers": workers, "round": round_index, "client": i, "client_ms": round(ms, 1),
                   "http_error": got.get("error", ""), "choice_changed": "", "max_abs_dp": ""}
            if "error" in got:
                http_422 += got["error"] == 422
                other_errors += got["error"] != 422
                examples.append({"error": got})
            elif "error" not in expected:
                expected_p, got_p = expected["answers"]["q"]["probabilities"], got["answers"]["q"]["probabilities"]
                delta = max(abs(expected_p[k] - got_p.get(k, -1)) for k in expected_p)
                max_delta = max(max_delta, delta)
                changed = expected["answers"]["q"]["choice"] != got["answers"]["q"]["choice"]
                changed_choice += changed
                changed_probabilities += delta > PROBABILITY_TOLERANCE
                row.update(choice_changed=changed, max_abs_dp=round(delta, 6))
                if (changed or delta > PROBABILITY_TOLERANCE) and len(examples) < 3:
                    examples.append({"client": i, "sequential": expected_p, "concurrent": got_p})
            rows.append(row)
    return {"workers": workers, "rounds": rounds, "requests": total, "sequential_with_error": sequential_errors,
            "answers_with_different_choice": changed_choice,
            "answers_with_probability_delta_gt_1e-3": changed_probabilities,
            "max_abs_dp_vs_sequential": round(max_delta, 6), "http_422": http_422, "other_http_errors": other_errors,
            "examples": examples}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--url", help="use a running server (load time and process RAM are not measured)")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (auto-detected by default)")
    parser.add_argument("--n", type=int, default=100, help="warm requests per state")
    parser.add_argument("--issue17", action="store_true", help="include the concurrency test")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "environment": environment()}
    process = None
    url = args.url
    started = time.time()
    if not url:
        url = f"http://127.0.0.1:{args.port}"
        if not port_is_free(args.port):
            sys.exit(f"port {args.port} is already in use: not starting another server")
        results["vram_before_start_gb"] = round(MemorySampler.vram_gb() or float("nan"), 2)
        process, started = start_server(args, "server.log")
    sampler = MemorySampler(process.pid if process else None)
    sampler.start()
    try:
        results["health"] = wait_health(url, 900)
        if process:
            results["load_s"] = round(time.time() - started, 1)
        print("server ready:", results["health"])
        sampler.phase = "inference"
        results["first_request_after_start"] = first_request(url)

        print("1/7 reproducing README/blog ...")
        results["readme"] = reproduce_readme(url)
        print("2/7 latency ...")
        results["latency"], rows = latency(url, args.n, sampler)
        write_csv("latency.csv", rows)
        print("3/7 one vs three questions ...")
        results["multi_question"], rows = multi_question(url)
        write_csv("multi_question.csv", rows)
        print("4/7 determinism ...")
        results["determinism"] = determinism(url)
        print("5/7 Spanish vs English (30 synthetic tickets) ...")
        results["es_vs_en"], rows = spanish_vs_english(url)
        write_csv("es_vs_en.csv", rows)
        print("6/7 out-of-options ticket ...")
        results["out_of_options"] = out_of_options(url)
        if args.issue17:
            print("7/7 issue #17 (concurrency) ...")
            sampler.phase = "concurrency"
            results["issue17"], issue17_rows = {}, []
            for workers in (8, 16):
                results["issue17"][str(workers)], worker_rows = issue17(url, workers=workers)
                issue17_rows += worker_rows
            write_csv("issue17.csv", issue17_rows)
        else:
            results["issue17"] = "not run (use --issue17)"
        if process:
            print("8/8 determinism after restarting the server ...")
            reference = results["determinism"]["example"]
            sampler.stop_flag = True
            sampler.join(timeout=2)
            results["peak_memory"] = sampler.peaks
            stop_server(process)
            time.sleep(2)
            results["port_free_after_first_stop"] = port_is_free(args.port)
            process, restarted = start_server(args, "server_restart.log")
            sampler = MemorySampler(process.pid)
            sampler.phase = "reload"
            sampler.start()
            wait_health(url, 900)
            results["reload_s"] = round(time.time() - restarted, 1)
            sampler.phase = "inference_after_restart"
            results["first_request_after_restart"] = first_request(url)
            results["determinism_after_restart"] = determinism_after_restart(url, reference)
            results["peak_memory_after_restart"] = sampler.peaks
    except Exception as error:  # noqa: BLE001
        import traceback
        results["run_error"] = traceback.format_exc()
        print("FAILED:", error)
    finally:
        sampler.stop_flag = True
        sampler.join(timeout=2)
        results.setdefault("peak_memory", sampler.peaks)
        if process:
            stop_server(process)
            time.sleep(2)
            results["port_free_at_end"] = port_is_free(args.port)
        (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        write_summary(results, args)
        print("done: results/results.json and results/summary.md")


def md_table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def environment_section(r: dict) -> list[str]:
    lines = ["## Environment", ""] + [f"- {k}: {v}" for k, v in r["environment"].items()]
    if "health" in r:
        h = r["health"]
        lines += [f"- server: device={h.get('device')}; prefix_cache={h.get('prefix_cache')}; max_length={h.get('max_length')}; "
                  f"num_slots={h.get('num_slots')}; base_model={h.get('base_model')}; temperature={h.get('temperature')}"]
    if "load_s" in r:
        lines += [f"- server start until /health OK: {r['load_s']} s"
                  + (f" (second start, after restart: {r['reload_s']} s)" if "reload_s" in r else "")]
    for key, label in (("first_request_after_start", "first request after start (real cold)"),
                       ("first_request_after_restart", "first request after restart (real cold)")):
        if key in r:
            lines += [f"- {label}: client {r[key]['client_ms']} ms, server `latency_ms` {r[key]['server_ms']} ms"]
    return lines + [""]


def memory_section(r: dict) -> list[str]:
    lines = ["## Peak memory", ""]
    baseline = r.get("vram_before_start_gb")
    if baseline is not None:
        lines += [f"Whole-GPU VRAM before starting the server (desktop and other processes): **{baseline} GB**. "
                  "The VRAM below is for the whole GPU (NVML), not only the server; the difference with this value "
                  "is an upper bound of what Decider uses.", ""]
    phases = {**r.get("peak_memory", {}),
              **{f"{k} (2nd start)": v for k, v in r.get("peak_memory_after_restart", {}).items()}}
    rows = []
    for phase, values in phases.items():
        delta = round(values["vram_gb"] - baseline, 2) if baseline is not None and "vram_gb" in values else ""
        rows.append([phase, values.get("ram_gb", ""), values.get("vram_gb", ""), delta])
    lines += md_table(["phase", "process peak RAM (GB)", "GPU peak VRAM (GB)", "peak VRAM minus baseline (GB)"], rows)
    return lines + ["", "RAM = RSS of the server process and its children, sampled every 0.1 s.", ""]


def latency_section(r: dict, n: int) -> list[str]:
    lines = ["## Latency (one noul question, sequential)", "",
             f"Warm samples per state: {n}. With N={n}, p95 interpolates between the 5th and 6th highest values and "
             "p90 between the 10th and 11th: they sit near the extreme and move with a single slow request. "
             "They are not production percentiles.", "",
             "The real cold latency (first request after start) is in the Environment section. The first-request "
             "column below was measured after other requests, so it is no longer cold.", "",
             "Client = HTTP + inference measured by this script (urllib, 127.0.0.1). Server = `latency_ms` from the "
             "response (engine only, no HTTP or serialization).", ""]
    rows = []
    for state in ("short", "long"):
        data = r["latency"][state]
        for side in ("client", "server"):
            s = data[side]
            rows.append([state, data["input_tokens"], side, r["latency"][state + "_cold_ms"] if side == "client" else "",
                         s["median"], s["p90"], s["p95"], s["min"], s["max"]])
    lines += md_table(["state", "input_tokens", "measured at", "first request of that state, engine already warm (ms)",
                       "median (ms)", "p90", "p95", "min", "max"], rows)
    return lines + [""]


def determinism_section(r: dict) -> list[str]:
    d = r["determinism"]
    lines = ["## Determinism", "", f"- Same request (1 choice + 1 noul) {d['repetitions']} times in a row: distinct answers = "
             f"**{d['distinct_answers']}** (1 = identical)."]
    if "determinism_after_restart" in r:
        t = r["determinism_after_restart"]
        lines += [f"- After stopping and restarting the server: {t['identical_to_reference']}/{t['repetitions_after_restart']} "
                  f"identical to the answer before the restart; max probability delta in choice "
                  f"{t['max_choice_probability_delta']}, in noul {t['max_noul_delta']}."]
    return lines + ["- Reference answer:", "", "```json", json.dumps(d["example"], indent=2), "```", ""]


def language_section(r: dict) -> list[str]:
    lines = ["## Spanish vs English (30 SYNTHETIC tickets: 8 billing, 7 sales, 7 retail, 8 other; 30 cases do not generalize)", ""]
    rows = [[v, x["hits"], x["accuracy"], x["mean_confidence"], x["confidence_ge_0.9"], x["confidence_ge_0.9_correct"]]
            for v, x in r["es_vs_en"].items()]
    lines += md_table(["variant", "hits", "accuracy", "mean confidence", "confidence >= 0.9", "of those, correct"], rows) + [""]
    for variant, x in r["es_vs_en"].items():
        labels = list(x["confusion_matrix"])
        lines += [f"Confusion matrix `{variant}` (rows = expected, columns = chosen):", ""]
        lines += md_table(["expected \\ chosen"] + labels,
                          [[e] + [x["confusion_matrix"][e][c] for c in labels] for e in labels]) + [""]
    return lines


def issue17_section(r: dict) -> list[str]:
    lines = ["## Issue #17 (concurrency against 127.0.0.1)", ""]
    if not isinstance(r["issue17"], dict):
        return lines + [str(r["issue17"]), ""]
    rows = [[w, v["requests"], v["answers_with_different_choice"], v["answers_with_probability_delta_gt_1e-3"],
             v["max_abs_dp_vs_sequential"], v["http_422"], v["other_http_errors"], v["sequential_with_error"]]
            for w, v in r["issue17"].items()]
    lines += md_table(["concurrent clients", "requests", "different choice", "probability delta (> 1e-3)",
                       "max abs(dp) vs sequential", "422", "500 or other", "sequential with error"], rows)
    return lines + ["", "Each round sends one distinct request per client (state, question and options of different "
                    "lengths) and compares it with the sequential answer to the same request. Details in `issue17.csv`.", "",
                    "```json", json.dumps(r["issue17"], indent=2, ensure_ascii=False), "```", ""]


def write_summary(r: dict, args) -> None:
    lines = ["# Strands Decider benchmark summary", "",
             f"Date: {r['date']}. Generated by `benchmarks/bench_decider.py` (raw data: `results.json` and the CSV files).", ""]
    if "run_error" in r:
        lines += ["> **The run failed partway.** Anything missing was not measured. Traceback:", "", "```",
                  r["run_error"], "```", ""]
    lines += environment_section(r) + memory_section(r)
    if "latency" in r:
        lines += latency_section(r, args.n)
    if "multi_question" in r:
        lines += ["## One vs three questions (same state, 30 alternating repetitions, client latency in ms)", ""]
        rows = [[k, v["n"], v["median"], v["p90"], v["p95"], v["min"], v["max"]] for k, v in r["multi_question"].items()]
        lines += md_table(["case", "n", "median", "p90", "p95", "min", "max"], rows) + [""]
    if "determinism" in r:
        lines += determinism_section(r)
    if "es_vs_en" in r:
        lines += language_section(r)
    for key, title in (("readme", "README/blog reproduction"),
                       ("out_of_options", "Out-of-options ticket (with and without other/unclear)")):
        if key in r:
            lines += [f"## {title}", "", "```json", json.dumps(r[key], indent=2, ensure_ascii=False), "```", ""]
    if "issue17" in r:
        lines += issue17_section(r)
    lines += ["## Shutdown", "", f"- port free after the first stop: {r.get('port_free_after_first_stop', 'n/a')}",
              f"- port free at the end: {r.get('port_free_at_end', 'n/a')}", ""]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
