"""Compare CPU inference variants of Strands Decider: fp32, bf16 and int8 dynamic quantization.

For each variant and thread count it reports the warm latency of a one-question and a
three-question request, the resident memory, and whether the answers agree with fp32
(same choice, and probabilities within a tolerance). Intended for an arm64 (Graviton) host.

Run one variant per process so peak memory is per variant, fp32 first (it is the reference):
    python benchmarks/cpu_variants.py --variant fp32 --threads 2 4 8
    python benchmarks/cpu_variants.py --variant bf16 --threads 2 4 8
    python benchmarks/cpu_variants.py --variant int8 --threads 2 4 8

Writes results/cpu_variants_<variant>.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

CHECKPOINT = os.getenv("DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STATE = "Help! My payouts have been failing for 3 days!"
ONE_QUESTION = {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
THREE_QUESTIONS = {
    **ONE_QUESTION,
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
}
PROBABILITY_TOLERANCE = 0.05


def build_engine(variant: str):
    """Load the engine on CPU in the requested precision."""
    from strands_decider.infer import SystemOneEngine, load_engine

    if variant == "bf16":
        SystemOneEngine._upcast_torso_for_cpu = lambda self: None  # type: ignore[method-assign]
    engine = load_engine(CHECKPOINT, device="cpu")
    if variant == "int8":
        if "qnnpack" in torch.backends.quantized.supported_engines and platform.machine() in ("aarch64", "arm64"):
            torch.backends.quantized.engine = "qnnpack"
        with torch.inference_mode():
            torso = engine.model.torso
            if hasattr(torso, "merge_and_unload"):
                torso = torso.merge_and_unload()
            engine.model.torso = torch.ao.quantization.quantize_dynamic(
                torso, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        gc.collect()
    return engine


def ask(engine, questions: dict) -> tuple[dict, float]:
    from strands_decider.schema import SystemOneRequest

    request = SystemOneRequest.model_validate({"state": STATE, "questions": questions})
    started = time.perf_counter()
    response = engine.evaluate(request).model_dump()
    return response["answers"], (time.perf_counter() - started) * 1000


def summarize_answers(answers: dict) -> dict:
    """Keep the fields used to compare variants: noul probability, chosen option, expected score."""
    summary = {}
    for name, answer in answers.items():
        summary[name] = {key: answer.get(key) for key in ("noul", "choice", "confidence", "score") if key in answer}
    return summary


def agrees(reference: dict, candidate: dict) -> bool:
    for name, ref in reference.items():
        cand = candidate.get(name, {})
        if ref.get("choice") != cand.get("choice"):
            return False
        for key in ("noul", "confidence"):
            if key in ref and abs(float(ref[key]) - float(cand.get(key, -1))) > PROBABILITY_TOLERANCE:
                return False
    return True


def measure(engine, questions: dict, warm: int) -> tuple[dict, list[float]]:
    answers, _ = ask(engine, questions)
    latencies = [ask(engine, questions)[1] for _ in range(warm)]
    return summarize_answers(answers), latencies


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Linux and macOS; 0 where unavailable)."""
    try:
        import resource
    except ImportError:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_reference() -> dict:
    path = RESULTS_DIR / "cpu_variants_fp32.json"
    if not path.exists():
        return {}
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    return {run["request"]: run["answers"] for run in runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["fp32", "bf16", "int8"], required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[os.cpu_count() or 1])
    parser.add_argument("--warm", type=int, default=10)
    args = parser.parse_args()

    reference = load_reference()
    load_started = time.perf_counter()
    engine = build_engine(args.variant)
    load_s = time.perf_counter() - load_started
    report = {"host": {"machine": platform.machine(), "cpus": os.cpu_count(), "torch": torch.__version__},
              "variant": args.variant, "load_s": round(load_s, 1), "runs": []}
    for threads in args.threads:
        torch.set_num_threads(threads)
        for label, questions in (("one_question", ONE_QUESTION), ("three_questions", THREE_QUESTIONS)):
            answers, latencies = measure(engine, questions, args.warm)
            row = {
                "variant": args.variant,
                "threads": threads,
                "request": label,
                "median_ms": round(statistics.median(latencies), 1),
                "max_ms": round(max(latencies), 1),
                "warm_n": len(latencies),
                "peak_rss_mb": round(peak_rss_mb()),
                "answers": answers,
                "agrees_with_fp32": agrees(reference[label], answers) if label in reference else None,
            }
            report["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "answers"}), flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"cpu_variants_{args.variant}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
