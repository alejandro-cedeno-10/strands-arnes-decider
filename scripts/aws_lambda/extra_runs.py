"""Extra measurements: a second fp32 cold start and a bf16 memory sweep.

    python scripts/aws_lambda/extra_runs.py --rebuild-fp32-from <invocations.jsonl>
    python scripts/aws_lambda/extra_runs.py --sweep [--memory 4096 6144 8192] [--threads 2]

Writes results/aws_lambda/lambda_extra_*.csv (one row per invocation, with the configured memory) and
results/aws_lambda/invocations_extra_*.jsonl. When it finishes, the function is set back to 10,240 MB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as c
import measure as m

MEMORY_SWEEP_MB = [4096, 6144, 8192]
WARM_PER_CONFIG = 3
DEFAULT_MEMORY_MB = 10240
EXTRA_COLUMNS = ["label", "variant", "configured_memory_mb"] + m.CSV_COLUMNS


def configure(lam, variant: str, memory_mb: int, marker: str, threads: int | None = None) -> None:
    variables = {"DECIDER_KEEP_BF16": "1" if variant == "bf16" else "0", "RUN_MARKER": marker}
    if threads:
        variables["TORCH_NUM_THREADS"] = str(threads)
    lam.update_function_configuration(
        FunctionName=c.FUNCTION_NAME,
        MemorySize=memory_mb,
        Environment={"Variables": variables},
    )
    lam.get_waiter("function_updated_v2").wait(FunctionName=c.FUNCTION_NAME, WaiterConfig={"Delay": 3, "MaxAttempts": 100})


def measure_config(lam, label: str, variant: str, memory_mb: int, warm: int, raw_path: Path, rows: list[dict],
                   threads: int | None = None) -> None:
    configure(lam, variant, memory_mb, label, threads)
    plan = [(m.COLD, m.NOUL_PAYLOAD)] + [(m.WARM, m.NOUL_PAYLOAD)] * warm
    for index, (kind, payload) in enumerate(plan):
        result, elapsed_ms = m.invoke(lam, payload)
        row = m.build_row(index, kind, result, elapsed_ms)
        row.update({"label": label, "variant": variant, "configured_memory_mb": memory_mb})
        rows.append(row)
        c.append_line(raw_path, json.dumps({"label": label, "i": index, "kind": kind, "client_ms": round(elapsed_ms, 1), **result},
                                           ensure_ascii=False))
        print(label, index, kind, row["client_ms"], row.get("duration_ms"), row.get("max_memory_mb"), row["function_error"], flush=True)
        if index == 0 and result["function_error"]:
            break


def write_csv(rows: list[dict], name: str) -> None:
    c.write_text(c.RESULTS_DIR / name, m.rows_to_csv(rows, EXTRA_COLUMNS))


def rebuild_from_jsonl(path: Path, label: str, variant: str, memory_mb: int) -> list[dict]:
    """Rebuild the CSV rows of an aborted run from its raw JSONL output."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("label", label) == label:
            row = m.build_row(raw["i"], raw["kind"], raw, raw["client_ms"])
            row.update({"label": label, "variant": variant, "configured_memory_mb": memory_mb})
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-fp32-from", help="raw JSONL of a previous fp32 run to rebuild as a second cold start")
    parser.add_argument("--sweep", action="store_true", help="run the bf16 memory sweep")
    parser.add_argument("--threads", type=int, help="set TORCH_NUM_THREADS in the function")
    parser.add_argument("--memory", type=int, nargs="*", default=MEMORY_SWEEP_MB, help="memory sizes in MB")
    args = parser.parse_args()
    if args.rebuild_fp32_from:
        rows = rebuild_from_jsonl(Path(args.rebuild_fp32_from), f"fp32_second_cold_{DEFAULT_MEMORY_MB}", "fp32", DEFAULT_MEMORY_MB)
        write_csv(rows, "lambda_extra_fp32.csv")
    if not args.sweep:
        return
    lam = m.lambda_client()
    rows: list[dict] = []
    suffix = f"_threads{args.threads}" if args.threads else ""
    raw_path = c.RESULTS_DIR / f"invocations_extra_sweep{suffix}_{args.memory[0]}.jsonl"
    raw_path.unlink(missing_ok=True)
    for memory_mb in args.memory:
        measure_config(lam, f"bf16_{memory_mb}{suffix}", "bf16", memory_mb, WARM_PER_CONFIG, raw_path, rows, args.threads)
        write_csv(rows, f"lambda_extra_sweep{suffix}_{args.memory[0]}.csv")
    configure(lam, "bf16", DEFAULT_MEMORY_MB, "final")


if __name__ == "__main__":
    main()
