"""Build results/aws_lambda/summary_lambda.md and lambda_spend.json from the measured CSV files.

Reads results/aws_lambda/lambda_{fp32,bf16}.csv (whichever exist) and lambda_extra_*.csv. The
"Reading" section is computed from that data; it contains no hardcoded figures. With DECIDER_RUN_TAG
(see common.py) it reads and writes a separate subfolder.

    python scripts/aws_lambda/summarize.py
"""

from __future__ import annotations

import csv
import json
import statistics

import common as c
import measure as m

FREE_TIER_GB_SECONDS = 400_000
FREE_TIER_REQUESTS = 1_000_000
PRICE_PER_GB_SECOND_ARM = 0.0000133334
PRICE_PER_REQUEST = 0.20 / 1_000_000
LIMIT_MB = 10240
TIMEOUT_MS = 900_000


def read_rows(path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict], memory_mb: float) -> dict:
    warm = [r for r in rows if r["kind"] == m.WARM]
    cold = [r for r in rows if r["kind"] == m.COLD]
    three = [r for r in rows if r["kind"] == m.THREE_QUESTIONS]
    durations = [number(r, "duration_ms") for r in warm]
    billed = [number(r, "billed_ms") for r in warm]
    clients = [number(r, "client_ms") for r in warm]
    gb_s_per_decision = statistics.median(billed) / 1000 * memory_mb / 1024
    return {
        "warm_n": len(warm),
        "duration_median": statistics.median(durations),
        "duration_p90": percentile(durations, 0.9),
        "duration_max": max(durations),
        "client_median": statistics.median(clients),
        "billed_median": statistics.median(billed),
        "cold_duration": number(cold[0], "duration_ms") if cold else None,
        "cold_init": number(cold[0], "init_ms") if cold else None,
        "cold_client": number(cold[0], "client_ms") if cold else None,
        "three_duration": number(three[0], "duration_ms") if three else None,
        "three_billed": number(three[0], "billed_ms") if three else None,
        "three_gb_s": number(three[0], "billed_ms") / 1000 * memory_mb / 1024 if three else None,
        "max_memory": max(number(r, "max_memory_mb") or 0 for r in rows),
        "gb_s_per_decision": gb_s_per_decision,
        "cost_per_decision": gb_s_per_decision * PRICE_PER_GB_SECOND_ARM + PRICE_PER_REQUEST,
        "free_decisions": min(FREE_TIER_GB_SECONDS / gb_s_per_decision, FREE_TIER_REQUESTS),
    }


def reading_lines(stats: dict, extra_stats: dict) -> list[str]:
    lines: list[str] = []
    fp32, bf16 = stats.get("fp32"), stats.get("bf16")
    for variant, s in stats.items():
        verdict = "fits" if s["max_memory"] < LIMIT_MB else "does NOT fit"
        lines.append(f"- **{variant} {verdict} in {LIMIT_MB:,} MB**: peak {s['max_memory']:,.0f} MB ({s['max_memory'] / LIMIT_MB:.1%}).")
    colds = ", ".join(f"{v} {s['cold_duration'] / 1000:.1f} s ({s['cold_duration'] / 60000:.1f} min, {s['cold_duration'] / TIMEOUT_MS:.0%} of the 900 s timeout)"
                      for v, s in stats.items() if s["cold_duration"])
    extra_colds = ", ".join(f"{label} ({memory:,.0f} MB) {cold / 1000:.1f} s" for label, (memory, cold, _, _) in extra_stats.items() if cold)
    lines.append(f"- **Cold start**: first cold start after deploying: {colds}. Cold starts from the extra runs (different memory or threads, "
                 f"not second cold starts of the same configuration): {extra_colds}. Treat it as `tens of seconds to several minutes`, not as one number. "
                 "The CloudWatch log of each cold start shows where the time goes; this data does not isolate the cause.")
    if fp32 and bf16:
        slower = bf16["duration_median"] / fp32["duration_median"] - 1
        lines.append(f"- **bf16 vs fp32 on Lambda**: {slower:.0%} slower ({bf16['duration_median']:,.0f} vs {fp32['duration_median']:,.0f} ms) "
                     f"and uses ~{bf16['max_memory'] / 1024:.1f} GB instead of ~{fp32['max_memory'] / 1024:.1f} GB.")
    for variant, s in stats.items():
        if s["three_duration"]:
            lines.append(f"- **Three questions (noul+choice+score), {variant}**: {s['three_duration'] / 1000:.1f} s in a single warm invocation (n = 1), "
                         f"{s['three_duration'] / s['duration_median']:.1f} times the one-question median ({s['duration_median'] / 1000:.1f} s): cost grows almost linearly with questions.")
    low_memory = ", ".join(f"{label} ({memory:,.0f} MB): {median / 1000:.1f} s" for label, (memory, _, median, _) in extra_stats.items() if median)
    lines.append(f"- **Less memory (extra runs, warm median)**: {low_memory}. The cause of the slowdown with less memory was not isolated; one hypothesis is "
                 "torch thread oversubscription versus the allocated vCPUs.")
    sizes = [n for (_, _, _, n) in extra_stats.values() if n]
    if sizes:
        lines.append(f"- **Small sample in the sweep**: {min(sizes)} to {max(sizes)} warm invocations per configuration; good for order of magnitude, not for p90.")
    return lines


def main_table(stats: dict, main_runs: dict) -> list[str]:
    lines = ["# Strands Decider on AWS Lambda (arm64)", "",
             "| Variant | Configured memory | Warm (n) | Median (ms) | p90 (ms) | Max (ms) | Billed median (ms) | Max memory used (MB) | Fits in 10,240 MB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for variant, rows in main_runs.items():
        s = stats[variant] = summarize(rows, LIMIT_MB)
        fits = "yes" if s["max_memory"] < LIMIT_MB else "NO"
        lines.append(
            f"| {variant} | 10,240 MB | {s['warm_n']} | {s['duration_median']:.0f} | {s['duration_p90']:.0f} | "
            f"{s['duration_max']:.0f} | {s['billed_median']:.0f} | {s['max_memory']:.0f} | {fits} ({s['max_memory'] / LIMIT_MB:.1%}) |"
        )
    lines += ["", "Duration = `Duration` field of the REPORT line (client duration adds ~100 ms of network). "
              "Warm invocations use one `noul` question.", "",
              "## Cold start and three questions", "",
              "| Variant | Cold: Duration (s) | Cold: Init (ms) | Cold: client (s) | Three questions (noul+choice+score): Duration (ms) |",
              "|---|---|---|---|---|"]
    for variant, s in stats.items():
        lines.append(f"| {variant} | {s['cold_duration'] / 1000:.1f} | {s['cold_init']:.0f} | {s['cold_client'] / 1000:.1f} | {s['three_duration']:.0f} |")
    return lines + [""]


def cost_table(stats: dict) -> list[str]:
    lines = ["## Cost and free tier", "",
             f"Arm price in us-east-1: ${PRICE_PER_GB_SECOND_ARM:.10f} per GB-s and $0.20 per million requests. "
             f"Lambda free tier (https://aws.amazon.com/lambda/pricing/): 1 million requests and {FREE_TIER_GB_SECONDS:,} GB-s per month.", "",
             "| Variant | GB-s per decision, 1 question | Cost per decision, 1 question (USD) | Free-tier decisions/month, 1 question | "
             "GB-s per decision, 3 questions (n = 1) | Cost per decision, 3 questions (USD) | Free-tier decisions/month, 3 questions |",
             "|---|---|---|---|---|---|---|"]
    for variant, s in stats.items():
        three_cost = s["three_gb_s"] * PRICE_PER_GB_SECOND_ARM + PRICE_PER_REQUEST
        three_free = min(FREE_TIER_GB_SECONDS / s["three_gb_s"], FREE_TIER_REQUESTS)
        lines.append(f"| {variant} (10,240 MB) | {s['gb_s_per_decision']:.1f} | {s['cost_per_decision']:.6f} | {s['free_decisions']:,.0f} | "
                     f"{s['three_gb_s']:.1f} | {three_cost:.6f} | {three_free:,.0f} |")
    return lines + [""]


def extra_table(extra: list[dict], extra_stats: dict) -> list[str]:
    labels: list[str] = []
    for row in extra:
        if row["label"] not in labels:
            labels.append(row["label"])
    lines = ["## Extra runs (bf16 with less memory and a second fp32 cold start)", "",
             "| Configuration | Configured memory (MB) | Cold: Duration (s) | Warm (n) | Median (ms) | Max memory used (MB) | GB-s per decision | Cost per decision (USD) | Free decisions/month |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label in sorted(labels, key=lambda name: (not name.startswith("fp32"), name)):
        rows = [r for r in extra if r["label"] == label]
        memory_mb = float(rows[0]["configured_memory_mb"])
        if not [r for r in rows if r["kind"] == m.WARM and not r["function_error"]]:
            lines.append(f"| {label} | {memory_mb:.0f} | failed | 0 | - | - | - | - | - |")
            continue
        s = summarize([r for r in rows if not r["function_error"]], memory_mb)
        extra_stats[label] = (memory_mb, s["cold_duration"], s["duration_median"], s["warm_n"])
        lines.append(
            f"| {label} | {memory_mb:.0f} | {s['cold_duration'] / 1000:.1f} | {s['warm_n']} | {s['duration_median']:.0f} | {s['max_memory']:.0f} | "
            f"{s['gb_s_per_decision']:.1f} | {s['cost_per_decision']:.6f} | {s['free_decisions']:,.0f} |"
        )
    return lines


def total_spend(main_runs: dict, extra: list[dict]) -> dict:
    total_billed_gb_s = 0.0
    for rows in main_runs.values():
        total_billed_gb_s += sum((number(r, "billed_ms") or 0) for r in rows) / 1000 * LIMIT_MB / 1024
    for row in extra:
        total_billed_gb_s += (number(row, "billed_ms") or 0) / 1000 * float(row["configured_memory_mb"]) / 1024
    return {"lambda_gb_s_total": round(total_billed_gb_s, 1), "lambda_usd": round(total_billed_gb_s * PRICE_PER_GB_SECOND_ARM, 4)}


def main() -> None:
    main_runs = {v: read_rows(c.RESULTS_DIR / f"lambda_{v}.csv") for v in ("fp32", "bf16")
                 if (c.RESULTS_DIR / f"lambda_{v}.csv").exists()}
    if not main_runs:
        raise SystemExit(f"No lambda_fp32.csv or lambda_bf16.csv in {c.RESULTS_DIR}: run measure.py first.")
    extra: list[dict] = []
    for path in sorted(c.RESULTS_DIR.glob("lambda_extra_*.csv")):
        extra.extend(read_rows(path))
    stats: dict = {}
    extra_stats: dict[str, tuple[float, float | None, float | None, int]] = {}
    lines = main_table(stats, main_runs) + cost_table(stats) + extra_table(extra, extra_stats)
    lines += ["", "## Reading (computed from the data above, no extrapolation)", "", *reading_lines(stats, extra_stats), ""]
    c.write_text(c.RESULTS_DIR / "summary_lambda.md", "\n".join(lines))
    spend = total_spend(main_runs, extra)
    c.write_text(c.RESULTS_DIR / "lambda_spend.json", json.dumps(spend, indent=2))
    print(json.dumps(spend))


if __name__ == "__main__":
    main()
