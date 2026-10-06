"""Savings calculator: deciding with an LLM agent versus putting Strands Decider (System 1) in front of it.

This is an ESTIMATE with explicit assumptions, not a bill. Change the parameters to match your data.

Prices (USD, us-east-1, checked on 2026-10-04 in the AWS price list):
- Nova Lite $0.06 / $0.24 per million tokens (input / output); Nova Pro $0.80 / $3.20.
- Claude Haiku 4.5 on Bedrock: $1.00 / $5.00 with a global profile, $1.10 / $5.50 with a regional profile.
- Lambda: Arm $0.0000133334 per GB-s, x86 $0.0000166667 per GB-s, $0.20 per million requests.
- Lambda free tier: 1 million requests and 400,000 GB-s per month (Always Free, per account).
- ECR $0.10 per GB-month. GPU EC2: g6.xlarge $0.8048/h, g5.xlarge $1.006/h (730 h per month).
- Arm provisioned concurrency: $0.0000033334 per provisioned GB-s.

Defaults are what was MEASURED with Nova Lite and 12 synthetic DevOps alerts:
- Decider resolves 3 of 12 alerts without the LLM (25 %), with thresholds fixed before measuring.
- 54 LLM calls for 12 alerts in the baseline (4.5 per decision).
- 190,036 input tokens / 54 calls = 3,520 per call (two other runs: 3,514 and 3,536); 5,825 / 54 = 108
  output tokens in that run (104 in the other two), rounded to 110.
- Arm64 Lambda with 10,240 MB and bf16 weights. The DevOps flow sends THREE questions per alert
  (choice + noul + score), so the default routing duration is 7.117 s, 71.2 GB-s per decision
  (a single measured invocation, n = 1). With ONE noul question the warm median is 2.414 s (24.1 GB-s; n = 20):
  use --duration-s 2.414. With fp32: 6.663 s for three questions and 2.167 s for one.

HumanInTheLoop classifier queries (one question, 2.414 s): both modes ask Decider before every tool that
changes the system. The first run made 6 for 12 alerts in hybrid mode (0.5 per alert) and 11 in the
baseline (11/12 per alert). They are added to Decider's cost in each mode, and the free tier applies to the
scenario's total GB-s.

Not included: CloudWatch logs, data transfer, cold starts (one fp32 cold start reached 8,520 GB-s) and the
cost of mistakes (wrong actions).

Usage:
    python cost/savings_calculator.py
    python cost/savings_calculator.py --model nova-pro
    python cost/savings_calculator.py --duration-s 2.414   # a single noul question per decision
    python cost/savings_calculator.py --alerts 50000 --s1-fraction 0.6 --duration-s 2.2
"""

from __future__ import annotations

import argparse

MODEL_PRICES = {
    "nova-lite": (0.06, 0.24, 0.015),
    "nova-pro": (0.80, 3.20, 0.20),
    "haiku-global": (1.00, 5.00, 0.10),
    "haiku-regional": (1.10, 5.50, 0.11),
}
GBS_PRICE = {"arm": 0.0000133334, "x86": 0.0000166667}
GPU_MONTHLY = {"g6.xlarge": 0.8048 * 730, "g5.xlarge": 1.006 * 730}
FREE_TIER_GBS = 400_000
FREE_TIER_REQUESTS = 1_000_000
SECONDS_PER_MONTH = 730 * 3600
PROVISIONED_GBS_PRICE = 0.0000033334
ECR_GB_MONTH_PRICE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alerts", type=int, default=10_000, help="decisions (alerts) per month")
    parser.add_argument("--s1-fraction", type=float, default=0.25, help="fraction Decider resolves without the LLM")
    parser.add_argument("--calls-per-decision", type=float, default=4.5)
    parser.add_argument("--tokens-in", type=int, default=3_520, help="input tokens per LLM call")
    parser.add_argument("--tokens-out", type=int, default=110, help="output tokens per LLM call")
    parser.add_argument("--model", choices=sorted(MODEL_PRICES), default="nova-lite")
    parser.add_argument("--price-in", type=float, default=None, help="USD per million input tokens (overrides --model)")
    parser.add_argument("--price-out", type=float, default=None, help="USD per million output tokens (overrides --model)")
    parser.add_argument("--price-cache", type=float, default=None, help="USD per million cached input tokens (overrides --model)")
    parser.add_argument("--cache-fraction", type=float, default=0.0,
                        help="fraction of input tokens read from the prompt cache (0 to 1)")
    parser.add_argument("--lambda-gb", type=float, default=10.0, help="Lambda memory in GB (10,240 MB = 10 GB)")
    parser.add_argument("--duration-s", type=float, default=7.117,
                        help="billed seconds per routing query in Lambda (bf16: 7.117 with 3 questions, n = 1; "
                             "2.414 with 1 noul question, n = 20; fp32: 6.663 and 2.167)")
    parser.add_argument("--hitl-hybrid", type=float, default=6 / 12,
                        help="HumanInTheLoop classifier queries per alert in hybrid mode (measured: 6/12)")
    parser.add_argument("--hitl-baseline", type=float, default=11 / 12,
                        help="HumanInTheLoop classifier queries per alert in the baseline (measured: 11/12)")
    parser.add_argument("--hitl-duration-s", type=float, default=2.414,
                        help="billed seconds per classifier query (one question, bf16, n = 20)")
    parser.add_argument("--architecture", choices=sorted(GBS_PRICE), default="arm")
    parser.add_argument("--gbs-price", type=float, default=None, help="USD per GB-s (overrides --architecture)")
    parser.add_argument("--request-price", type=float, default=0.20, help="USD per million requests")
    parser.add_argument("--no-free-tier", action="store_true", help="charge everything from the first request")
    parser.add_argument("--image-gb", type=float, default=4.1, help="compressed image size in ECR, GB")
    parser.add_argument("--provisioned-environments", type=int, default=0,
                        help="environments with provisioned concurrency (no cold start, always billed)")
    parser.add_argument("--gpu-instance", choices=sorted(GPU_MONTHLY), default=None)
    parser.add_argument("--gpu-monthly", type=float, default=None, help="fixed monthly cost of a GPU machine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    price_in, price_out, price_cache = MODEL_PRICES[args.model]
    price_in = args.price_in if args.price_in is not None else price_in
    price_out = args.price_out if args.price_out is not None else price_out
    price_cache = args.price_cache if args.price_cache is not None else price_cache
    gbs_price = args.gbs_price if args.gbs_price is not None else GBS_PRICE[args.architecture]

    effective_price_in = price_in * (1 - args.cache_fraction) + price_cache * args.cache_fraction
    llm_call = args.tokens_in * effective_price_in / 1e6 + args.tokens_out * price_out / 1e6
    llm_decision = llm_call * args.calls_per_decision
    routing_gbs = args.lambda_gb * args.duration_s
    routing_query = routing_gbs * gbs_price + args.request_price / 1e6
    hitl_gbs = args.lambda_gb * args.hitl_duration_s
    hitl_query = hitl_gbs * gbs_price + args.request_price / 1e6

    n = args.alerts
    baseline_llm = n * llm_decision
    hybrid_llm = (n - n * args.s1_fraction) * llm_decision
    ecr = args.image_gb * ECR_GB_MONTH_PRICE
    provisioned = args.provisioned_environments * args.lambda_gb * SECONDS_PER_MONTH * PROVISIONED_GBS_PRICE

    def lambda_cost(gbs: float, requests: float, with_free_tier: bool) -> float:
        if with_free_tier:
            gbs, requests = max(0.0, gbs - FREE_TIER_GBS), max(0.0, requests - FREE_TIER_REQUESTS)
        return gbs * gbs_price + requests * args.request_price / 1e6 + ecr + provisioned

    baseline_gbs = n * args.hitl_baseline * hitl_gbs
    baseline_requests = n * args.hitl_baseline
    hybrid_gbs = n * routing_gbs + n * args.hitl_hybrid * hitl_gbs
    hybrid_requests = n + n * args.hitl_hybrid
    free_tier = not args.no_free_tier
    suffix = " (free tier)" if free_tier else " (no free tier)"

    rows = [
        ("Baseline: everything to the agent; self-hosted Decider only as classifier", baseline_llm, 0.0),
        ("Hybrid: self-hosted Decider routes and classifies", hybrid_llm, 0.0),
        ("Baseline with the classifier in Lambda" + suffix, baseline_llm,
         lambda_cost(baseline_gbs, baseline_requests, free_tier)),
        ("Hybrid with Decider in Lambda" + suffix, hybrid_llm, lambda_cost(hybrid_gbs, hybrid_requests, free_tier)),
    ]
    if free_tier:
        rows += [
            ("Baseline with the classifier in Lambda (no free tier)", baseline_llm,
             lambda_cost(baseline_gbs, baseline_requests, False)),
            ("Hybrid with Decider in Lambda (no free tier)", hybrid_llm,
             lambda_cost(hybrid_gbs, hybrid_requests, False)),
        ]
    gpu = args.gpu_monthly if args.gpu_monthly is not None else (GPU_MONTHLY[args.gpu_instance] if args.gpu_instance else None)
    if gpu is not None:
        rows.append(("Hybrid with Decider on a GPU instance running all month", hybrid_llm, gpu))

    print(f"Assumptions: {n:,} decisions/month; Decider resolves {args.s1_fraction:.0%}; "
          f"{args.calls_per_decision:g} LLM calls per decision; {args.tokens_in:,} tokens in / {args.tokens_out} tokens out per call; "
          f"{args.model} ${price_in}/${price_out} per MTok, {args.cache_fraction:.0%} of input from cache; "
          f"Lambda {args.architecture} {args.lambda_gb:g} GB: {args.duration_s:g} s per routing query and {args.hitl_duration_s:g} s per "
          f"classifier query ({args.hitl_hybrid:.3g} per alert in hybrid, {args.hitl_baseline:.3g} in baseline); "
          f"{args.image_gb:g} GB image in ECR.\n")
    print(f"Cost of ONE decision with the LLM:             ${llm_decision:.5f}  ({llm_call:.5f} per call)")
    print(f"Cost of ONE routing query (Lambda):            ${routing_query:.6f}  ({routing_gbs:.1f} GB-s)")
    print(f"Cost of ONE classifier query (Lambda):         ${hitl_query:.6f}  ({hitl_gbs:.1f} GB-s)")
    print(f"Lambda GB-s per month: baseline {baseline_gbs:,.0f}; hybrid {hybrid_gbs:,.0f} (free tier: {FREE_TIER_GBS:,}).")
    print()
    print("| Scenario | LLM (USD/month) | Decider (USD/month) | Total (USD/month) |")
    print("|---|---:|---:|---:|")
    for name, llm, decider in rows:
        print(f"| {name} | {llm:,.2f} | {decider:,.2f} | {llm + decider:,.2f} |")

    extra_per_alert = routing_query + (args.hitl_hybrid - args.hitl_baseline) * hitl_query
    break_even = extra_per_alert / llm_decision if llm_decision else float("inf")
    if break_even >= 1:
        print("\nBreak-even: with paid Lambda, hybrid mode does NOT pay off against the baseline with this model.")
    else:
        print(f"\nBreak-even with paid Lambda (hybrid vs baseline, both with the classifier in Lambda): "
              f"hybrid pays off if Decider resolves more than {break_even:.1%} of alerts (current fraction: "
              f"{args.s1_fraction:.0%}). Treats the measured classifier queries as fixed.")
    print("Self-hosted Decider: marginal cost per query is 0; the cost is the machine.")
    if args.provisioned_environments:
        print(f"Provisioned concurrency: {args.provisioned_environments} environment(s) of {args.lambda_gb:g} GB = "
              f"${provisioned:,.2f} per month, before duration.")
    print("Not included: CloudWatch logs, data transfer, cold starts, or the cost of mistakes (wrong actions).")


if __name__ == "__main__":
    main()
