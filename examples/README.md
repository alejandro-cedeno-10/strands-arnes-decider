# Examples

System 1 is Strands Decider: a small local model served at `127.0.0.1:8099`. System 2 is Amazon Bedrock: `us.amazon.nova-lite-v1:0` for routine work and `us.amazon.nova-pro-v1:0` for advanced work.

| File | What it shows |
|---|---|
| `a_http_client.py` | Minimal `/v1/systemone` client with the three primitives (`noul`, `choice`, `score`) |
| `b_gate_interventions.py` | `DeciderGate`, a fail-closed tool gate built on `InterventionHandler`; run directly, it prints the probabilities with and without a city |
| `b_fail_closed.py` | Decider down + `on_error="deny"`: the tool does not run |
| `c_hitl_decider.py` | `HumanInTheLoop` with Decider as the risk classifier |
| `d_gate_bedrock.py` | Nova Lite agent + gate: without a city the tool does not run; with "Quito" it does (`REPS=3`) |
| `e_router_decider.py` | `ModelRouter` with a Decider strategy: Nova Lite or Nova Pro |
| `f_harness_decider.py` | `create_harness` (strands-harness) with and without `DeciderGate` |
| `g_devops_self_healing.py` | 12 synthetic alerts: System 1 resolves the clear ones, the rest go to the harness agent |
| `bedrock_models.py` | `nova_lite()`, `nova_pro()` and an optional ledger of the real `usage` tokens |

## Prerequisites

- Python 3.12 and the locked dependencies in `env/requirements-lock.txt` (examples a to e) or `env/requirements-lock-harness.txt` (examples f and g).
- An NVIDIA GPU is recommended to serve Decider; CPU works but is slow.
- AWS credentials with access to Amazon Bedrock Nova Lite and Nova Pro in `us-east-1`.

## How to run

```bash
source env/env.sh
export AWS_PROFILE=<profile with Bedrock access> AWS_REGION=us-east-1
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cuda --host 127.0.0.1 --port 8099

export BEDROCK_USAGE_LEDGER=$PWD/results/bedrock_usage.jsonl   # optional: records real token usage
python examples/d_gate_bedrock.py
python examples/g_devops_self_healing.py [--baseline]          # run from the harness virtualenv
```

The Bedrock demos cost a few US cents per full run. Decider serves one request at a time reliably (upstream issue #17): do not run two examples in parallel against the same server.
