# Strands Decider in front of a Strands agent

Companion code for the article *"Un modelo System One casi gratis: Strands Decider en tu PC y en AWS Lambda"* (Spanish, on Medium).

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) is a small open-source (Apache-2.0) decision model. This repository puts it in front of an agent built with Strands and Amazon Bedrock:

- **System 1:** Decider answers short, constrained questions (`noul`, `choice`, `score`) and resolves the clear cases.
- **System 2:** a Strands agent on Amazon Bedrock (Nova Lite / Nova Pro) handles everything Decider is unsure about.

Decider runs on a local GPU for development, and on CPU in AWS: an EC2 Graviton3 instance (bf16, the faster and cheaper-per-decision option measured) or AWS Lambda (arm64).

## Repository layout

| Path | Contents |
|---|---|
| `examples/` | Runnable examples: HTTP client, fail-closed tool gate, `HumanInTheLoop` classifier, model router, harness integration and a 12-alert DevOps case. See [`examples/README.md`](examples/README.md). |
| `deploy/ec2/` | `serve.py` (official FastAPI app, bf16 on CPU, one evaluation at a time for issue #17), a systemd unit and a concurrency smoke test. |
| `deploy/lambda/` | Lambda handler, container image (Dockerfile, CodeBuild buildspec), SAM template and local tests. See [`deploy/lambda/README.md`](deploy/lambda/README.md). |
| `scripts/aws_lambda/` | boto3 scripts to build the image in CodeBuild, deploy the function, measure it and tear everything down. |
| `scripts/aws_ec2/benchmark.py` | Temporary EC2 Graviton instance driven through SSM (no SSH, no inbound rules): CPU benchmark, head fine-tuning run, teardown. |
| `finetune/` | Adapt Decider to your use case: generate labelled alerts with Bedrock, build splits, fine-tune only the readout head (frozen torso, cached features), evaluate the "Decider only" policy. |
| `scripts/issue17_control.py` | Reproduces the upstream concurrency issue (#17) against a sequential control and a locked server. |
| `benchmarks/` | Latency, multi-question and language benchmarks, and `cpu_variants.py` (fp32 vs bf16 vs int8 on CPU). |
| `cost/savings_calculator.py` | Monthly cost estimate: LLM only vs. Decider self-hosted vs. Decider on Lambda, including the `HumanInTheLoop` classifier queries. |
| `env/` | Environment script and locked dependencies. |

Run outputs are written to `results/`, which is not versioned.

## Quick start

Requirements: Python 3.12, an NVIDIA GPU (recommended), and AWS credentials with access to Amazon Bedrock Nova in `us-east-1`.

```bash
git clone https://github.com/alejandro-cedeno-10/strands-arnes-decider.git
cd strands-arnes-decider
python -m venv .venv
source env/env.sh
$PY -m pip install -r env/requirements-lock.txt

python deploy/lambda/download_weights.py
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --device cuda --host 127.0.0.1 --port 8099

python examples/a_http_client.py
python examples/d_gate_bedrock.py
```

Examples `f` and `g` use `strands-harness` and need a separate virtualenv with `env/requirements-lock-harness.txt`.

Estimate costs without touching AWS:

```bash
python cost/savings_calculator.py                     # Nova Lite, three questions per alert
python cost/savings_calculator.py --duration-s 2.414  # one question per alert
python cost/savings_calculator.py --model nova-pro
```

## Hosting on EC2 Graviton

```bash
python deploy/ec2/serve.py --host 127.0.0.1 --port 8099   # bf16 on CPU, serialized evaluations
python deploy/ec2/smoke_test.py --clients 8 --requests 24
```

Measured on a c7g.2xlarge (Graviton3): 0.49 s for one question and 1.27 s for three with bf16 and 8 threads, about five times faster than Lambda arm64. int8 dynamic quantization was slower and changed answers. An always-on instance is a fixed cost: run the calculator with `--ec2-instance` to see where it beats Lambda.

## Adapting Decider to your case

```bash
python finetune/generate_alerts.py        # labelled synthetic alerts (Bedrock, a few cents)
python finetune/build_examples.py         # train / calib / test splits, hand-written holdout kept apart
python finetune/tune_head.py --out results/finetune/checkpoint
python finetune/evaluate.py --checkpoint results/finetune/checkpoint --name tuned
```

Only the readout head is trained; the torso and its LoRA stay frozen, so the torso runs once and training takes seconds on CPU. Evaluate on cases the tuning never saw: on synthetic test alerts accuracy rose from 76.7 % to 95 %, while on 12 hand-written alerts it dropped from 91.7 % to 83.3 %.

## Deploying to AWS Lambda

Follow [`deploy/lambda/README.md`](deploy/lambda/README.md). In short:

```bash
python scripts/aws_lambda/build_image.py      # ECR + CodeBuild role + arm64 build (~4 min)
python scripts/aws_lambda/deploy_function.py  # execution role + function
python scripts/aws_lambda/measure.py bf16
python scripts/aws_lambda/teardown.py         # deletes every resource named strands-decider-demo*
```

These scripts create billable resources. `teardown.py` deletes resources by fixed names; run it only in an account where nothing else uses the `strands-decider-demo` prefix.

## Known limitations

- **Concurrency:** `strands-decider` 0.1.0 can return wrong answers with HTTP 200 under concurrent requests (upstream issue #17). Serve one evaluation at a time per model instance: a queue with a single consumer, or a lock around the whole evaluation.
- **Thresholds:** the confidence thresholds in the examples are illustrative. Calibrate them with your own traffic.
- **Cost figures** are estimates built from a small lab run (12 synthetic alerts); see the article for the assumptions.

## Security

No credentials are stored in this repository. The scripts read the AWS account from STS at runtime and mask it as `<account-id>` in saved outputs.
