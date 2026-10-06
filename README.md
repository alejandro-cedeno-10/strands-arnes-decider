# Strands Decider in front of a Strands agent

Companion code for the article *"Un modelo System One casi gratis: Strands Decider en tu PC y en AWS Lambda"* (Spanish, on Medium).

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) is a small open-source (Apache-2.0) decision model. This repository puts it in front of an agent built with Strands and Amazon Bedrock:

- **System 1:** Decider answers short, constrained questions (`noul`, `choice`, `score`) and resolves the clear cases.
- **System 2:** a Strands agent on Amazon Bedrock (Nova Lite / Nova Pro) handles everything Decider is unsure about.

Decider runs on a local GPU for development and on AWS Lambda (CPU, arm64) for deployment.

## Repository layout

| Path | Contents |
|---|---|
| `examples/` | Runnable examples: HTTP client, fail-closed tool gate, `HumanInTheLoop` classifier, model router, harness integration and a 12-alert DevOps case. See [`examples/README.md`](examples/README.md). |
| `deploy/lambda/` | Lambda handler, container image (Dockerfile, CodeBuild buildspec), SAM template and local tests. See [`deploy/lambda/README.md`](deploy/lambda/README.md). |
| `scripts/aws_lambda/` | boto3 scripts to build the image in CodeBuild, deploy the function, measure it and tear everything down. |
| `scripts/issue17_control.py` | Reproduces the upstream concurrency issue (#17) against a sequential control and a locked server. |
| `benchmarks/` | Local latency, multi-question and language benchmarks for the Decider server. |
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
