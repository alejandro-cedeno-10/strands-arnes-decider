# Strands Decider on AWS Lambda

This folder packages Strands Decider 2B as an arm64 container image for AWS Lambda (CPU only).
It was deployed and measured on real Lambda (us-east-1, October 2026) and then torn down.

Summary of the measurements:

- Lambda **works** at 10,240 MB: ~2.2 s (fp32) or ~2.4 s (bf16) per warm decision with one question,
  and 6.7–7.1 s with three questions.
- Cold starts ranged from tens of seconds to ~14 minutes.
- fp32 peaked at 10,229 of 10,240 MB; bf16 at ~3.6–3.8 GB. **bf16 is the recommended variant.**

> **Cost warning:** deploying in your account is billed (CodeBuild, ECR storage, Lambda duration,
> CloudWatch Logs). The full measurement run cost about USD 0.40. Run the teardown when you finish.

## When it makes sense

Lambda has no GPU, so Decider runs on CPU. A few seconds per decision is fine for **asynchronous or
low-frequency** decisions (alert triage via EventBridge, ticket classification via SQS, batch jobs).
For an inline gate before every tool call of a conversational agent, use a GPU host instead.

```
CloudWatch alarm / EventBridge ──► "triage" Lambda (agent) ──invoke──► "decider" Lambda (CPU, 10 GB)
                                          │
                                          └──► Amazon Bedrock (System 2) only when Decider is unsure
```

- The Decider function has **no public URL**: the agent invokes it with `lambda:InvokeFunction` (IAM).
- Standard Lambda runs one invocation per environment, so the issue #17 race does not apply.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Lambda Python 3.12 base, CPU-only torch, weights baked in at pinned revisions |
| `buildspec.yml` | CodeBuild steps: build arm64 image, inspect size, push to ECR |
| `download_weights.py` | Downloads the pinned model revisions during the build |
| `handler.py` | Lambda handler; same JSON contract as `POST /v1/systemone` |
| `lambda_client.py` | `ask()` client with the same interface as `examples/a_http_client.py` |
| `test_handler.py` | Local handler test with a stub engine (no model) |
| `test_local.ps1` | Runs the image locally with the Lambda Runtime Interface Emulator |
| `template.yaml` | AWS SAM template (**not tested**) |

## Deploy, measure, tear down

The scripts live in `scripts/aws_lambda/` and use boto3 only (no local Docker). Credentials come from
`AWS_PROFILE` or the default chain; region from `AWS_REGION` (default `us-east-1`). Every resource is
named with the prefix `strands-decider-demo`, tagged `Project=strands-decider-demo`, and recorded in
`results/aws_lambda/created_resources.json`. Saved outputs mask the account ID as `<account-id>`.

```bash
python deploy/lambda/test_handler.py                 # local check, no AWS
python scripts/aws_lambda/build_image.py             # ECR repo + CodeBuild role + arm64 build (~4 min)
python scripts/aws_lambda/deploy_function.py         # execution role + function (10,240 MB, 900 s)
python scripts/aws_lambda/measure.py fp32            # 1 cold + 20 warm + 1 three-question invocation
python scripts/aws_lambda/measure.py bf16
python scripts/aws_lambda/extra_runs.py --sweep      # optional: bf16 memory sweep
python scripts/aws_lambda/summarize.py               # results/aws_lambda/summary_lambda.md
python scripts/aws_lambda/teardown.py                # delete everything and verify NotFound
```

1. **Image build.** `build_image.py` creates the ECR repository, a CodeBuild role with a minimal inline
   policy (project logs, push to that repository, `GetAuthorizationToken`), and an `ARM_CONTAINER`
   project. The build context travels base64-encoded inside the inline buildspec, so no S3 bucket is needed.
2. **Function.** `deploy_function.py` creates a role with `AWSLambdaBasicExecutionRole` and the function
   (`PackageType=Image`, `arm64`, 10,240 MB, 900 s). Switch to bf16 with `DECIDER_KEEP_BF16=1`.
3. **Measurement.** `measure.py` writes per-invocation CSV/JSONL and REPORT lines to `results/aws_lambda/`.
4. **Teardown.** See the warning below.

> **Teardown deletes by fixed names.** `teardown.py` deletes the function, both log groups, the CodeBuild
> project, the ECR repository (with `force=True`) and both IAM roles **by the names in `common.py`**, not
> only by the resource record. If your account already has resources with those names, they are deleted
> too. Review `PREFIX` in `scripts/aws_lambda/common.py` before running it.

Organization tags can be added with `DECIDER_EXTRA_TAGS='{"Owner": "..."}'`. If your account has a lower
Lambda memory quota than 10,240 MB, request an increase in Service Quotas.

**Local Docker alternative:** `docker build --platform linux/arm64 -t strands-decider-lambda .`, push to
ECR, and create the function with that `ImageUri`. You need ~7 GB of free disk.

## Limits that matter

From [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html):

- Memory 128–10,240 MB; CPU scales with memory up to ~6 vCPUs.
- Container images up to 10 GB uncompressed. This image is 6.20 GB uncompressed (4.10 GB in ECR).
- Maximum timeout 900 s.

## Cold starts (measured)

| Case | Duration (REPORT) |
|---|---|
| fp32, first invocation after deploying | 851.6 s (14.2 min), 95 % of the 900 s timeout |
| fp32, second cold start | 38.4 s |
| bf16, first cold start | 190.7 s |
| bf16 at other memory sizes | 4,096 MB: 151.9 s; 6,144 MB: 127.5 s (26.6 s with 3 threads); 8,192 MB: 28.6 s |

The cause of the long first cold start was not isolated (one hypothesis is lazy image loading from ECR
plus fp32 memory pressure). Use bf16, a 900 s timeout, and a warm-up invocation after each deploy.
Provisioned concurrency avoids cold starts but is billed even without traffic, and only helps if the
model load moves out of the first invocation. `cold_start_load_ms` in the response measures the model load.

## Cost per decision (measured)

Arm price in us-east-1: $0.0000133334 per GB-s and $0.20 per million requests. Free tier: 1 million
requests and 400,000 GB-s per month.

| Configuration | GB-s per decision | Cost per decision | Decisions/month within the free tier |
|---|---|---|---|
| fp32, 10,240 MB | 21.7 | $0.000289 | ~18,400 |
| bf16, 10,240 MB | 24.1 | $0.000322 | ~16,600 |
| bf16, 6,144 MB, `TORCH_NUM_THREADS=3` (3 warm) | 13.9 | $0.000186 | ~28,800 |
| bf16, 10,240 MB, three questions (n = 1) | 71.2 | $0.000949 | ~5,600 |

Excludes ECR, CloudWatch Logs and cold starts (one 852 s fp32 cold start cost 8,520 GB-s, ≈ $0.11).

**Watch torch threads.** With default threads, bf16 at 4,096 and 6,144 MB took 100–130 s per decision;
at 8,192 MB, 2.8 s. With `TORCH_NUM_THREADS=3`, 6,144 MB took 2.3 s. If you lower memory, pin threads and measure.

## Problems found

| Problem | Fix |
|---|---|
| The PyPI torch wheel for aarch64 is not CPU-only: torch 2.11.0 pulls ~4 GB of CUDA libraries and the image reached 10.09 GB. | Install torch from `https://download.pytorch.org/whl/cpu` on arm64 too; the image dropped to 6.20 GB. |
| fp32 uses 10,229 of 10,240 MB. | Use bf16. |
| Torch prints `Error in cpuinfo: failed to parse the list of possible processors` on import. | Harmless in these tests. |

## Alternatives

| Option | When |
|---|---|
| ECS/EC2 with GPU (g5/g6) behind an internal, authenticated load balancer | inline gates, high steady volume |
| Your own SageMaker endpoint | if you already operate models on SageMaker |
| Same host as the agent (sidecar) | local development or a single agent with a GPU |
| Lambda (this folder) | asynchronous decisions, irregular traffic, no GPU |

Lambda Managed Instances also have no GPU and run several invocations per environment (separate
processes in Python, each loading its own model copy); limit per-environment concurrency to 1 if you use them.
