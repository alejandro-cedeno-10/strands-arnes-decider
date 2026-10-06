"""Strands Decider on AWS Lambda (container image, CPU only).

Contract: the same JSON as POST /v1/systemone:
    {"state": ..., "questions": {...}}
Accepts a direct invocation (boto3 `lambda.invoke`) or an HTTP event with `body` (Function URL with
AuthType AWS_IAM, or API Gateway). Returns {"model", "answers", "usage", "latency_ms", "cold_start_load_ms"}.

Design notes:
- The model is loaded ONCE per execution environment, lazily on the first invocation, and reused
  afterwards. The first invocation pays the model load.
- Standard Lambda serves one invocation at a time per environment, so the issue #17 race
  (`_last_offsets`) does not apply. Lambda Managed Instances DO run concurrent requests per
  environment (separate processes in Python), and each process would load its own model copy.
- Weights are baked into the image (HF_HOME=/opt/hf, HF_HUB_OFFLINE=1): no downloads on cold start.
- DECIDER_KEEP_BF16=1 keeps the torso in bf16 on CPU instead of upcasting to fp32: about half the
  memory in exchange for more latency. Measure before choosing.
- TORCH_NUM_THREADS optionally pins the number of torch threads.
"""

from __future__ import annotations

import base64
import json
import os
import time

CHECKPOINT = os.getenv("DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19")
_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from strands_decider.infer import SystemOneEngine, load_engine

        if os.getenv("DECIDER_KEEP_BF16") == "1":
            SystemOneEngine._upcast_torso_for_cpu = lambda self: None  # type: ignore[method-assign]
        torch_threads = os.getenv("TORCH_NUM_THREADS")
        if torch_threads:
            import torch

            torch.set_num_threads(int(torch_threads))
        _ENGINE = load_engine(CHECKPOINT, device="cpu")
    return _ENGINE


def _is_http_event(event) -> bool:
    """Function URL and API Gateway events carry the request in `body`."""
    return isinstance(event, dict) and "body" in event


def _payload(event: dict) -> dict:
    if not _is_http_event(event):
        return event
    body = event["body"] or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _respond(event: dict, status: int, body: dict) -> dict:
    """Return an HTTP response for HTTP events; for direct invocations, raise on errors."""
    if _is_http_event(event):
        return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}
    if status != 200:
        raise ValueError(json.dumps(body))
    return body


def lambda_handler(event, context):
    """Validate the request, evaluate it, and map caller errors (schema or truncated options) to 422."""
    from pydantic import ValidationError
    from strands_decider.schema import SystemOneRequest

    try:
        request = SystemOneRequest.model_validate(_payload(event))
    except (ValidationError, json.JSONDecodeError) as exc:
        return _respond(event, 422, {"error": "invalid request", "detail": str(exc)[:2000]})

    cold = _ENGINE is None
    load_started = time.perf_counter()
    engine = _engine()
    load_ms = (time.perf_counter() - load_started) * 1000
    evaluate_started = time.perf_counter()
    try:
        response = engine.evaluate(request)
    except ValueError as exc:
        return _respond(event, 422, {"error": str(exc)})
    output = response.model_dump()
    output["latency_ms"] = round((time.perf_counter() - evaluate_started) * 1000, 2)
    output["cold_start_load_ms"] = round(load_ms, 1) if cold else 0.0
    return _respond(event, 200, output)
