"""Decider-on-Lambda client with the same interface as examples/a_http_client.ask().

Requires the deployed function and AWS credentials.

    from lambda_client import ask
    ask("Help! My payouts...", {"team": {"type": "choice", "instructions": "...", "criteria": {...}}})

To use it in the examples, replace the `a_http_client.ask` import with this one.
The function name defaults to the one created by scripts/aws_lambda/deploy_function.py
(`strands-decider-demo`); override it with DECIDER_FUNCTION.
Region: AWS_REGION or AWS_DEFAULT_REGION (default us-east-1).
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

FUNCTION = os.getenv("DECIDER_FUNCTION", "strands-decider-demo")
_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1")


class DeciderError(RuntimeError):
    pass


def ask(state: Any, questions: dict[str, dict[str, Any]], **_: Any) -> dict[str, Any]:
    response = _client.invoke(FunctionName=FUNCTION, Payload=json.dumps({"state": state, "questions": questions}))
    body = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise DeciderError(body.get("errorMessage", str(body)))
    return body
