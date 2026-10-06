"""Shared settings and helpers for the scripts that deploy Strands Decider to AWS Lambda.

Everything uses boto3. Profile: AWS_PROFILE (or the default boto3 credential chain when unset).
Region: AWS_REGION or AWS_DEFAULT_REGION (default us-east-1). Every resource is recorded in
results/aws_lambda/created_resources.json as soon as it exists, so it can be deleted later.

Extra tags required by your organization: DECIDER_EXTRA_TAGS='{"Owner": "...", "CostCenter": "..."}'.
To keep separate runs apart: DECIDER_RUN_TAG=my-run writes to results/aws_lambda/my-run/.
The AWS account ID is always read from STS at runtime and masked as <account-id> in saved outputs.
"""

from __future__ import annotations

import datetime
import json
import os
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

PROFILE = os.getenv("AWS_PROFILE")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
RUN_TAG = os.getenv("DECIDER_RUN_TAG", "")
ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = ROOT / "deploy" / "lambda"
RESULTS_DIR = ROOT / "results" / "aws_lambda" / RUN_TAG
RESOURCES_FILE = ROOT / "results" / "aws_lambda" / "created_resources.json"
ACCOUNT_MASK = "<account-id>"

PREFIX = "strands-decider-demo"
TAGS = {
    "Project": PREFIX,
    "Environment": "test",
    "IsTemporary": "true",
    **json.loads(os.getenv("DECIDER_EXTRA_TAGS", "{}")),
}
FUNCTION_NAME = PREFIX
ECR_REPOSITORY = PREFIX
CODEBUILD_PROJECT = f"{PREFIX}-build"
CODEBUILD_ROLE = f"{PREFIX}-codebuild-role"
LAMBDA_ROLE = f"{PREFIX}-lambda-role"
IMAGE_TAG = "v1"


def session() -> boto3.Session:
    return boto3.Session(profile_name=PROFILE, region_name=REGION)


@lru_cache(maxsize=1)
def account_id() -> str:
    return session().client("sts").get_caller_identity()["Account"]


def mask_account(text: str) -> str:
    """Replace the current AWS account ID with <account-id> before anything is written to disk.

    Offline steps (summarize.py) may run without credentials; then there is no account to mask.
    """
    try:
        account = account_id()
    except (BotoCoreError, ClientError):
        return text
    return text.replace(account, ACCOUNT_MASK)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mask_account(text), encoding="utf-8")


def append_line(path: Path, line: str) -> None:
    """Append one masked line and flush it, so an aborted run keeps every invocation recorded so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(mask_account(line) + "\n")


def tag_list()-> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in TAGS.items()]


def tag_map() -> dict[str, str]:
    return dict(TAGS)


def _load_resources() -> list[dict]:
    if RESOURCES_FILE.exists():
        return json.loads(RESOURCES_FILE.read_text(encoding="utf-8"))
    return []


def _save_resources(resources: list[dict]) -> None:
    write_text(RESOURCES_FILE, json.dumps(resources, indent=2, ensure_ascii=False))


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def record_resource(resource_type: str, name: str, arn: str) -> None:
    resources = _load_resources()
    if any(r["type"] == resource_type and r["name"] == name and r["status"] == "created" for r in resources):
        return
    resources.append({
        "type": resource_type,
        "name": name,
        "arn": arn,
        "created_at": _now(),
        "status": "created",
        "deleted_at": None,
    })
    _save_resources(resources)


def mark_deleted(resource_type: str, name: str) -> None:
    resources = _load_resources()
    for resource in resources:
        if resource["type"] == resource_type and resource["name"] == name and resource["status"] == "created":
            resource["status"] = "deleted"
            resource["deleted_at"] = _now()
    _save_resources(resources)


def created_resources() -> list[dict]:
    return [r for r in _load_resources() if r["status"] == "created"]
