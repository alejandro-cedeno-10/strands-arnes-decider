"""Build the arm64 Decider image in AWS CodeBuild and push it to ECR (no local Docker needed).

Creates an ECR repository, a CodeBuild IAM role with a minimal inline policy, and a CodeBuild
project with NO_SOURCE. Starts the build, waits for it, and saves the log to
results/aws_lambda/codebuild.log.

The build context (Dockerfile, handler.py, download_weights.py) travels inside the inline buildspec,
base64-encoded, so no S3 bucket is required.

    python scripts/aws_lambda/build_image.py
"""

from __future__ import annotations

import base64
import json
import sys
import time

import yaml
from botocore.exceptions import ClientError

import common as c

CONTEXT_FILES = ["Dockerfile", "handler.py", "download_weights.py"]
LOG_GROUP = f"/aws/codebuild/{c.CODEBUILD_PROJECT}"
POLL_SECONDS = 30
IAM_PROPAGATION_SECONDS = 12


def ensure_absent_or_ours(kind: str, name: str, exists: bool) -> bool:
    """Refuse to touch a pre-existing resource that this project did not create."""
    ours = any(r["type"] == kind and r["name"] == name for r in c.created_resources())
    if exists and not ours:
        sys.exit(f"{kind} '{name}' already exists and was not created by these scripts: rename PREFIX in common.py")
    return exists


def create_repository(ecr) -> dict:
    try:
        repo = ecr.describe_repositories(repositoryNames=[c.ECR_REPOSITORY])["repositories"][0]
        ensure_absent_or_ours("ecr_repository", c.ECR_REPOSITORY, True)
        return repo
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
    repo = ecr.create_repository(
        repositoryName=c.ECR_REPOSITORY,
        imageScanningConfiguration={"scanOnPush": False},
        tags=c.tag_list(),
    )["repository"]
    c.record_resource("ecr_repository", c.ECR_REPOSITORY, repo["repositoryArn"])
    return repo


def codebuild_policy(repo_arn: str, account: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [
                    f"arn:aws:logs:{c.REGION}:{account}:log-group:{LOG_GROUP}",
                    f"arn:aws:logs:{c.REGION}:{account}:log-group:{LOG_GROUP}:*",
                ],
            },
            {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": repo_arn,
            },
        ],
    }


def create_role(iam, repo_arn: str, account: str) -> str:
    try:
        role = iam.get_role(RoleName=c.CODEBUILD_ROLE)["Role"]
        ensure_absent_or_ours("iam_role", c.CODEBUILD_ROLE, True)
        return role["Arn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    role = iam.create_role(
        RoleName=c.CODEBUILD_ROLE,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="CodeBuild: builds the Strands Decider Lambda image",
        Tags=c.tag_list(),
    )["Role"]
    c.record_resource("iam_role", c.CODEBUILD_ROLE, role["Arn"])
    iam.put_role_policy(
        RoleName=c.CODEBUILD_ROLE,
        PolicyName="build-minimal",
        PolicyDocument=json.dumps(codebuild_policy(repo_arn, account)),
    )
    time.sleep(IAM_PROPAGATION_SECONDS)
    return role["Arn"]


def inline_buildspec() -> str:
    """Return deploy/lambda/buildspec.yml with the build-context files written out in pre_build."""
    spec = yaml.safe_load((c.DEPLOY_DIR / "buildspec.yml").read_text(encoding="utf-8"))
    writers = []
    for filename in CONTEXT_FILES:
        encoded = base64.b64encode((c.DEPLOY_DIR / filename).read_bytes()).decode("ascii")
        writers.append(f"echo '{encoded}' | base64 -d > {filename}")
    spec["phases"]["pre_build"]["commands"] = writers + spec["phases"]["pre_build"]["commands"]
    return yaml.safe_dump(spec, sort_keys=False, width=10**9)


def project_arguments(role_arn: str, repo_uri: str) -> dict:
    return dict(
        name=c.CODEBUILD_PROJECT,
        source={"type": "NO_SOURCE", "buildspec": inline_buildspec()},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={
            "type": "ARM_CONTAINER",
            "image": "aws/codebuild/amazonlinux-aarch64-standard:3.0",
            "computeType": "BUILD_GENERAL1_LARGE",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "ECR_REGISTRY", "value": repo_uri.split("/")[0], "type": "PLAINTEXT"},
                {"name": "ECR_REPOSITORY", "value": c.ECR_REPOSITORY, "type": "PLAINTEXT"},
                {"name": "IMAGE_TAG", "value": c.IMAGE_TAG, "type": "PLAINTEXT"},
            ],
        },
        serviceRole=role_arn,
        timeoutInMinutes=60,
        logsConfig={"cloudWatchLogs": {"status": "ENABLED", "groupName": LOG_GROUP}},
        tags=[{"key": k, "value": v} for k, v in c.TAGS.items()],
    )


def create_project(codebuild, role_arn: str, repo_uri: str) -> None:
    """Create the CodeBuild project, retrying while the new IAM role propagates."""
    existing = codebuild.batch_get_projects(names=[c.CODEBUILD_PROJECT])["projects"]
    if ensure_absent_or_ours("codebuild_project", c.CODEBUILD_PROJECT, bool(existing)):
        codebuild.update_project(**project_arguments(role_arn, repo_uri))
        return
    kwargs = project_arguments(role_arn, repo_uri)
    for attempt in range(6):
        try:
            project = codebuild.create_project(**kwargs)["project"]
            c.record_resource("codebuild_project", c.CODEBUILD_PROJECT, project["arn"])
            c.record_resource("log_group", LOG_GROUP, f"arn:aws:logs:{c.REGION}:{c.account_id()}:log-group:{LOG_GROUP}")
            return
        except ClientError as exc:
            if "assume" not in str(exc).lower() or attempt == 5:
                raise
            time.sleep(10)


def run_build(codebuild) -> dict:
    build_id = codebuild.start_build(projectName=c.CODEBUILD_PROJECT)["build"]["id"]
    print("build:", build_id, flush=True)
    while True:
        build = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
        print(time.strftime("%H:%M:%S"), build["buildStatus"], build.get("currentPhase"), flush=True)
        if build["buildStatus"] != "IN_PROGRESS":
            return build
        time.sleep(POLL_SECONDS)


def fetch_log_lines(logs, group: str, stream: str) -> list[str]:
    lines: list[str] = []
    token = None
    while True:
        kwargs = {"logGroupName": group, "logStreamName": stream, "startFromHead": True}
        if token:
            kwargs["nextToken"] = token
        page = logs.get_log_events(**kwargs)
        lines.extend(event["message"].rstrip("\n") for event in page["events"])
        if page.get("nextForwardToken") == token or not page["events"]:
            return lines
        token = page["nextForwardToken"]


def save_log(logs, build: dict) -> None:
    lines = fetch_log_lines(logs, build["logs"]["groupName"], build["logs"]["streamName"])
    phases = [f"{p['phaseType']}={p.get('phaseStatus', '')}:{p.get('durationInSeconds', '')}s" for p in build["phases"]]
    header = (
        f"# build {build['id']} status={build['buildStatus']} start={build['startTime']} end={build.get('endTime')}\n"
        f"# phases: {'; '.join(phases)}\n"
    )
    c.write_text(c.RESULTS_DIR / "codebuild.log", header + "\n".join(lines))


def main() -> None:
    sess = c.session()
    account = c.account_id()
    repo = create_repository(sess.client("ecr"))
    role_arn = create_role(sess.client("iam"), repo["repositoryArn"], account)
    codebuild = sess.client("codebuild")
    create_project(codebuild, role_arn, repo["repositoryUri"])
    build = run_build(codebuild)
    save_log(sess.client("logs"), build)
    print("final status:", build["buildStatus"])
    if build["buildStatus"] != "SUCCEEDED":
        sys.exit(1)
    images = sess.client("ecr").describe_images(repositoryName=c.ECR_REPOSITORY)["imageDetails"]
    for image in images:
        print("ECR:", image.get("imageTags"), "compressed:", image["imageSizeInBytes"], "bytes")


if __name__ == "__main__":
    main()
