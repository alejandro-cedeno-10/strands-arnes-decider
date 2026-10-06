"""Create the execution role and the Decider Lambda function (arm64 image, 10,240 MB, 900 s timeout).

The image must already be in ECR (run build_image.py first).

    python scripts/aws_lambda/deploy_function.py
"""

from __future__ import annotations

import json
import sys
import time

from botocore.exceptions import ClientError

import common as c

BASIC_EXECUTION_POLICY = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
LOG_GROUP = f"/aws/lambda/{c.FUNCTION_NAME}"
MEMORY_MB = 10240
TIMEOUT_SECONDS = 900
IAM_PROPAGATION_SECONDS = 12


def create_role(iam) -> str:
    try:
        role = iam.get_role(RoleName=c.LAMBDA_ROLE)["Role"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
    else:
        if not any(r["name"] == c.LAMBDA_ROLE for r in c.created_resources()):
            sys.exit(f"role {c.LAMBDA_ROLE} already exists and was not created by these scripts")
        return role["Arn"]
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    role = iam.create_role(
        RoleName=c.LAMBDA_ROLE,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="Execution role for the Strands Decider Lambda function",
        Tags=c.tag_list(),
    )["Role"]
    c.record_resource("iam_role", c.LAMBDA_ROLE, role["Arn"])
    iam.attach_role_policy(RoleName=c.LAMBDA_ROLE, PolicyArn=BASIC_EXECUTION_POLICY)
    time.sleep(IAM_PROPAGATION_SECONDS)
    return role["Arn"]


def create_function(lam, role_arn: str, image_uri: str) -> dict:
    """Create the function, retrying while the new role propagates, and wait until it is active."""
    try:
        lam.get_function(FunctionName=c.FUNCTION_NAME)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    else:
        if not any(r["type"] == "lambda_function" for r in c.created_resources()):
            sys.exit(f"function {c.FUNCTION_NAME} already exists and was not created by these scripts")
        return lam.get_function(FunctionName=c.FUNCTION_NAME)
    for attempt in range(6):
        try:
            created = lam.create_function(
                FunctionName=c.FUNCTION_NAME,
                PackageType="Image",
                Code={"ImageUri": image_uri},
                Role=role_arn,
                Architectures=["arm64"],
                MemorySize=MEMORY_MB,
                Timeout=TIMEOUT_SECONDS,
                Environment={"Variables": {"DECIDER_KEEP_BF16": "0"}},
                Tags=c.tag_map(),
            )
            break
        except ClientError as exc:
            if "cannot be assumed" not in str(exc) or attempt == 5:
                raise
            time.sleep(10)
    c.record_resource("lambda_function", c.FUNCTION_NAME, created["FunctionArn"])
    c.record_resource("log_group", LOG_GROUP, f"arn:aws:logs:{c.REGION}:{c.account_id()}:log-group:{LOG_GROUP}")
    lam.get_waiter("function_active_v2").wait(FunctionName=c.FUNCTION_NAME, WaiterConfig={"Delay": 5, "MaxAttempts": 120})
    return lam.get_function(FunctionName=c.FUNCTION_NAME)


def main() -> None:
    sess = c.session()
    ecr = sess.client("ecr")
    repo = ecr.describe_repositories(repositoryNames=[c.ECR_REPOSITORY])["repositories"][0]
    image_uri = f"{repo['repositoryUri']}:{c.IMAGE_TAG}"
    images = ecr.describe_images(repositoryName=c.ECR_REPOSITORY)["imageDetails"]
    role_arn = create_role(sess.client("iam"))
    described = create_function(sess.client("lambda"), role_arn, image_uri)
    described.pop("ResponseMetadata", None)
    described["ecr_image_details"] = images
    c.write_text(c.RESULTS_DIR / "describe_function.json", json.dumps(described, indent=2, default=str, ensure_ascii=False))
    configuration = described["Configuration"]
    print(configuration["State"], configuration["MemorySize"], configuration["Timeout"], configuration["Architectures"])
    print("compressed image size in ECR (bytes):", [i["imageSizeInBytes"] for i in images])


if __name__ == "__main__":
    main()
