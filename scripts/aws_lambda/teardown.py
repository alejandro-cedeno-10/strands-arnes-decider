"""Delete the resources created by build_image.py and deploy_function.py, then verify they are gone.

Reverse creation order: Lambda function, log groups, CodeBuild project, ECR repository (force),
IAM roles. Then it checks that nothing tagged Project=<PREFIX> remains. The evidence (one line per
NotFound check) goes to results/aws_lambda/teardown.txt with the account ID masked.

WARNING: deletion uses the fixed names in common.py, not the resource record. If your account already
has a repository or role with those names, it is deleted too (the repository with force=True).
Review PREFIX in common.py before running this.

    python scripts/aws_lambda/teardown.py
"""

from __future__ import annotations

import datetime

from botocore.exceptions import ClientError

import common as c

EVIDENCE: list[str] = []
MISSING_CODES = ("ResourceNotFoundException", "NoSuchEntity", "RepositoryNotFoundException", "NotFound", "404")


def note(message: str) -> None:
    line = c.mask_account(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}  {message}")
    EVIDENCE.append(line)
    print(line, flush=True)


def is_missing(exc: ClientError) -> bool:
    return exc.response["Error"]["Code"] in MISSING_CODES


def delete_function(sess) -> None:
    lam = sess.client("lambda")
    try:
        lam.delete_function(FunctionName=c.FUNCTION_NAME)
    except ClientError as exc:
        if not is_missing(exc):
            raise
    try:
        lam.get_function(FunctionName=c.FUNCTION_NAME)
        note(f"ERROR: function {c.FUNCTION_NAME} still exists")
    except ClientError as exc:
        note(f"lambda get_function {c.FUNCTION_NAME} -> {exc.response['Error']['Code']}")
        c.mark_deleted("lambda_function", c.FUNCTION_NAME)


def delete_log_group(sess, name: str) -> None:
    logs = sess.client("logs")
    try:
        logs.delete_log_group(logGroupName=name)
    except ClientError as exc:
        if not is_missing(exc):
            raise
    groups = logs.describe_log_groups(logGroupNamePrefix=name)["logGroups"]
    exact = [g for g in groups if g["logGroupName"] == name]
    note(f"logs describe_log_groups {name} -> {'STILL EXISTS' if exact else 'NotFound (empty list)'}")
    if not exact:
        c.mark_deleted("log_group", name)


def delete_project(sess) -> None:
    codebuild = sess.client("codebuild")
    codebuild.delete_project(name=c.CODEBUILD_PROJECT)
    remaining = codebuild.batch_get_projects(names=[c.CODEBUILD_PROJECT])
    note(f"codebuild batch_get_projects {c.CODEBUILD_PROJECT} -> projects={remaining['projects']} projectsNotFound={remaining['projectsNotFound']}")
    if not remaining["projects"]:
        c.mark_deleted("codebuild_project", c.CODEBUILD_PROJECT)


def delete_repository(sess) -> None:
    ecr = sess.client("ecr")
    try:
        ecr.delete_repository(repositoryName=c.ECR_REPOSITORY, force=True)
    except ClientError as exc:
        if not is_missing(exc):
            raise
    try:
        ecr.describe_repositories(repositoryNames=[c.ECR_REPOSITORY])
        note(f"ERROR: repository {c.ECR_REPOSITORY} still exists")
    except ClientError as exc:
        note(f"ecr describe_repositories {c.ECR_REPOSITORY} -> {exc.response['Error']['Code']}")
        c.mark_deleted("ecr_repository", c.ECR_REPOSITORY)


def delete_role(sess, role_name: str) -> None:
    iam = sess.client("iam")
    try:
        for policy_name in iam.list_role_policies(RoleName=role_name)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
        for attached in iam.list_attached_role_policies(RoleName=role_name)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role_name, PolicyArn=attached["PolicyArn"])
        iam.delete_role(RoleName=role_name)
    except ClientError as exc:
        if not is_missing(exc):
            raise
    try:
        iam.get_role(RoleName=role_name)
        note(f"ERROR: role {role_name} still exists")
    except ClientError as exc:
        note(f"iam get_role {role_name} -> {exc.response['Error']['Code']}")
        c.mark_deleted("iam_role", role_name)


def sweep_by_tag(sess) -> None:
    tagging = sess.client("resourcegroupstaggingapi")
    found: list[str] = []
    for page in tagging.get_paginator("get_resources").paginate(TagFilters=[{"Key": "Project", "Values": [c.PREFIX]}]):
        found += [mapping["ResourceARN"] for mapping in page["ResourceTagMappingList"]]
    note(f"tagging get_resources Project={c.PREFIX} -> {len(found)} resources {found}")


def main() -> None:
    sess = c.session()
    note(f"account {c.account_id()} region {c.REGION}; resources to delete: {[(r['type'], r['name']) for r in c.created_resources()]}")
    delete_function(sess)
    delete_log_group(sess, f"/aws/lambda/{c.FUNCTION_NAME}")
    delete_log_group(sess, f"/aws/codebuild/{c.CODEBUILD_PROJECT}")
    delete_project(sess)
    delete_repository(sess)
    delete_role(sess, c.LAMBDA_ROLE)
    delete_role(sess, c.CODEBUILD_ROLE)
    pending = c.created_resources()
    note(f"resources still marked as created: {[(r['type'], r['name']) for r in pending]}")
    sweep_by_tag(sess)
    c.write_text(c.RESULTS_DIR / "teardown.txt", "\n".join(EVIDENCE) + "\n")


if __name__ == "__main__":
    main()
