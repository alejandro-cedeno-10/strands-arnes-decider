"""Benchmark Strands Decider CPU variants on a temporary EC2 Graviton instance, driven through SSM.

No SSH keys and no inbound rules: the instance only needs outbound internet to install packages
and download the weights. Every resource is tagged and recorded in results/aws_ec2/resources.json.

Usage:
    python scripts/aws_ec2/benchmark.py launch --subnet subnet-... [--instance-type c7g.2xlarge]
    python scripts/aws_ec2/benchmark.py run [--threads 2 4 8]
    python scripts/aws_ec2/benchmark.py finetune
    python scripts/aws_ec2/benchmark.py teardown

Extra tags required by your organization: EC2_BENCH_EXTRA_TAGS='{"Owner": "..."}'.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PREFIX = "strands-decider-bench"
REGION = os.getenv("AWS_REGION", "us-east-1")
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "aws_ec2"
RESOURCES_FILE = RESULTS_DIR / "resources.json"
AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
TAGS = {"Project": PREFIX, "Environment": "test", "IsTemporary": "true",
        **json.loads(os.getenv("EC2_BENCH_EXTRA_TAGS", "{}"))}
SETUP_SCRIPT = """set -euxo pipefail
dnf install -y python3.12 python3.12-pip git
python3.12 -m venv /opt/venv
/opt/venv/bin/pip install --quiet torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
/opt/venv/bin/pip install --quiet strands-decider==0.1.0
mkdir -p /opt/bench/benchmarks /opt/bench/deploy/lambda
"""


def session() -> boto3.Session:
    return boto3.Session(region_name=REGION)


def tag_list(name: str) -> list[dict]:
    return [{"Key": k, "Value": v} for k, v in {**TAGS, "Name": name}.items()]


def load_resources() -> dict:
    return json.loads(RESOURCES_FILE.read_text(encoding="utf-8")) if RESOURCES_FILE.exists() else {}


def save_resources(resources: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCES_FILE.write_text(json.dumps(resources, indent=1), encoding="utf-8")


def ensure_instance_profile(iam, resources: dict) -> str:
    """Role and instance profile with only AmazonSSMManagedInstanceCore."""
    name = f"{PREFIX}-ssm-role"
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                        Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    resources["role"] = name
    save_resources(resources)
    iam.attach_role_policy(RoleName=name, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    try:
        iam.create_instance_profile(InstanceProfileName=name, Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])
        iam.add_role_to_instance_profile(InstanceProfileName=name, RoleName=name)
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    resources["instance_profile"] = name
    save_resources(resources)
    time.sleep(15)
    return name


def launch(subnet: str, instance_type: str) -> None:
    s = session()
    ec2, iam, ssm = s.client("ec2"), s.client("iam"), s.client("ssm")
    resources = load_resources()
    vpc = ec2.describe_subnets(SubnetIds=[subnet])["Subnets"][0]["VpcId"]
    group = resources.get("security_group") or ec2.create_security_group(
        GroupName=f"{PREFIX}-sg", Description="Decider benchmark: no inbound, outbound only", VpcId=vpc,
        TagSpecifications=[{"ResourceType": "security-group", "Tags": tag_list(f"{PREFIX}-sg")}])["GroupId"]
    resources["security_group"] = group
    save_resources(resources)
    profile = ensure_instance_profile(iam, resources)
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    for attempt in range(6):
        try:
            instance = ec2.run_instances(
                ImageId=ami, InstanceType=instance_type, MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": profile},
                NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [group],
                                    "AssociatePublicIpAddress": True}],
                BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                                      "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                MetadataOptions={"HttpTokens": "required"},
                InstanceInitiatedShutdownBehavior="terminate",
                TagSpecifications=[{"ResourceType": t, "Tags": tag_list(PREFIX)} for t in ("instance", "volume")],
            )["Instances"][0]["InstanceId"]
            break
        except ClientError as exc:
            if "Invalid IamInstanceProfile" not in str(exc) or attempt == 5:
                raise
            time.sleep(10)
    resources["instance"] = instance
    save_resources(resources)
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance])
    print(json.dumps({"instance": instance, "instance_type": instance_type}))


def run_command(ssm, instance: str, script: str, timeout_s: int) -> str:
    command = ssm.send_command(InstanceIds=[instance], DocumentName="AWS-RunShellScript",
                               Parameters={"commands": [script], "executionTimeout": [str(timeout_s)]},
                               TimeoutSeconds=600)["Command"]["CommandId"]
    while True:
        time.sleep(10)
        try:
            result = ssm.get_command_invocation(CommandId=command, InstanceId=instance)
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if result["Status"] not in ("Pending", "InProgress", "Delayed"):
            if result["Status"] != "Success":
                raise RuntimeError(f"{result['Status']}: {result['StandardErrorContent'][-3000:]}")
            return result["StandardOutputContent"]


def wait_for_ssm(ssm, instance: str) -> None:
    for _ in range(60):
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [instance]}])
        if info["InstanceInformationList"]:
            return
        time.sleep(10)
    raise TimeoutError("instance never registered with SSM")


def upload_file(local: Path, remote: str) -> str:
    encoded = base64.b64encode(local.read_bytes()).decode()
    return f"echo {encoded} | base64 -d > {remote}"


def push_file(ssm, instance: str, local: Path, remote: str, chunk: int = 30_000) -> None:
    """Upload a file of any size through SSM, in base64 chunks appended on the instance."""
    encoded = base64.b64encode(local.read_bytes()).decode()
    run_command(ssm, instance, f"mkdir -p $(dirname {remote}) && rm -f {remote}.b64", 120)
    for start in range(0, len(encoded), chunk):
        run_command(ssm, instance, f"echo -n {encoded[start:start + chunk]} >> {remote}.b64", 120)
    run_command(ssm, instance, f"base64 -d {remote}.b64 > {remote} && rm {remote}.b64", 120)


FINETUNE_FILES = ["finetune/build_examples.py", "finetune/tune_head.py", "finetune/evaluate.py",
                  "examples/g_devops_self_healing.py", "deploy/lambda/download_weights.py",
                  "deploy/ec2/serve.py", "deploy/ec2/smoke_test.py",
                  "results/finetune/train.jsonl", "results/finetune/calib.jsonl",
                  "results/finetune/test_alerts.jsonl", "results/finetune/holdout_alerts.jsonl"]
FINETUNE_STEPS = [
    ("tune_head", "python finetune/tune_head.py --out results/finetune/checkpoint", 2400),
    ("eval_tuned", "python finetune/evaluate.py --checkpoint results/finetune/checkpoint --name tuned", 1800),
    ("eval_v19", "python finetune/evaluate.py --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 --name v19", 1800),
    ("serve_smoke", "\n".join([
        "nohup python deploy/ec2/serve.py --checkpoint results/finetune/checkpoint > /tmp/serve.log 2>&1 &",
        "for i in $(seq 150); do curl -sf 127.0.0.1:8099/health > /dev/null && break; sleep 2; done",
        "python deploy/ec2/smoke_test.py --clients 8 --requests 24",
        "pkill -f deploy/ec2/serve.py",
    ]), 1800),
]


def finetune() -> None:
    ssm = session().client("ssm")
    instance = load_resources()["instance"]
    wait_for_ssm(ssm, instance)
    log = RESULTS_DIR / "finetune.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_command(ssm, instance, SETUP_SCRIPT, 1800)
    for relative in FINETUNE_FILES:
        push_file(ssm, instance, ROOT / relative, f"/opt/bench/{relative}")
    env = "cd /opt/bench\nexport HF_HOME=/opt/hf HF_HUB_OFFLINE=1 PATH=/opt/venv/bin:$PATH\n"
    run_command(ssm, instance, env.replace(" HF_HUB_OFFLINE=1", "") + "python deploy/lambda/download_weights.py", 1800)
    for name, command, timeout in FINETUNE_STEPS:
        output = run_command(ssm, instance, env + f"{{\n{command}\n}} 2>&1 | grep -vE 'Loading weights|Warning|warn' | tail -40", timeout)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"### {name}\n{output}\n")
        print(f"### {name}\n{output[-4000:]}", flush=True)


def run(threads: list[int]) -> None:
    ssm = session().client("ssm")
    instance = load_resources()["instance"]
    wait_for_ssm(ssm, instance)
    run_command(ssm, instance, SETUP_SCRIPT, 1800)
    uploads = " && ".join([
        upload_file(ROOT / "benchmarks" / "cpu_variants.py", "/opt/bench/benchmarks/cpu_variants.py"),
        upload_file(ROOT / "deploy" / "lambda" / "download_weights.py", "/opt/bench/deploy/lambda/download_weights.py"),
    ])
    run_command(ssm, instance, f"{uploads} && cd /opt/bench && HF_HOME=/opt/hf /opt/venv/bin/python deploy/lambda/download_weights.py", 1800)
    lines = []
    for variant in ("fp32", "bf16", "int8"):
        output = run_command(ssm, instance, (
            f"cd /opt/bench && HF_HOME=/opt/hf HF_HUB_OFFLINE=1 /opt/venv/bin/python benchmarks/cpu_variants.py "
            f"--variant {variant} --threads {' '.join(map(str, threads))} --warm 10 2>/dev/null; "
            f"cat results/cpu_variants_{variant}.json | head -c 1 > /dev/null"), 3000)
        rows = [line for line in output.splitlines() if line.startswith("{")]
        lines += rows
        print("\n".join(rows), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "cpu_variants.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def teardown() -> None:
    s = session()
    ec2, iam = s.client("ec2"), s.client("iam")
    resources = load_resources()
    if "instance" in resources:
        ec2.terminate_instances(InstanceIds=[resources["instance"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[resources["instance"]])
        print("instance terminated")
    if "security_group" in resources:
        ec2.delete_security_group(GroupId=resources["security_group"])
        print("security group deleted")
    if "instance_profile" in resources:
        try:
            iam.remove_role_from_instance_profile(InstanceProfileName=resources["instance_profile"], RoleName=resources["role"])
        except ClientError:
            pass
        iam.delete_instance_profile(InstanceProfileName=resources["instance_profile"])
        print("instance profile deleted")
    if "role" in resources:
        for policy in iam.list_attached_role_policies(RoleName=resources["role"])["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=resources["role"], PolicyArn=policy["PolicyArn"])
        iam.delete_role(RoleName=resources["role"])
        print("role deleted")
    left = s.client("resourcegroupstaggingapi").get_resources(TagFilters=[{"Key": "Project", "Values": [PREFIX]}])
    active = [r["ResourceARN"].split(":")[-1] for r in left["ResourceTagMappingList"]]
    print(json.dumps({"tagged_resources_left": active}))
    RESOURCES_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--subnet", required=True)
    launch_parser.add_argument("--instance-type", default="c7g.2xlarge")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--threads", nargs="+", type=int, default=[2, 4, 8])
    sub.add_parser("finetune")
    sub.add_parser("teardown")
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.subnet, args.instance_type)
    elif args.command == "run":
        run(args.threads)
    elif args.command == "finetune":
        finetune()
    else:
        teardown()


if __name__ == "__main__":
    main()
