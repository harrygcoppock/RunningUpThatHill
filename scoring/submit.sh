#!/bin/bash
# Score a submission on one L4 (a network-isolated SageMaker job, or an EC2 fallback), then print a Markdown report.
# Run from the repo root (main). Works from GitHub Actions or locally with AWS credentials.
# Usage: scoring/submit.sh <dir containing model.py tokenizer.py train.py> <name>
set -euo pipefail
SUB=$1
NAME=$2
: "${AWS_REGION:=eu-west-2}" "${BUCKET:=gpu-poor-golf-970450051890-eu-west-2}" "${MAX_BYTES:=16000000}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IMAGE=763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker
JOB=golf-$NAME-$(date -u +%m%d-%H%M%S)
S3=s3://$BUCKET/submissions/$JOB

aws s3 cp --recursive --quiet "$SUB" "$S3/submission/"
aws s3 cp --quiet eval.py "$S3/referee/eval.py"
aws s3 cp --quiet scoring/sandbox.sh "$S3/referee/sandbox.sh"
aws s3 cp --quiet scoring/sample.py "$S3/referee/sample.py"

channel() { echo "{\"ChannelName\":\"$1\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$2\",\"S3DataDistributionType\":\"FullyReplicated\"}}}"; }
CHANNELS="[$(channel submission "$S3/submission/"),$(channel referee "$S3/referee/"),$(channel train "s3://$BUCKET/data/train/"),$(channel val "s3://$BUCKET/data/val/")]"

# Each run gets one L4: the SageMaker slot if it is free, otherwise an EC2 g6.xlarge (same GPU, same image,
# same sandbox.sh, run by scoring/ec2_job.sh). If every L4 is busy, wait for one.
sagemaker() {
    aws sagemaker create-training-job --training-job-name "$JOB" \
        --role-arn "arn:aws:iam::$ACCOUNT:role/GpuPoorGolfSageMaker" \
        --algorithm-specification "{\"TrainingImage\":\"$IMAGE\",\"TrainingInputMode\":\"File\",\"ContainerEntrypoint\":[\"bash\",\"/opt/ml/input/data/referee/sandbox.sh\",\"$MAX_BYTES\"]}" \
        --input-data-config "$CHANNELS" \
        --output-data-config "{\"S3OutputPath\":\"s3://$BUCKET/output/\"}" \
        --resource-config "{\"InstanceType\":\"ml.g6.xlarge\",\"InstanceCount\":1,\"VolumeSizeInGB\":30,\"KeepAlivePeriodInSeconds\":${KEEP_ALIVE:-0}}" \
        --stopping-condition '{"MaxRuntimeInSeconds":2400}' \
        --enable-network-isolation 2>&1 >/dev/null
}
# Where an EC2 L4 may start: region, AMI (Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261113,
# the same build in both regions), security group (no inbound), then one subnet per zone that offers g6.xlarge.
# Results and data stay in the London bucket; a us-east-1 host pulls the image from its own region.
EC2_PLACES="eu-west-2 ami-05bedcf82aa73215c sg-02511bb4c3452de18 subnet-0334fa408f3687e11 subnet-01d799b3867b3a5ef subnet-05739149f5dee84d3
us-east-1 ami-0a654363d2b9a01db sg-04e0c82334e97feba subnet-05d6cd2dc5fa4bc1d subnet-0f00a86212ef31d32 subnet-0c1d396c906fdb379 subnet-019129493d88c5616 subnet-0779a0a4ade1ab52d"
ec2() {  # prints "<region> <instance id>" if an L4 started anywhere; EC2 problems never fail the run, they just mean "busy"
    local ud region ami sg subnets subnet out
    ud=$(mktemp)
    while read -r region ami sg subnets; do
        { echo '#!/bin/bash'
          printf 'export JOB=%q BUCKET=%q IMAGE=%q MAX_BYTES=%q AWS_DEFAULT_REGION=%q\n' \
              "$JOB" "$BUCKET" "${IMAGE/.$AWS_REGION./.$region.}" "$MAX_BYTES" "$AWS_REGION"
          tail -n +2 scoring/ec2_job.sh; } > "$ud"
        for subnet in $subnets; do
            if out=$(aws ec2 run-instances --region "$region" --subnet-id "$subnet" --image-id "$ami" \
                --instance-type g6.xlarge --iam-instance-profile Name=GpuPoorGolfEC2 --security-group-ids "$sg" \
                --instance-initiated-shutdown-behavior terminate --user-data "file://$ud" \
                --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
                --metadata-options HttpTokens=required,HttpPutResponseHopLimit=1 \
                --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$JOB},{Key=golf,Value=scoring}]" \
                                     "ResourceType=volume,Tags=[{Key=Name,Value=$JOB},{Key=golf,Value=scoring}]" \
                --query 'Instances[0].InstanceId' --output text 2>&1); then
                echo "$region $out"; return 0
            fi
            case "$out" in
                *InsufficientInstanceCapacity*|*VcpuLimitExceeded*) ;;
                *) echo "EC2 $region $subnet: ${out##*: }" >&2 ;;
            esac
        done
    done <<< "$EC2_PLACES"
    return 1
}
while true; do
    if [ -z "${EC2_ONLY:-}" ]; then  # EC2_ONLY=1 skips SageMaker, e.g. to test the EC2 path
        if err=$(sagemaker); then break; fi
        case "$err" in
            *ResourceLimitExceeded*) ;;
            *) echo "$err" >&2; exit 1 ;;
        esac
    fi
    if out=$(ec2); then read -r EC2_REGION INSTANCE <<< "$out"; break; fi
    echo "all L4s busy, retrying in 60s" >&2; sleep 60
done
if [ -n "${INSTANCE:-}" ]; then
    BACKEND=EC2
    echo "started $JOB on EC2 $INSTANCE ($EC2_REGION)" >&2
    for _ in $(seq 100); do  # up to 50 min; the instance stops itself after 40
        [ "$(aws ec2 describe-instances --region "$EC2_REGION" --instance-ids "$INSTANCE" \
            --query 'Reservations[0].Instances[0].State.Name' --output text)" = terminated ] && break
        sleep 30
    done
    aws ec2 terminate-instances --region "$EC2_REGION" --instance-ids "$INSTANCE" >/dev/null || true
    HOST_LOG=$(aws s3 cp --quiet "s3://$BUCKET/output/$JOB/output/host.log" - 2>/dev/null | tail -c 3000 || true)
else
    BACKEND=SageMaker
    echo "started $JOB on SageMaker" >&2
    aws sagemaker wait training-job-completed-or-stopped --training-job-name "$JOB" || true  # a failed job still gets a report
fi

TMP=$(mktemp -d)
aws s3 cp --quiet "s3://$BUCKET/output/$JOB/output/model.tar.gz" "$TMP/" && tar -xzf "$TMP/model.tar.gz" -C "$TMP" || true
HOST_LOG=${HOST_LOG:-} python3 - "$TMP" "$JOB" "$BACKEND" <<'EOF'
import json, os, sys
d, job, backend = sys.argv[1], sys.argv[2], sys.argv[3]
def read(f):
    p = os.path.join(d, f)
    return open(p).read() if os.path.exists(p) else ""
r = json.loads(read("results.json") or "{}")
r["job"] = job
r["backend"] = backend
json.dump(r, open("result.json", "w"))  # for scoring/publish.py
print(f"### GPU-Poor Golf score\n")
if r.get("bpb"):
    print(f"**val bpb: {float(r['bpb']):.4f}**\n")
    print("| size (bytes) | parameters | steps | tokens |\n|---|---|---|---|")
    print(f"| {int(r['size']):,} | {int(r['params'] or 0):,} | {r['steps']} | {int(r['tokens'] or 0):,} |")
    if r.get("sample"):
        print("\n**Rollout** (temperature 0.8):\n\n```text\n" + r["sample"].replace("`", "'") + "\n```")
elif "train_exit" not in r and os.environ.get("HOST_LOG"):  # the EC2 host failed before the sandbox ran
    print("**The scoring machine failed before running your code** (not your fault). Last lines of its log:\n")
    print("```\n" + os.environ["HOST_LOG"] + "\n```")
else:
    stage = "training" if r.get("train_exit", 1) != 0 else "scoring"
    log = read("train.log" if stage == "training" else "eval.log")
    tail = "\n".join(log.strip().splitlines()[-25:])[-3000:]
    print(f"**Failed during {stage}.** Last lines of the log:\n\n```\n{tail}\n```")
print(f"\n<sub>job `{job}` on {backend}</sub>")
EOF
