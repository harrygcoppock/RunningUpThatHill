#!/bin/bash
# Boot script for an EC2 L4 (g6.xlarge) that scores one submission when the SageMaker slot is busy.
# It reproduces what SageMaker does: the same image, the same channel layout, sandbox.sh unchanged,
# and the container gets no network (so no internet and no access to this host's AWS credentials).
# submit.sh prepends JOB, BUCKET, IMAGE and MAX_BYTES, then passes the result as user data.
set -uo pipefail
shutdown -h +40  # hard stop, whatever happens below; the instance terminates on shutdown
exec > /var/log/golf.log 2>&1
S3=s3://$BUCKET/submissions/$JOB
OUT=s3://$BUCKET/output/$JOB/output
D=/golf/input/data
M=/golf/model
mkdir -p $D $M

finish() {
    tar -czf /golf/model.tar.gz -C $M .
    aws s3 cp --quiet /golf/model.tar.gz "$OUT/model.tar.gz"
    aws s3 cp --quiet /var/log/golf.log "$OUT/host.log"
    shutdown -h now
}
trap finish EXIT

for c in submission referee; do aws s3 cp --quiet --recursive "$S3/$c/" "$D/$c/"; done
for c in train val; do aws s3 cp --quiet --recursive "s3://$BUCKET/data/$c/" "$D/$c/"; done

REGISTRY=${IMAGE%%/*}  # 763104351884.dkr.ecr.<region>.amazonaws.com
aws ecr get-login-password --region "$(echo "$REGISTRY" | cut -d. -f4)" | docker login --username AWS --password-stdin "$REGISTRY"
docker pull --quiet "$IMAGE" || { echo "golf: could not pull the scoring image"; df -h /; exit 1; }
echo "golf: image ready at $(date -u +%T)"

docker run --rm --gpus all --network none --shm-size 8g \
    -v $D:/opt/ml/input/data -v $M:/opt/ml/model \
    --entrypoint bash "$IMAGE" /opt/ml/input/data/referee/sandbox.sh "$MAX_BYTES"
echo "golf: sandbox exited $? at $(date -u +%T)"
