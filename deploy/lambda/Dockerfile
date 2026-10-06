# Strands Decider on AWS Lambda (container image, arm64, CPU only).
#
# Built in AWS CodeBuild (see buildspec.yml and scripts/aws_lambda/build_image.py), no local Docker needed.
# The weights (~4.4 GB: Qwen3.5-2B-Base 4.3 GB plus adapter and head 88 MB) are downloaded from
# Hugging Face during the build, at pinned revisions. Lambda's image limit is 10 GB uncompressed.
#
# Manual build (arm64, for AWS):     docker build --platform linux/arm64 -t strands-decider-lambda .
# Manual build (x86, local testing): docker build --platform linux/amd64 -t strands-decider-lambda:x86 .

FROM public.ecr.aws/lambda/python:3.12

ENV HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    XDG_CACHE_HOME=/tmp/cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    PYTHONUNBUFFERED=1

# CPU-only torch on both arm64 and amd64. The PyPI torch wheels for aarch64 are NOT CPU-only:
# torch 2.11.0 pulls ~4 GB of CUDA 13 libraries (nvidia-*), which pushed the image to 10.09 GB,
# right at Lambda's limit. The PyTorch CPU index ships the wheel without those dependencies.
# Versions match env/requirements-lock.txt.
RUN pip install --no-cache-dir "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
      "strands-decider==0.1.0" "transformers==5.18.0" "peft==0.21.2" \
      "accelerate==1.15.0" "huggingface-hub==1.33.0"

# Weights baked into the image at verified revisions. strands-decider 0.1.0 requests both repos
# at their "main" branch; download_weights.py writes refs/main so that, offline, "main" resolves
# to exactly those revisions.
COPY download_weights.py /tmp/download_weights.py
RUN python /tmp/download_weights.py && rm /tmp/download_weights.py

ENV HF_HUB_OFFLINE=1
# From har1101/strands-decider-handson (not verified here): avoids a BLAS loop on aarch64 CPUs.
ENV NVPL_BLAS_DEBUG_CPU_TYPE=1

COPY handler.py ${LAMBDA_TASK_ROOT}/
CMD ["handler.lambda_handler"]
