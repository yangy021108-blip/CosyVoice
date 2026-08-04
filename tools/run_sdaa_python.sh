#!/usr/bin/env bash
set -eo pipefail

source /opt/tecoai/setvars.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env_py310
set -u

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /workspace/CosyVoice
exec python "$@"
