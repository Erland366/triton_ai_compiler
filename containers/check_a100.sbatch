#!/usr/bin/env bash
#SBATCH --job-name=bf16-check
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/check_a100_%j.out
# Sanity and repeat check of the BF16 kernels on one A100, building the image if needed.
# Memory and CPUs are explicit because this cluster's default is about 2 GB and 2 CPUs.
# Submit from the repository root:
#
#   mkdir -p logs && sbatch containers/check_a100.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?error: submit this with sbatch}"
cuda_tag=cu126
sif="$PWD/containers/triton-$cuda_tag.sif"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "error: Slurm did not set CUDA_VISIBLE_DEVICES" >&2; exit 1; }
echo "job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse HEAD)"
git status --porcelain --untracked-files=no
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader

[[ -f "$sif" ]] || containers/build-cuda.sh "$cuda_tag"
cat "$sif.sha256"

# No host $HOME or /tmp in the container, and only the GPU Slurm assigned: jobs on this
# cluster can see every GPU on the node.
run_tmp="$PWD/logs/tmp_$SLURM_JOB_ID"
mkdir -p "$run_tmp"
trap 'rm -rf "$run_tmp"' EXIT
singularity exec --nv --cleanenv --no-mount tmp,cwd \
  --home "$run_tmp" --bind "$run_tmp:/tmp" --bind "$PWD:$PWD" --pwd "$PWD" \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env "PYTHONPATH=$PWD" \
  --env "TRITON_CACHE_DIR=/tmp/triton-cache" \
  "$sif" python amd/check_bf16_kernels.py --repeats 20
