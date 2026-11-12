#!/usr/bin/env bash
#SBATCH --job-name=subnormal-probe
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe_a100_%j.out
# Subnormal handling of A100's matrix and FMA paths (probes/subnormal.py). It needs the
# image from check_a100.sbatch and a clean checkout. Submit from the repository root:
#
#   mkdir -p logs && sbatch containers/probe_a100.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?error: submit this with sbatch}"
sif="$PWD/containers/triton-cu126.sif"
[[ -f "$sif" ]] || { echo "error: $sif not found; run containers/check_a100.sbatch first" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "error: Slurm did not set CUDA_VISIBLE_DEVICES" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "error: uncommitted changes; a measurement needs a clean checkout" >&2; exit 1; }

commit="$(git rev-parse HEAD)"
echo "job=$SLURM_JOB_ID node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES commit=$commit"
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader
cat "$sif.sha256"

run_tmp="$PWD/logs/tmp_$SLURM_JOB_ID"
mkdir -p "$run_tmp"
trap 'rm -rf "$run_tmp"' EXIT
singularity exec --nv --cleanenv --no-mount tmp,cwd \
  --home "$run_tmp" --bind "$run_tmp:/tmp" --bind "$PWD:$PWD" --pwd "$PWD" \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env "PYTHONPATH=$PWD" \
  --env "TRITON_CACHE_DIR=/tmp/triton-cache" \
  --env "CAPTURE_GIT_COMMIT=$commit" --env "CAPTURE_GIT_DIRTY=0" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  "$sif" python -m probes.subnormal --out "logs/subnormal_a100_$SLURM_JOB_ID.json"
