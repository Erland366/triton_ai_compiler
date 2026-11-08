#!/usr/bin/env bash
# Build the CUDA image with Singularity:
#
#   containers/build-cuda.sh CUDA_TAG
#
# CUDA_TAG is the PyTorch wheel index the node's driver supports (cu126 on driver 570).
# Writes triton-CUDA_TAG.sif, its .sha256, and triton-CUDA_TAG.frozen.txt (commit the
# last one). Temporary files stay in RAM (/dev/shm), because --fakeroot cannot use NFS for
# them, and are removed on exit.
set -euo pipefail
[[ $# -eq 1 ]] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
cuda_tag="$1"
[[ "$cuda_tag" =~ ^cu[0-9]+$ ]] || { echo "error: CUDA_TAG must look like cu126, got '$cuda_tag'" >&2; exit 1; }
cd "$(dirname "$0")"

min_shm_gb=30
shm_gb="$(df --output=avail -BG /dev/shm | tail -1 | tr -dc 0-9)"
(( shm_gb >= min_shm_gb )) || { echo "error: /dev/shm has ${shm_gb} GB free, need ${min_shm_gb} GB for the build" >&2; exit 1; }

build_tmp="$(mktemp -d /dev/shm/triton-build.XXXXXX)"
trap 'rm -rf "$build_tmp"' EXIT
export SINGULARITY_TMPDIR="$build_tmp" TMPDIR="$build_tmp"
export SINGULARITY_CACHEDIR="$PWD/.singularity-cache"
mkdir -p "$SINGULARITY_CACHEDIR"

# Build under a temporary name, so an interrupted build never leaves a file that
# check_a100.sbatch would take for a finished image.
sif="triton-$cuda_tag.sif"
partial="$sif.partial"
singularity build --fakeroot --force --build-arg "CUDA_TAG=$cuda_tag" "$partial" triton-cuda.def
mv "$partial" "$sif"
sha256sum "$sif" > "$sif.sha256"
singularity exec --cleanenv "$sif" cat /opt/locks/frozen.txt > "triton-$cuda_tag.frozen.txt"
