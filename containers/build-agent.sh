#!/usr/bin/env bash
# Build the agent image for MI210 with Apptainer:
#
#   containers/build-agent.sh
#
# Writes containers/agent-rocm.sif, its .sha256, and agent-rocm.frozen.txt (commit the
# last one). Temporary files stay in RAM (/dev/shm), because --fakeroot cannot use NFS
# for them, and are removed on exit.
set -euo pipefail
cd "$(dirname "$0")"
build_tmp="$(mktemp -d /dev/shm/agent-build.XXXXXX)"
trap 'rm -rf "$build_tmp"' EXIT
export APPTAINER_TMPDIR="$build_tmp" TMPDIR="$build_tmp"
export APPTAINER_CACHEDIR="$PWD/.singularity-cache"
mkdir -p "$APPTAINER_CACHEDIR"
apptainer build --fakeroot --force agent-rocm.sif.partial agent-rocm.def
mv agent-rocm.sif.partial agent-rocm.sif
sha256sum agent-rocm.sif > agent-rocm.sif.sha256
apptainer exec --cleanenv agent-rocm.sif cat /opt/locks/frozen.txt > agent-rocm.frozen.txt
