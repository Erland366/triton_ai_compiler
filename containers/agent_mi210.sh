#!/usr/bin/env bash
# Run a command inside the agent container on one MI210:
#
#   containers/agent_mi210.sh RUN_DIR LOGIN_DIR COMMAND [ARGS...]
#
# Inside, the repository is read-only at /repo, RUN_DIR is the only writable project
# directory (at /runs), and LOGIN_DIR holds the agent's own Claude Code login (at
# /login, used as CLAUDE_CONFIG_DIR). --containall hides the host's $HOME, /vast and
# /tmp, gives the container its own process and IPC namespaces, and starts from a clean
# environment; only the variables below are passed in. Network access is not limited:
# Apptainer cannot do that here without root, so the agent itself has no shell tool.
# HIP_VISIBLE_DEVICES must name the one GPU to use.
set -euo pipefail
[[ $# -ge 3 ]] || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
run_dir="$(realpath "$1")" login_dir="$(realpath "$2")"; shift 2
repo="$(cd "$(dirname "$0")/.." && pwd)"
sif="$repo/containers/agent-rocm.sif"
[[ -f /opt/amdgpu/share/libdrm/amdgpu.ids ]] || { echo "error: /opt/amdgpu/share/libdrm/amdgpu.ids not found on the host" >&2; exit 1; }
[[ -f "$sif" ]] || { echo "error: $sif not found; run containers/build-agent.sh" >&2; exit 1; }
[[ -d "$run_dir" && -d "$login_dir" ]] || { echo "error: RUN_DIR and LOGIN_DIR must exist" >&2; exit 1; }
[[ "${HIP_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || { echo "error: set HIP_VISIBLE_DEVICES to one GPU index" >&2; exit 1; }
case "$run_dir/" in "$repo"/*) echo "error: RUN_DIR must be outside the repository, which is mounted read-only" >&2; exit 1 ;; esac

exec apptainer exec --rocm --containall --no-mount home,cwd \
  --bind "$repo:/repo:ro" --bind "$run_dir:/runs" --bind "$login_dir:/login" \
  --bind "/opt/amdgpu/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids:ro" \
  --pwd /repo \
  --env "PYTHONPATH=/repo" \
  --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
  --env "TAIC_CLAUDE_CONFIG_DIR=/login" \
  --env "HOST_PARENT_PID=$$" \
  --env "https_proxy=${https_proxy:-${HTTPS_PROXY:-}}" --env "HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}" \
  --env "http_proxy=${http_proxy:-${HTTP_PROXY:-}}" --env "no_proxy=${no_proxy:-${NO_PROXY:-}}" \
  "$sif" "$@"
