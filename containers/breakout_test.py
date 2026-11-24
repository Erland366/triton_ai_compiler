"""Try to break out of the agent container. Run it inside the container:

    containers/agent_mi210.sh RUN_DIR LOGIN_DIR python /repo/containers/breakout_test.py

Each check is something the agent's process must not be able to do (or, for the last
ones, must be able to do for the agent to work). Exits nonzero if any must-fail check
succeeds or any must-work check fails. Network reach is reported, not enforced.
"""

import os
import socket
import sys
from pathlib import Path

HOST_HOME = Path("/vast/users/qirong.ho")
REPO = Path("/repo")


def blocked_read(path):
    try:
        os.listdir(path)
    except OSError as exc:
        return True, f"{type(exc).__name__}"
    return False, "listed"


def blocked_write(path):
    try:
        path.write_text("breakout test\n")
    except OSError as exc:
        return True, f"{type(exc).__name__}"
    path.unlink()
    return False, "written"


def blocked_signal(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True, "no such process in this namespace"
    except PermissionError:
        return True, "PermissionError"
    return False, "process reachable"


def host_processes_hidden():
    names = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                names.append((entry / "comm").read_text().strip())
            except OSError:
                continue
    leaked = sorted(
        {n for n in names if n in ("slurmstepd", "sshd", "tmux: server", "systemd")}
    )
    return (
        not leaked,
        f"{len(names)} processes visible, host daemons: {leaked or 'none'}",
    )


def reachable(host, port=443):
    try:
        socket.create_connection((host, port), timeout=10).close()
    except OSError as exc:
        return f"not reachable directly ({type(exc).__name__})"
    return "reachable directly"


def main():
    must_fail = {
        "read the host home directory": blocked_read(HOST_HOME),
        "read /vast": blocked_read("/vast"),
        "read the host's SSH keys": blocked_read(HOST_HOME / ".ssh"),
        "write into the repository": blocked_write(REPO / "breakout_test_write"),
        "write into the captures": blocked_write(
            REPO / "captures" / "breakout_test_write"
        ),
        "signal the host launcher process": blocked_signal(
            int(os.environ["HOST_PARENT_PID"])
        ),
        "see host processes": host_processes_hidden(),
    }
    must_work = {
        "write the run directory": not blocked_write(Path("/runs/breakout_test_write"))[
            0
        ],
        "write the login directory": not blocked_write(
            Path("/login/breakout_test_write")
        )[0],
        "read the repository": (REPO / "taic_amd" / "agent.py").exists(),
    }
    failed = False
    for check, (blocked, how) in must_fail.items():
        failed |= not blocked
        print(f"{'PASS' if blocked else 'FAIL'} cannot {check}: {how}")
    for check, ok in must_work.items():
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} can {check}")
    import torch

    gpus = torch.cuda.device_count()
    failed |= gpus != 1
    print(f"{'PASS' if gpus == 1 else 'FAIL'} sees exactly one GPU: {gpus}")
    print(f"INFO environment variables passed in: {sorted(os.environ)}")
    for host in ("api.anthropic.com", "example.com"):
        print(
            f"INFO {host}: {reachable(host)}; proxy set: {bool(os.environ.get('https_proxy'))}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
