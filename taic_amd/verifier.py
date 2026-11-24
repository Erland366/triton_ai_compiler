"""Verify a candidate kernel in a subprocess with a timeout, and summarize the result.

    python -m taic_amd.verifier --kernel K --capture captures/a100-ref-1 \
        [--candidate kernel.llir] [--dump-native dir] [--timeout 600]

Without --candidate it scores native AMD Triton, which is the agent's starting point;
--dump-native also saves that native kernel's .llir and .amdgcn.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def verify(kernel, capture, candidate=None, config=None, dump_native=None, timeout=600):
    with tempfile.TemporaryDirectory(prefix="taic_verify_") as directory:
        out = Path(directory) / "report.json"
        command = [
            sys.executable, "-m", "taic_amd.evaluate",
            "--kernel", kernel, "--capture", str(capture), "--out", str(out),
        ]  # fmt: skip
        if candidate is not None:
            command += ["--candidate", str(candidate)]
        if config is not None:
            command += ["--config", json.dumps(config)]
        if dump_native is not None:
            command += ["--dump-native", str(dump_native)]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        try:
            completed = subprocess.run(
                command, cwd=REPO, env=env, capture_output=True, text=True,
                timeout=timeout, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"No result within {timeout} s."}
        if not out.exists():
            return {
                "status": "crashed",
                "error": f"Exit code {completed.returncode}.",
                "stderr": completed.stderr[-4000:],
            }
        return json.loads(out.read_text())


def summary(report):
    if report["status"] != "ok":
        detail = report.get("traceback") or report.get("stderr") or ""
        tail = "\n".join(detail.strip().splitlines()[-6:])
        return f"{report['status']}: {report.get('error')}\n{tail}".rstrip()
    visible = report["visible"]
    line = (
        f"exact={report['exact']} repeats={report['repeats']} "
        f"sealed_all_equal={report['sealed_all_equal']} | visible: "
        f"{visible['exact_inputs']}/{visible['inputs']} inputs exact, "
        f"{visible['bit_equal_fraction']:.4%} of values bit-equal, "
        f"{visible['within_1_step_fraction']:.4%} within 1 step, "
        f"max {visible['max_steps']} steps"
    )
    if "latency_ms" in report:
        line += f" | latency {report['latency_ms']:.4f} ms"
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--config", type=json.loads)
    parser.add_argument("--dump-native", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify(
        args.kernel,
        args.capture.resolve(),
        args.candidate.resolve() if args.candidate else None,
        args.config,
        args.dump_native.resolve() if args.dump_native else None,
        args.timeout,
    )
    print(json.dumps(report, indent=2) if args.json else summary(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
