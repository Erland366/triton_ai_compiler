"""Evaluate one candidate kernel against a reference capture, in this process.

A candidate is a kernel's LLVM IR (`.llir`) or AMDGPU assembly (`.amdgcn`), injected
with Triton's TRITON_KERNEL_OVERRIDE in place of what Triton would compile. Run it
through taic_amd.verifier, which starts it as a subprocess with a timeout, so a crash
or hang cannot take the caller down. Writes a JSON report to --out.

Visible inputs (development and special) get full detail. Sealed inputs only report
whether all of them matched. Latency is measured only for a bit-exact candidate.
With --quick, only four development inputs run, once each, with no sealed inputs and
no latency: a fast check for trying ideas, which can never mark a candidate exact.

    python -m taic_amd.evaluate --kernel K --candidate kernel.llir \
        --capture captures/a100-ref-1 --out report.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

WORK_DIR = Path(tempfile.mkdtemp(prefix="taic_amd_"))
os.environ.update(
    TRITON_CACHE_DIR=str(WORK_DIR / "cache"),
    TRITON_DUMP_DIR=str(WORK_DIR / "dump"),
    TRITON_OVERRIDE_DIR=str(WORK_DIR / "override"),
    TRITON_ALWAYS_COMPILE="1",
)

import torch
from triton.testing import do_bench

from bf16_kernels import KERNELS
from capture.compare import compare_values
from capture.inputs import test_inputs
from capture.run import raw_bytes, to_device

STAGES = ("llir", "amdgcn")
VISIBLE_SETS = ("development", "special")
LAUNCHES_PER_INPUT = 2
# A quick check: one random and three adversarial development inputs, launched once.
QUICK_INPUTS = tuple(f"development_{index:02d}_" for index in (0, 8, 10, 15))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--config", type=json.loads)
    parser.add_argument("--role", default="tuned", choices=("tuned", "held_out"))
    parser.add_argument("--dump-native", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def launch(kernel, inputs):
    output = kernel.forward(to_device(inputs))
    torch.cuda.synchronize()
    return output


def dump_native(kernel, inputs):
    """Compile natively once, returning the dump directory with .llir and .amdgcn."""
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    try:
        type(kernel).kernel.device_caches.clear()
        launch(kernel, inputs)
    finally:
        del os.environ["TRITON_KERNEL_DUMP"]
    dumped = list((WORK_DIR / "dump").glob("*/*.amdgcn"))
    if len(dumped) != 1:
        raise RuntimeError(f"Expected one dumped kernel, found {dumped}.")
    return dumped[0].parent


def inject(kernel, source_dir, candidate):
    stage = candidate.suffix.removeprefix(".")
    if stage not in STAGES:
        raise ValueError(f"Candidate must end in .llir or .amdgcn, got {candidate}.")
    stem = next(source_dir.glob("*.amdgcn")).stem
    target = WORK_DIR / "override" / source_dir.name
    target.mkdir(parents=True)
    text = candidate.read_text()
    (target / f"{stem}.{stage}").write_text(text)
    os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
    type(kernel).kernel.device_caches.clear()
    return stage, text


def score(kernel, reference, capture_dir, inputs, launches):
    by_name = {output["name"]: output for output in reference["outputs"]}
    visible, sealed_all_equal, repeats = [], True, True
    for name, input_set, cpu_inputs in inputs:
        expected = by_name[name]
        first = raw_bytes(launch(kernel, cpu_inputs))
        for _ in range(launches - 1):
            repeats &= raw_bytes(launch(kernel, cpu_inputs)) == first
        if input_set not in VISIBLE_SETS:
            sealed_all_equal &= (
                hashlib.sha256(first).hexdigest() == expected["output_sha256"]
            )
            continue
        saved = (capture_dir / expected["file"]).read_bytes()
        result = compare_values(first, saved, expected["output_dtype"])
        result["name"] = name
        visible.append(result)
    values = sum(r["values"] for r in visible)
    return {
        "repeats": repeats,
        "visible": {
            "inputs": len(visible),
            "exact_inputs": sum(r["bit_equal"] == r["values"] for r in visible),
            "bit_equal_fraction": sum(r["bit_equal"] for r in visible) / values,
            "within_1_step_fraction": sum(r["within_1_step"] for r in visible) / values,
            "max_steps": max(r["max_steps"] for r in visible),
            "per_input": visible,
        },
        "sealed_all_equal": sealed_all_equal,
    }


def evaluate(args):
    manifest = json.loads((args.capture / "manifest.json").read_text())
    reference = manifest["kernels"][args.kernel]["configs"][args.role]
    kernel_class = KERNELS[args.kernel]
    kernel = kernel_class(args.config or reference["config"])
    inputs = list(test_inputs(kernel_class.input_family, manifest["smoke"]))
    source_dir = dump_native(kernel, inputs[0][2])
    if args.dump_native is not None:
        args.dump_native.mkdir(parents=True, exist_ok=True)
        for path in source_dir.glob("*"):
            if path.suffix.removeprefix(".") in STAGES:
                shutil.copy(path, args.dump_native / f"kernel{path.suffix}")
    report = {
        "kernel": args.kernel,
        "config": kernel.config,
        "capture": manifest["name"],
    }
    if args.candidate is None:
        report["candidate"] = "native"
    else:
        stage, text = inject(kernel, source_dir, args.candidate)
        report["candidate"] = str(args.candidate)
        launch(kernel, inputs[0][2])
        compiled = kernel.last_launch.asm[stage]
        if compiled.strip() != text.strip():
            raise RuntimeError("Triton did not use the candidate; the override failed.")
    if args.quick:
        inputs = [entry for entry in inputs if entry[0].startswith(QUICK_INPUTS)]
        report.update(
            score(kernel, reference, args.capture / args.kernel / args.role, inputs, 1)
        )
        report.update(quick=True, exact=False, repeats=None, sealed_all_equal=None)
        report["status"] = "ok"
        return report
    report.update(
        score(
            kernel,
            reference,
            args.capture / args.kernel / args.role,
            inputs,
            LAUNCHES_PER_INPUT,
        )
    )
    report["quick"] = False
    report["exact"] = (
        report["repeats"]
        and report["sealed_all_equal"]
        and report["visible"]["exact_inputs"] == report["visible"]["inputs"]
    )
    if report["exact"]:
        device_inputs = to_device(inputs[0][2])
        report["latency_ms"] = do_bench(
            lambda: kernel.forward(device_inputs),
            warmup=100,
            rep=200,
            return_mode="median",
        )
    report["status"] = "ok"
    return report


def main():
    args = parse_args()
    try:
        report = evaluate(args)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a report for the agent
        report = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
    finally:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
