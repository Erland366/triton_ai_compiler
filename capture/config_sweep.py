"""Measure whether a kernel's output bits depend on its config, on this GPU only.

Autotunes every config (as capture/run.py does), then hashes the outputs of this
GPU's tuned config and of a reference capture's tuned and held-out configs on every
test input. Saves no outputs and compares nothing with another vendor.

    python -m capture.config_sweep captures/a100-ref-1 --out logs/sweep.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from bf16_kernels import KERNELS
from capture.inputs import test_inputs
from capture.run import autotune, choose_configs, environment, sha256, to_device


def output_hashes(kernel_class, config, inputs):
    kernel = kernel_class(config)
    hashes = {}
    try:
        for name, _, cpu_inputs in inputs:
            hashes[name] = sha256(kernel.forward(to_device(cpu_inputs)))
    except (OutOfResources, CompilationError) as exc:
        return {"rejected": f"{type(exc).__name__}: {exc}"}
    torch.cuda.synchronize()
    return hashes


def identical(first, second):
    if "rejected" in first or "rejected" in second:
        return None
    return sum(first[name] == second[name] for name in first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists.")
    manifest = json.loads((args.capture / "manifest.json").read_text())
    summary = {"environment": environment(), "capture": manifest["name"]}
    summary["kernels"] = {}
    family_inputs = {}
    for kernel_name, record in manifest["kernels"].items():
        print(kernel_name, flush=True)
        kernel_class = KERNELS[kernel_name]
        family = kernel_class.input_family
        if family not in family_inputs:
            family_inputs[family] = list(test_inputs(family))
        inputs = family_inputs[family]
        records = autotune(kernel_class, smoke=False)
        ran = [r for r in records if "latency_ms" in r]
        configs = {
            "local_tuned": choose_configs(kernel_class, records)["tuned"],
            "reference_tuned": record["configs"]["tuned"]["config"],
            "reference_held_out": record["configs"]["held_out"]["config"],
        }
        hashes = {
            role: output_hashes(kernel_class, config, inputs)
            for role, config in configs.items()
        }
        pairs = {
            f"{a} vs {b}": identical(hashes[a], hashes[b])
            for a, b in (
                ("reference_tuned", "reference_held_out"),
                ("local_tuned", "reference_tuned"),
                ("local_tuned", "reference_held_out"),
            )
        }
        distinct = len({r["output_sha256"] for r in ran})
        print(
            f"  {len(ran)} configs ran, {len(records) - len(ran)} rejected, "
            f"{distinct} distinct output hashes on the autotune input"
        )
        for role, config in configs.items():
            note = " (rejected here)" if "rejected" in hashes[role] else ""
            print(f"  {role}: {config}{note}")
        for pair, count in pairs.items():
            shown = "n/a" if count is None else f"{count} of {len(inputs)}"
            print(f"  identical inputs, {pair}: {shown}")
        summary["kernels"][kernel_name] = {
            "autotune": records,
            "configs": configs,
            "hashes": hashes,
            "identical_inputs": pairs,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
