"""Test whether flushing FP32 subnormals changes how native MatMul compares with A100.

Runs both MatMul kernels at the reference's tuned config on three development inputs,
with Triton's AMD option allow_flush_denorm off (its default, IEEE subnormals) and on
(flush to zero), and compares each output with the saved reference bytes.

    python amd/denorm_probe.py captures/a100-ref-1
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import triton

from bf16_kernels import KERNELS
from capture.compare import compare_values
from capture.inputs import test_inputs
from capture.run import raw_bytes, to_device

KERNEL_NAMES = (
    "MatrixMultiplicationBF16OutputBF16",
    "MatrixMultiplicationBF16OutputFloat32",
)
INPUT_NAMES = (
    "development_00_random",
    "development_08_subnormals",
    "development_15_tiny",
)
DENORM_MODE = re.compile(r"\.amdhsa_float_denorm_mode_32\s+(\d+)")


def launch(kernel, inputs, flush):
    a, b = inputs
    c = torch.empty(
        (a.shape[0], b.shape[1]), device=a.device, dtype=kernel.output_dtype
    )
    config = kernel.config
    handle = kernel.kernel[
        (
            triton.cdiv(a.shape[0], config["block_m"]),
            triton.cdiv(b.shape[1], config["block_n"]),
        )
    ](
        a,
        b,
        c,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        a.shape[1],
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"],
        num_warps=config["num_warps"],
        allow_flush_denorm=flush,
    )
    torch.cuda.synchronize()
    mode = DENORM_MODE.search(handle.asm["amdgcn"])
    return c, mode.group(1) if mode else "not found"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    inputs = {
        name: tensors
        for name, _, tensors in test_inputs("matmul")
        if name in INPUT_NAMES
    }
    for kernel_name in KERNEL_NAMES:
        reference = manifest["kernels"][kernel_name]["configs"]["tuned"]
        outputs = {o["name"]: o for o in reference["outputs"]}
        kernel = KERNELS[kernel_name](reference["config"])
        kernel.output_dtype = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }[outputs[INPUT_NAMES[0]]["output_dtype"]]
        print(f"{kernel_name} {reference['config']}")
        for flush in (False, True):
            for name in INPUT_NAMES:
                output, mode = launch(kernel, to_device(inputs[name]), flush)
                saved = args.capture / kernel_name / "tuned" / outputs[name]["file"]
                result = compare_values(
                    raw_bytes(output),
                    saved.read_bytes(),
                    outputs[name]["output_dtype"],
                )
                print(
                    f"  flush={flush!s:5} denorm_mode_32={mode} {name[15:]:11} "
                    f"bit-equal {result['bit_equal'] / result['values']:8.3%}  "
                    f"within 1 step {result['within_1_step'] / result['values']:8.3%}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
