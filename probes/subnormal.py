"""Test how a GPU's matrix and scalar paths handle FP32 and BF16 subnormals.

Each case is a 32x32x32 matrix product built so that every output has two exactly
known values: the result when subnormals are kept, and the result when they are
flushed to zero. The cases are computed by three Triton paths on the same inputs:

    bf16_dot   tl.dot on BF16 inputs (tensor cores on NVIDIA, MFMA BF16 on AMD)
    fp32_dot   tl.dot on FP32 inputs with input_precision="ieee"
    fp32_loop  a loop over K of FP32 multiply-adds, with no dot instruction

Prints, per path and case, which of the two expected values every output equals,
and the matrix instructions found in the compiled code.

    python -m probes.subnormal --out logs/subnormal_<gpu>.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from capture.run import environment

SIZE = 32
INSTRUCTION = re.compile(
    r"\b(v_mfma\w*|v_dot\w*|v_fma\w*|v_fmac\w*|v_pk_fma\w*|mma\.\S+|wgmma\.\S+|fma\.rn\.f32)"
)


@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, SIZE: tl.constexpr):
    rows = tl.arange(0, SIZE)
    a = tl.load(a_ptr + rows[:, None] * SIZE + rows[None, :])
    b = tl.load(b_ptr + rows[:, None] * SIZE + rows[None, :])
    c = tl.dot(a, b, out_dtype=tl.float32, input_precision="ieee")
    tl.store(c_ptr + rows[:, None] * SIZE + rows[None, :], c)


@triton.jit
def loop_kernel(a_ptr, b_ptr, c_ptr, SIZE: tl.constexpr):
    rows = tl.arange(0, SIZE)
    c = tl.zeros((SIZE, SIZE), dtype=tl.float32)
    for k in range(SIZE):
        a = tl.load(a_ptr + rows * SIZE + k).to(tl.float32)
        b = tl.load(b_ptr + k * SIZE + rows).to(tl.float32)
        c += a[:, None] * b[None, :]
    tl.store(c_ptr + rows[:, None] * SIZE + rows[None, :], c)


def zeros():
    return torch.zeros((SIZE, SIZE), dtype=torch.float64)


def case_product_subnormal():
    # 2^-70 * 2^-70 = 2^-140: each output is one subnormal product.
    a, b = zeros(), zeros()
    a[:, 0], b[0, :] = 2.0**-70, 2.0**-70
    return a, b, 2.0**-140, 0.0


def case_input_subnormal():
    # A BF16 subnormal input times 1.
    a, b = zeros(), zeros()
    a[:, 0], b[0, :] = 2.0**-130, 1.0
    return a, b, 2.0**-130, 0.0


def case_subnormal_products_sum_to_normal():
    # 32 products of 2^-130 each (subnormal) sum to 2^-125 (normal).
    a, b = zeros(), zeros()
    a[:, :], b[:, :] = 2.0**-70, 2.0**-60
    return a, b, 2.0**-125, 0.0


def case_normal_plus_subnormal_product():
    # 2^-126 (normal) + 2^-140 (subnormal product), exact in FP32.
    a, b = zeros(), zeros()
    a[:, 0], b[0, :] = 1.0, 2.0**-126
    a[:, 1], b[1, :] = 2.0**-70, 2.0**-70
    return a, b, 2.0**-126 + 2.0**-140, 2.0**-126


def case_normal_control():
    # No subnormals anywhere: both expected values are equal.
    a, b = zeros(), zeros()
    a[:, 0], b[0, :] = 2.0**-60, 2.0**-60
    return a, b, 2.0**-120, 2.0**-120


CASES = (
    case_product_subnormal,
    case_input_subnormal,
    case_subnormal_products_sum_to_normal,
    case_normal_plus_subnormal_product,
    case_normal_control,
)
PATHS = {
    "bf16_dot": (dot_kernel, torch.bfloat16),
    "fp32_dot": (dot_kernel, torch.float32),
    "fp32_loop": (loop_kernel, torch.bfloat16),
}


def classify(output, kept, flushed):
    values = output.double().flatten()
    if kept == flushed:
        return "matches" if bool((values == kept).all()) else "other values"
    if bool((values == kept).all()):
        return "kept"
    if bool((values == flushed).all()):
        return "flushed"
    return "mixed or other: " + ", ".join(
        f"{v:.6e}" for v in values.unique()[:4].tolist()
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists.")
    report = {"environment": environment(), "paths": {}}
    print(report["environment"]["device"])
    for path, (kernel, dtype) in PATHS.items():
        report["paths"][path] = {"cases": {}}
        instructions = set()
        for case in CASES:
            a, b, kept, flushed = case()
            exact_inputs = a.to(dtype).double().equal(a) and b.to(dtype).double().equal(
                b
            )
            if not exact_inputs:
                raise ValueError(f"{case.__name__} inputs are not exact in {dtype}.")
            a_dev, b_dev = a.to(dtype).cuda(), b.to(dtype).cuda()
            c = torch.empty((SIZE, SIZE), dtype=torch.float32, device="cuda")
            handle = kernel[(1,)](a_dev, b_dev, c, SIZE=SIZE)
            torch.cuda.synchronize()
            for stage in ("amdgcn", "ptx"):
                instructions.update(INSTRUCTION.findall(handle.asm.get(stage, "")))
            verdict = classify(c.cpu(), kept, flushed)
            name = case.__name__.removeprefix("case_")
            report["paths"][path]["cases"][name] = verdict
            print(f"  {path:9} {name:40} {verdict}")
        report["paths"][path]["instructions"] = sorted(instructions)
        print(f"  {path:9} instructions: {sorted(instructions) or 'none'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
