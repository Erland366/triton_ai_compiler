"""Check that TRITON_KERNEL_OVERRIDE injects LLVM IR and assembly on an AMD GPU.

The kernel computes x + y, so 3 + 5 gives 8 natively, 15 if an injected multiply
ran, and 5 if an injected max ran. Exits nonzero if an injection misbehaves.
"""

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

WORK_DIR = Path(tempfile.mkdtemp(prefix="triton_injection_"))
DUMP_DIR, OVERRIDE_DIR = WORK_DIR / "dump", WORK_DIR / "override"
os.environ.update(
    TRITON_CACHE_DIR=str(WORK_DIR / "cache"),
    TRITON_DUMP_DIR=str(DUMP_DIR),
    TRITON_OVERRIDE_DIR=str(OVERRIDE_DIR),
    TRITON_ALWAYS_COMPILE="1",
)

import torch
import triton
import triton.language as tl

BLOCK = 64
KERNEL_NAME = "add_kernel"
NATIVE, MULTIPLIED, MAXED = 8.0, 15.0, 5.0
FADD_PATTERN = re.compile(
    r"(%\S+) = fadd(?: [a-z]+)* <1 x float> (%[^\s,]+), (%[^\s,]+)"
)
INLINE_MULTIPLY = """%lhs.inline = extractelement <1 x float> \\2, i64 0
  %rhs.inline = extractelement <1 x float> \\3, i64 0
  %mul.inline = call float asm "v_mul_f32 $0, $1, $2", "=v,v,v"(float %lhs.inline, \
float %rhs.inline)
  \\1 = insertelement <1 x float> poison, float %mul.inline, i64 0"""
ADD_PATTERN = re.compile(r"\bv_add_f32(_e32|_e64)?\b")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


def run():
    add_kernel.device_caches.clear()
    x = torch.full((BLOCK,), 3.0, device="cuda", dtype=torch.float32)
    y = torch.full((BLOCK,), 5.0, device="cuda", dtype=torch.float32)
    out = torch.zeros(BLOCK, device="cuda", dtype=torch.float32)
    add_kernel[(1,)](x, y, out, BLOCK, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    values = out.unique().tolist()
    if len(values) != 1:
        raise RuntimeError(f"Expected one distinct output value, got {values}.")
    return values[0]


def expect(name, actual, expected):
    passed = actual == expected
    print(f"{'PASS' if passed else 'FAIL'} {name}: got {actual}, expected {expected}")
    return passed


def inline_multiply(llir):
    injected, count = FADD_PATTERN.subn(INLINE_MULTIPLY, llir)
    if count == 0:
        raise RuntimeError(f"No single-element fadd found in the LLVM IR:\n{llir}")
    return injected


def swap_add(assembly, opcode):
    injected, count = ADD_PATTERN.subn(opcode + r"\1", assembly)
    if count == 0:
        raise RuntimeError(f"No v_add_f32 found in the assembly:\n{assembly}")
    return injected


def inject(kernel_dir, extension, text):
    shutil.rmtree(kernel_dir, ignore_errors=True)
    kernel_dir.mkdir(parents=True)
    (kernel_dir / f"{KERNEL_NAME}.{extension}").write_text(text)


def main():
    print(f"device: {torch.cuda.get_device_name(0)}, triton {triton.__version__}")
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    results = [expect("native", run(), NATIVE)]
    del os.environ["TRITON_KERNEL_DUMP"]

    dumped = list(DUMP_DIR.glob(f"*/{KERNEL_NAME}.amdgcn"))
    if len(dumped) != 1:
        raise RuntimeError(f"Expected one dumped kernel in {DUMP_DIR}, got {dumped}.")
    source_dir = dumped[0].parent
    llir = (source_dir / f"{KERNEL_NAME}.llir").read_text()
    assembly = dumped[0].read_text()
    kernel_dir = OVERRIDE_DIR / source_dir.name

    os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
    inject(kernel_dir, "llir", inline_multiply(llir))
    results.append(expect("llir inline asm", run(), MULTIPLIED))

    inject(kernel_dir, "amdgcn", swap_add(assembly, "v_mul_f32"))
    results.append(expect("amdgcn asm", run(), MULTIPLIED))

    inject(kernel_dir, "amdgcn", swap_add(assembly, "v_max_f32"))
    results.append(expect("edited file, always compile", run(), MAXED))

    del os.environ["TRITON_ALWAYS_COMPILE"]
    inject(kernel_dir, "amdgcn", swap_add(assembly, "v_mul_f32"))
    first = run()
    inject(kernel_dir, "amdgcn", swap_add(assembly, "v_max_f32"))
    second = run()
    print(f"INFO default cache: before edit {first}, after edit {second}")
    if first == second:
        print("WARNING default cache returned a stale kernel after an edit")

    shutil.rmtree(WORK_DIR)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
