"""Learning lab: dump a small probe kernel, edit its code by hand, and run your edit.

The probe kernels and cases are those of probes/subnormal.py: a 32x32x32 product on
BF16 inputs (`bf16_dot`), on FP32 inputs (`fp32_dot`), or as a loop of FP32
multiply-adds (`fp32_loop`).

    python lab/run.py dump bf16_dot lab/work/bf16_dot
    python lab/run.py run bf16_dot product_subnormal
    python lab/run.py run bf16_dot product_subnormal --edited lab/work/bf16_dot/kernel.amdgcn

`run` prints the distinct output values (as floats and FP32 bit patterns) and whether
they equal the kept or the flushed expected value. With --edited, your .amdgcn or .llir
file replaces what Triton compiled, through TRITON_KERNEL_OVERRIDE.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

WORK_DIR = Path(tempfile.mkdtemp(prefix="lab_"))
os.environ.update(
    TRITON_CACHE_DIR=str(WORK_DIR / "cache"),
    TRITON_DUMP_DIR=str(WORK_DIR / "dump"),
    TRITON_OVERRIDE_DIR=str(WORK_DIR / "override"),
    TRITON_ALWAYS_COMPILE="1",
)

import torch

from probes.subnormal import CASES, PATHS, SIZE, classify

CASE_BY_NAME = {case.__name__.removeprefix("case_"): case for case in CASES}


def launch(kernel, dtype, case_name):
    a, b, kept, flushed = CASE_BY_NAME[case_name]()
    c = torch.empty((SIZE, SIZE), dtype=torch.float32, device="cuda")
    kernel.device_caches.clear()
    handle = kernel[(1,)](a.to(dtype).cuda(), b.to(dtype).cuda(), c, SIZE=SIZE)
    torch.cuda.synchronize()
    return c.cpu(), handle, kept, flushed


def dump(path, out):
    kernel, dtype = PATHS[path]
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    launch(kernel, dtype, "normal_control")
    del os.environ["TRITON_KERNEL_DUMP"]
    out.mkdir(parents=True, exist_ok=True)
    for source in (WORK_DIR / "dump").glob("*/*"):
        if source.suffix in (".llir", ".amdgcn"):
            shutil.copy(source, out / f"kernel{source.suffix}")
            print(f"wrote {out / f'kernel{source.suffix}'}")


def run(path, case_name, edited):
    kernel, dtype = PATHS[path]
    if edited is not None:
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        launch(kernel, dtype, case_name)
        del os.environ["TRITON_KERNEL_DUMP"]
        source = next((WORK_DIR / "dump").glob("*/*.amdgcn"))
        target = WORK_DIR / "override" / source.parent.name
        target.mkdir(parents=True)
        shutil.copy(edited, target / f"{source.stem}{edited.suffix}")
        os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
    output, handle, kept, flushed = launch(kernel, dtype, case_name)
    if edited is not None:
        stage = edited.suffix.removeprefix(".")
        if handle.asm[stage].strip() != edited.read_text().strip():
            raise RuntimeError("Triton did not use your file; the override failed.")
        print(f"ran your edited {edited}")
    values = output.flatten()
    print(
        f"{path} / {case_name}: expected {kept:.6e} if kept, {flushed:.6e} if flushed"
    )
    for value in values.unique()[:8].tolist():
        bits = torch.tensor([value], dtype=torch.float32).view(torch.int32).item()
        count = int((values == value).sum())
        print(
            f"  {value:.9e}  bits 0x{bits & 0xFFFFFFFF:08x}  in {count} of {values.numel()} outputs"
        )
    print(f"  verdict: {classify(output, kept, flushed)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dump_parser = commands.add_parser("dump")
    dump_parser.add_argument("path", choices=sorted(PATHS))
    dump_parser.add_argument("out", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("path", choices=sorted(PATHS))
    run_parser.add_argument("case", choices=sorted(CASE_BY_NAME))
    run_parser.add_argument("--edited", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "dump":
            dump(args.path, args.out)
        else:
            run(args.path, args.case, args.edited)
    finally:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
