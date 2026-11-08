"""Capture reference outputs of the BF16 kernels on one GPU.

For each kernel: autotune over every config (rejecting only configs that fail to
compile or exceed hardware limits), then run the tuned config and one held-out config
on every test input, three times each. Development and special outputs are saved as
raw bytes, while sealed outputs keep only their SHA-256. Inputs are not saved: their
SHA-256 is recorded so another machine can regenerate and check them.

    python -m capture.run --name a100-ref-1 --out captures
    python -m capture.run --name smoke --out /tmp/captures --smoke

Exits nonzero if any output fails to repeat, so the capture cannot be a reference.
"""

import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from triton.testing import do_bench

from bf16_kernels import KERNELS, REFERENCE_KERNELS
from capture.inputs import autotune_input, test_inputs

LAUNCHES_PER_INPUT = 3
SAVED_SETS = ("development", "special")
ASSEMBLY_STAGES = ("ttgir", "llir", "ptx", "amdgcn")
SMOKE_CONFIGS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", choices=REFERENCE_KERNELS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def raw_bytes(tensor):
    integer_type = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(integer_type).cpu().numpy().tobytes()


def sha256(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(raw_bytes(tensor))
    return digest.hexdigest()


def as_tuple(inputs):
    return inputs if isinstance(inputs, tuple) else (inputs,)


def to_device(inputs):
    if isinstance(inputs, tuple):
        return tuple(tensor.cuda() for tensor in inputs)
    return inputs.cuda()


def git_state():
    commit = os.environ.get("CAPTURE_GIT_COMMIT")
    dirty = os.environ.get("CAPTURE_GIT_DIRTY")
    if commit is not None:
        return commit, dirty == "1"
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def environment():
    commit, dirty = git_state()
    env = {
        "created": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    driver = Path("/proc/driver/nvidia/version")
    if driver.exists():
        env["nvidia_driver"] = driver.read_text().splitlines()[0]
    return env


def tuning_candidates(kernel_class):
    names = tuple(kernel_class.tuning_options)
    values = itertools.product(*(kernel_class.tuning_options[n] for n in names))
    return [dict(zip(names, combination, strict=True)) for combination in values]


def autotune(kernel_class, smoke):
    inputs = to_device(autotune_input(kernel_class.input_family))
    candidates = tuning_candidates(kernel_class)
    if smoke:
        default = dict(kernel_class.default_config)
        candidates = [default] + [c for c in candidates if c != default][:SMOKE_CONFIGS]
    records = []
    for config in candidates:
        kernel = kernel_class(config)
        try:
            output = kernel.forward(inputs)
            torch.cuda.synchronize()
            latency_ms = do_bench(
                lambda kernel=kernel: kernel.forward(inputs),
                warmup=100,
                rep=200,
                return_mode="median",
            )
        except (OutOfResources, CompilationError) as exc:
            records.append(
                {"config": config, "rejected": f"{type(exc).__name__}: {exc}"}
            )
            continue
        records.append(
            {
                "config": config,
                "latency_ms": latency_ms,
                "output_sha256": sha256(output),
            }
        )
    print(f"  autotuned {len(records)} configs", flush=True)
    return records


def choose_configs(kernel_class, records):
    ranked = sorted(
        (record for record in records if "latency_ms" in record),
        key=lambda record: record["latency_ms"],
    )
    if not ranked:
        raise RuntimeError(f"No config of {kernel_class.__name__} ran.")
    tuned = ranked[0]["config"]
    default = dict(kernel_class.default_config)
    runnable = [record["config"] for record in ranked]
    if default != tuned and default in runnable:
        return {"tuned": tuned, "held_out": default}
    if len(runnable) < 2:
        raise RuntimeError(f"{kernel_class.__name__} has no held-out config.")
    return {"tuned": tuned, "held_out": runnable[1]}


def save_assembly(kernel, directory):
    stages = {}
    for stage in ASSEMBLY_STAGES:
        text = kernel.last_launch.asm.get(stage)
        if text is not None:
            path = directory / f"kernel.{stage}"
            path.write_text(text)
            stages[stage] = path.name
    return stages


def capture_config(kernel_class, config, inputs, directory):
    directory.mkdir(parents=True)
    kernel = kernel_class(config)
    results = []
    for name, input_set, cpu_inputs in inputs:
        device_inputs = to_device(cpu_inputs)
        output = kernel.forward(device_inputs)
        torch.cuda.synchronize()
        first = raw_bytes(output)
        repeats_equal = True
        for _ in range(LAUNCHES_PER_INPUT - 1):
            again = kernel.forward(device_inputs)
            torch.cuda.synchronize()
            repeats_equal &= raw_bytes(again) == first
        result = {
            "name": name,
            "set": input_set,
            "input_sha256": sha256(*as_tuple(cpu_inputs)),
            "output_sha256": hashlib.sha256(first).hexdigest(),
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
            "repeats_equal": repeats_equal,
        }
        if input_set in SAVED_SETS:
            path = directory / f"{name}.bin"
            path.write_bytes(first)
            result["file"] = path.name
        results.append(result)
    return {
        "config": config,
        "assembly": save_assembly(kernel, directory),
        "outputs": results,
    }


def write_checksums(root):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    root = args.out / args.name
    if root.exists():
        raise FileExistsError(f"{root} exists; choose another --name.")
    env = environment()
    if env["git_dirty"] and not args.smoke:
        raise RuntimeError(
            "Uncommitted changes: a reference capture needs a clean tree."
        )
    root.mkdir(parents=True)
    manifest = {"name": args.name, "smoke": args.smoke, "environment": env}
    manifest["kernels"] = {}
    family_inputs = {}
    all_repeat = True
    for kernel_name in args.kernels or REFERENCE_KERNELS:
        print(kernel_name, flush=True)
        kernel_class = KERNELS[kernel_name]
        family = kernel_class.input_family
        if family not in family_inputs:
            family_inputs[family] = list(test_inputs(family, args.smoke))
        records = autotune(kernel_class, args.smoke)
        kernel_record = {"autotune": records, "configs": {}}
        for role, config in choose_configs(kernel_class, records).items():
            print(f"  capturing {role} {config}", flush=True)
            captured = capture_config(
                kernel_class, config, family_inputs[family], root / kernel_name / role
            )
            kernel_record["configs"][role] = captured
            all_repeat &= all(r["repeats_equal"] for r in captured["outputs"])
        manifest["kernels"][kernel_name] = kernel_record
    manifest["all_outputs_repeat"] = all_repeat
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_checksums(root)
    print(f"wrote {root}; all outputs repeat: {all_repeat}")
    return 0 if all_repeat else 1


if __name__ == "__main__":
    sys.exit(main())
