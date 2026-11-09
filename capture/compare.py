"""Compare this GPU's native Triton output with a reference capture, bit for bit.

For each kernel and each config in the capture, runs that config here on the same
regenerated test inputs. Development and special outputs are compared value by value
against the saved reference bytes; sealed outputs only by SHA-256. Run
capture.check_inputs first: this assumes the inputs regenerate identically.

    python -m capture.compare captures/a100-ref-1 --out logs/compare.json

Differences are counted in representable steps: adjacent values of the output type
differ by one step, and +0 and -0 by zero steps (their bits still count as unequal).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bf16_kernels import KERNELS
from capture.inputs import test_inputs
from capture.run import environment, raw_bytes, to_device

INTEGER_TYPES = {"torch.bfloat16": np.int16, "torch.float32": np.int32}
FLOAT_TYPES = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}


def ordered(bits):
    wide = bits.astype(np.int64)
    magnitude = wide & ((1 << (8 * bits.itemsize - 1)) - 1)
    return np.where(wide < 0, -magnitude, magnitude)


def compare_values(actual, expected, dtype):
    integer_type = INTEGER_TYPES[dtype]
    got = np.frombuffer(actual, dtype=integer_type)
    want = np.frombuffer(expected, dtype=integer_type)
    as_float = lambda bits: torch.from_numpy(bits.copy()).view(FLOAT_TYPES[dtype])
    nan_got = torch.isnan(as_float(got).float()).numpy()
    nan_want = torch.isnan(as_float(want).float()).numpy()
    both_numbers = ~nan_got & ~nan_want
    steps = np.abs(ordered(got) - ordered(want))[both_numbers]
    return {
        "values": int(got.size),
        "bit_equal": int((got == want).sum()),
        "nan_positions_agree": bool((nan_got == nan_want).all()),
        "nan_count_reference": int(nan_want.sum()),
        "nan_count_here": int(nan_got.sum()),
        "within_1_step": int((steps <= 1).sum()),
        "max_steps": int(steps.max()) if steps.size else 0,
    }


def compare_config(kernel_class, reference, directory, inputs):
    kernel = kernel_class(reference["config"])
    by_name = {output["name"]: output for output in reference["outputs"]}
    results = []
    for name, input_set, cpu_inputs in inputs:
        expected = by_name[name]
        actual = raw_bytes(kernel.forward(to_device(cpu_inputs)))
        torch.cuda.synchronize()
        result = {
            "name": name,
            "set": input_set,
            "hash_equal": hashlib.sha256(actual).hexdigest()
            == expected["output_sha256"],
        }
        if "file" in expected:
            saved = (directory / expected["file"]).read_bytes()
            result.update(compare_values(actual, saved, expected["output_dtype"]))
        results.append(result)
    return results


def summarize(results):
    lines = {}
    for input_set in ("development", "sealed", "special"):
        chosen = [r for r in results if r["set"] == input_set]
        line = f"{sum(r['hash_equal'] for r in chosen)}/{len(chosen)} identical"
        if input_set != "sealed":
            values = sum(r["values"] for r in chosen)
            equal = sum(r["bit_equal"] for r in chosen)
            line += (
                f", {equal / values:.4%} of values bit-equal, max "
                f"{max(r['max_steps'] for r in chosen)} steps"
            )
            if input_set == "special":
                agree = all(r["nan_positions_agree"] for r in chosen)
                line += f", NaN positions agree: {agree}"
        lines[input_set] = line
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists.")
    manifest = json.loads((args.capture / "manifest.json").read_text())
    report = {"environment": environment(), "capture": manifest["name"]}
    report["kernels"] = {}
    family_inputs = {}
    for kernel_name, record in manifest["kernels"].items():
        print(kernel_name, flush=True)
        kernel_class = KERNELS[kernel_name]
        family = kernel_class.input_family
        if family not in family_inputs:
            family_inputs[family] = list(test_inputs(family, manifest["smoke"]))
        report["kernels"][kernel_name] = {}
        for role, reference in record["configs"].items():
            results = compare_config(
                kernel_class,
                reference,
                args.capture / kernel_name / role,
                family_inputs[family],
            )
            summary = summarize(results)
            print(f"  {role} {reference['config']}")
            for input_set, line in summary.items():
                print(f"    {input_set}: {line}")
            report["kernels"][kernel_name][role] = {
                "config": reference["config"],
                "summary": summary,
                "results": results,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
