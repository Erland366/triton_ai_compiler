"""Regenerate a capture's test inputs on this machine and compare their SHA-256.

    python -m capture.check_inputs captures/a100-ref-1

Inputs are not copied between machines, so this must pass before any output of the
capture is compared with this machine's outputs. Exits nonzero on any mismatch.
"""

import argparse
import json
import sys
from pathlib import Path

from bf16_kernels import KERNELS
from capture.inputs import test_inputs
from capture.run import as_tuple, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    expected = {}
    for kernel_name, record in manifest["kernels"].items():
        family = KERNELS[kernel_name].input_family
        for config in record["configs"].values():
            for output in config["outputs"]:
                key = (family, output["name"])
                if (
                    expected.setdefault(key, output["input_sha256"])
                    != output["input_sha256"]
                ):
                    raise ValueError(f"The capture disagrees with itself on {key}.")
    mismatches = 0
    for family in sorted({family for family, _ in expected}):
        for name, _, inputs in test_inputs(family, manifest["smoke"]):
            want = expected.get((family, name))
            if want is None:
                continue
            if sha256(*as_tuple(inputs)) != want:
                print(f"MISMATCH {family} {name}")
                mismatches += 1
    print(f"checked {len(expected)} inputs, {mismatches} mismatched")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
