"""Sanity-check the BF16 kernels on one GPU, at several configs per kernel.

For each config: is the output finite, how close is it to Torch, and do the bits
repeat across launches on one input? This is not a cross-vendor measurement.
Exits nonzero if a config fails to run or its output is not finite. A config that
exceeds hardware limits is reported as rejected, as an autotuner would treat it.
"""

import argparse
import sys

import torch
import triton
from triton.runtime.errors import OutOfResources

from bf16_kernels import KERNELS

SEED = 0
SMALL_MATMUL = {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 4}
LARGE_MATMUL = {"block_m": 256, "block_n": 256, "block_k": 256, "num_warps": 16}
SMALL_BLOCK = {"block_size": 32, "num_warps": 4}
LARGE_BLOCK = {"block_size": 2048, "num_warps": 16}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def extra_configs(kernel_class):
    if "block_m" in kernel_class.tuning_options:
        return [SMALL_MATMUL, LARGE_MATMUL]
    if "block_size" not in kernel_class.tuning_options:
        return [{"num_warps": 8}, {"num_warps": 16}]
    return [SMALL_BLOCK, LARGE_BLOCK]


def to_device(inputs):
    if isinstance(inputs, tuple):
        return tuple(tensor.cuda() for tensor in inputs)
    return inputs.cuda()


def raw_bits(tensor):
    integer_type = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.view(integer_type).clone()


def check_config(kernel_class, config, inputs, reference, repeats):
    kernel = kernel_class(config)
    first = kernel.forward(inputs)
    torch.cuda.synchronize()
    bits = raw_bits(first)
    repeats_differing = 0
    for _ in range(repeats - 1):
        output = kernel.forward(inputs)
        torch.cuda.synchronize()
        repeats_differing += int(not torch.equal(bits, raw_bits(output)))

    finite = bool(first.float().isfinite().all())
    difference = (first.float() - reference.float()).abs()
    match_torch = (bits == raw_bits(reference)).float().mean().item()
    print(
        f"  {config}: finite {finite}, max |diff| vs Torch "
        f"{difference.max().item():.3e}, bit-equal to Torch {match_torch:.4%}, "
        f"repeats differing {repeats_differing}/{repeats - 1}"
    )
    return finite


def main():
    args = parse_args()
    print(
        f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        f"triton {triton.__version__}"
    )
    passed = True
    for name, kernel_class in KERNELS.items():
        print(name)
        generator = torch.Generator().manual_seed(SEED)
        inputs = to_device(kernel_class().random_input(generator))
        reference = kernel_class().forward_torch(inputs)
        configs = [kernel_class.default_config, *extra_configs(kernel_class)]
        for config in configs:
            try:
                passed &= check_config(
                    kernel_class, config, inputs, reference, args.repeats
                )
            except OutOfResources as exc:
                print(f"  {config}: rejected, {exc}")
            except (RuntimeError, ValueError) as exc:
                print(f"  {config}: FAILED {type(exc).__name__}: {exc}")
                passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
