# Findings

Measured results, newest last. Each entry names its evidence so it can be rechecked. Plans and decisions live in [`research-plan.md`](research-plan.md).

## F1: `atomic_add` ReductionSum does not repeat on either vendor (2026-11-08)

With more than two tiles per row, the order in which tiles add into the output changes from launch to launch, and so do the bits. At block sizes 1024 and 32, 19 of 19 repeat launches differed on both A100 and MI210. At block size 2048 (two tiles), 0 of 19 differed, as expected, since one addition gives the same result in either order. A one-tile-per-row version repeated on both vendors at 4, 8 and 16 warps.

Evidence: `amd/check_bf16_kernels.py --repeats 20` on MI210, and `containers/check_a100.sbatch` on A100 (output pasted in the session, not kept as a file).

## F2: On A100, MatMul and GELU bits do not depend on the config (2026-11-09)

In capture `a100-ref-381110`, every MatMul config that ran (153 of 192, for both BF16 and FP32 output) gave the same output hash on the autotune input. The tuned and held-out configs also gave identical bits on all 34 test inputs, adversarial and special inputs included. GELU behaved the same (21 configs, 34 of 34 inputs). ReductionSum is the exception: its three warp counts gave three different hashes, and its tuned and held-out configs differed on 27 of 34 inputs.

Likely reasons, not yet tested: each GELU output depends on one input value, and Triton's MatMul on A100 appears to accumulate along K in the same order whatever the tile sizes. So "A100 Triton's MatMul output" is one target, not one per config.

Consequence: held-out configs cannot test generalization for MatMul or GELU.

Evidence: `captures/a100-ref-381110/manifest.json` (job 381110, `gpu-55`, commit `1cbec4d`); checksums and input regeneration verified on MI210.

## F3: On MI210, the same pattern holds (2026-11-09)

The same sweep on MI210 found the same dependence on config as A100. Every MatMul config that ran (135 of 192, for both outputs; more were rejected than on A100 because MI210 has less shared memory) gave one output hash on the autotune input, and GELU's 21 configs gave one each. On all 34 test inputs, MatMul and GELU gave identical bits under MI210's own tuned config and under A100's tuned and held-out configs. ReductionSum again depends on the warp count: three warp counts gave three hashes, and 8 and 4 warps agreed on only 5 of 34 inputs (7 of 34 on A100).

This compares MI210 with itself only and says nothing yet about MI210 against A100. It does mean that, for MatMul and GELU, comparing one config per vendor covers every config.

Evidence: `capture/config_sweep.py` on one MI210 (`auh7-1b-gpu-296`, job 354563, uncommitted tree at `2ba26f4`), output `logs/mi210_config_sweep.json` (untracked).
