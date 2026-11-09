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

## F4: First comparison of native MI210 Triton with the A100 reference (2026-11-09)

Native Triton 3.6.0 on one MI210, run at each A100 config on the regenerated test inputs and compared bit for bit with capture `a100-ref-381110`. Tuned and held-out configs gave the same results, except ReductionSum, whose warp count changes its bits on both vendors.

| Kernel | Random inputs, values bit-equal | Whole outputs identical (dev + sealed) | Notes |
|---|---|---|---|
| GELU, BF16 output | 100% | 32 of 32, all special inputs too | Already matches, including every finite BF16 input value |
| GELU, FP32 output | 86.97% | 6 of 32 | Differences of up to 3,776 steps, likely from the `erf` implementations; the BF16 rounding hides them all |
| MatMul, BF16 output | 99.74% | 4 of 32 (the halfway cases) | |
| MatMul, FP32 output | 0.36% | 4 of 32 (the halfway cases) | Nearly every value differs, consistent with a different accumulation inside the matrix instructions |
| ReductionSum per row | 72.80% (8 warps) | 4 of 32 (subnormal and constant rows) | Sorted rows only 65 to 67% equal |

Patterns worth following up, all hypotheses:

- **Halfway MatMul cases match exactly on both outputs.** Their sums are exact in FP32, so this suggests both vendors round FP32 to BF16 the same way, and the MatMul differences come from accumulation.
- **The `tiny` MatMul case differs almost everywhere** (0.43% bit-equal with BF16 output, 0% with FP32), while ReductionSum's subnormal rows match exactly. Products land near the bottom of FP32's normal range, so the matrix instructions may handle subnormals differently between vendors, while ordinary additions do not.
- **NaN payloads differ in FP32 outputs.** On the special inputs, NaN positions agree everywhere, but MatMul and ReductionSum FP32 NaNs have different bits. By the agreed definition these count, since A100 repeats them.
- **Cancelling inputs differ everywhere**, as expected, since cancellation magnifies any difference in order.

Step counts mislead near zero, where a small absolute difference spans many representable values.

Evidence: `capture/compare.py` at commit `93bc84e` on `auh7-1b-gpu-296` (job 354563, GPU 0), output `logs/compare_mi210_vs_a100-ref-381110.json` (untracked).

## F5: MI210's matrix instructions flush subnormals, A100's keep them (2026-11-09)

Setting Triton's AMD option `allow_flush_denorm` changed the compiled kernel's FP32 denormal mode (`.amdhsa_float_denorm_mode_32` from 3 to 0) but changed no output bit of either MatMul kernel, on the random, subnormal or tiny inputs. On the tiny case (inputs scaled by 2^-63, so 79% of products fall below 2^-126), with FP32 output and the first 256 rows measured against float64 sums of the same BF16 inputs:

| | Zero outputs | Subnormal outputs | Median relative error |
|---|---|---|---|
| float64 sums | 0% | 1.24% | none (the yardstick) |
| A100 | 0% | 1.24% | 4.7e-6 |
| MI210 | 1.19% | 0% | 0.38 |

So A100 keeps subnormal products and sums, and MI210's matrix instructions flush them whatever the denormal mode. This is read from outputs, not confirmed in AMD's documentation. Ordinary FP32 additions on MI210 keep subnormals (ReductionSum's subnormal rows matched A100 exactly, F4), so matching A100 for small values needs a path other than the native matrix instructions.

Evidence: `amd/denorm_probe.py` and a float64 check of `development_15_tiny`, both on `auh7-1b-gpu-296` GPU 0, against `captures/a100-ref-381110`.
