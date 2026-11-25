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

So A100 keeps subnormal products and sums, and MI210's matrix instructions flush them whatever the denormal mode. AMD documents this: "On the MI200, MFMA_F16, MFMA_BF16, and MFMA_BF16_1K flush subnormal input/output data to zero" (MIOpen, `docs/conceptual/MI200-alt-implementation.rst`, read 2026-11-10; the same text appears in ROCm docs since 5.0.2). Our kernel uses `v_mfma_f32_32x32x8bf16_1k`. The documentation says inputs and outputs; that intermediate products are flushed too is inferred from our outputs. PyTorch's numerical accuracy notes reportedly add that FP32 and FP64 MFMA do not flush and that no alternate BF16 path exists (seen only in a search summary, not read directly). No NVIDIA documentation on `mma` subnormal handling was found; A100's behavior here rests on our measurement. Ordinary FP32 additions on MI210 keep subnormals (ReductionSum's subnormal rows matched A100 exactly, F4), so matching A100 for small values needs a path other than the native matrix instructions.

Evidence: `amd/denorm_probe.py` and a float64 check of `development_15_tiny`, both on `auh7-1b-gpu-296` GPU 0, against `captures/a100-ref-381110`.

## F6: Direct test: A100's BF16 tensor cores keep subnormals, MI210's BF16 MFMA flushes them (2026-11-12)

`probes/subnormal.py` builds 32x32x32 products whose outputs have two exactly known values, one if subnormals are kept and one if they are flushed: a subnormal product, a subnormal BF16 input, 32 subnormal products summing to a normal value, and a normal value plus a subnormal product, with an all-normal control. Every output of every case matched one of the two values exactly.

| Path | A100 | MI210 |
|---|---|---|
| `tl.dot`, BF16 inputs | kept, all 4 cases (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`) | flushed, all 4 cases (`v_mfma_f32_32x32x8bf16_1k`) |
| `tl.dot`, FP32 inputs, IEEE | kept (`fma.rn.f32`, no tensor cores) | kept (`v_mfma_f32_32x32x2f32`) |
| loop of FP32 multiply-adds | kept (`fma.rn.f32`) | kept (`v_pk_fma_f32`) |

This confirms F5 directly on both vendors, and confirms on MI210 that the FP32 matrix instruction keeps subnormals. BF16 values are exact in FP32, so an MI210 kernel can feed BF16 inputs to `v_mfma_f32_32x32x2f32` and keep subnormals; whether its summation order matches A100's is untested.

Evidence: A100 job 388887 (`gpu-49`, GPU 0, commit `13e59c4`, image `fa21051f…`), log `logs/probe_a100_388887.out`; MI210 on `auh7-1b-gpu-296` (job 354563, GPU 0, commit `13e59c4`), `logs/subnormal_mi210_354563.json`. Both untracked.

## F7: At K=1024 the pattern is the same, with less disagreement (2026-11-23)

Capture `a100-ref-388901` holds both MatMul kernels at K=1024 (M=N=4096). On A100 the config again does not change the bits: all 153 configs that ran gave one output hash, and tuned and held-out configs were identical on 34 of 34 inputs. Native MI210 against it, with K=4096 (F4) for comparison, on the random development inputs:

| Kernel | K=4096 bit-equal | K=1024 bit-equal | K=1024 within 1 step |
|---|---|---|---|
| MatMul, BF16 output | 99.744% | 99.941% | 99.996% |
| MatMul, FP32 output | 0.357% | 2.177% | 5.451% |

The halfway cases still match exactly, and the `tiny` case still differs almost everywhere (0.44% with BF16 output, 0% with FP32), as subnormal flushing (F5, F6) predicts. A shorter sum gives fewer chances for the two accumulation orders to round differently, which fits the higher agreement.

Evidence: capture job 388901 (`gpu-53`, GPU 2, commit `13e59c4`), checksums and input regeneration verified on MI210; `capture/compare.py` at commit `17f3a8d` on `auh7-1b-gpu-296` (job 354563, GPU 0), output `logs/compare_mi210_vs_a100-ref-388901.json` (untracked).

## F8: First agent pilot, one active hour on MatMul with FP32 output (2026-11-24)

The correctness-first agent (`taic_amd/agent.py`, Claude Opus 5.5 at effort high, in the agent container) ran for one active hour from native AMD Triton's LLVM IR. It found no exact candidate. Its best, candidate 21 of 22, doubled the share of visible values that match A100:

| Input | Native | Best candidate |
|---|---|---|
| Random (8 inputs) | 0.36% | 6.54% (17.9% within 1 step) |
| Cancelling large values | 0% | 99.57% |
| Zeros and smallest normals | 5.36% | 33.95% |
| Tiny values (subnormal products) | 0% | 6.50% |
| Halfway cases | 100% | 100% |
| Special: NaN, infinity | 0%, 3.31% | 0%, 0.002% |
| All visible inputs | 11.94% | 23.50% |

Unprompted, the agent replaced the kernel with its own model of A100's tensor cores: each group of 16 products is aligned to the largest exponent among them and the running sum, converted to integers, and added in fixed point. The near-exact cancelling case is some evidence for that model, since cancellation magnifies differences in how sums are formed; this is an inference, not a confirmed finding.

Of the hour, 75% went to verification (22 runs, about 2 minutes each, since the agent's kernel is slow scalar code), 2 candidates crashed on syntax errors in patches, and no usage limit was hit. Its reasoning was not logged, because Opus 5.5 omits thinking by default.

This is a plumbing check and one run of one model, so it says nothing yet about how well the method works.

Evidence: `/vast/users/qirong.ho/erland/taic_runs/pilot-1h/20261124T131053Z_MatrixMultiplicationBF16OutputFloat32/` (untracked), code at commit `50997ef`, against `captures/a100-ref-381110`.
