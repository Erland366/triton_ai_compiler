# Research plan: bit-exact kernels from NVIDIA to AMD

Agreed with the user on 2026-11-06. Terms are defined in [`CONTEXT.md`](../CONTEXT.md).

## Research question

On MI210, can a BF16 kernel reproduce the output bits of a pinned A100 Triton kernel over a fixed set of test inputs? If so, what is the portability cost against autotuned native AMD Triton, and how much of it can an LLM agent recover beyond the emulation baseline?

Three outcomes count as answers:

- **Matched, and the agent helps:** the full trade-off story.
- **Matched, but the agent adds nothing:** a deterministic method and a measured cost.
- **Not matched within the stop rule:** a report of which outputs and kernels could and could not be matched.

## Decisions

| Area | Decision |
|---|---|
| Objective | Bit equality is required, and speed is the objective. The paper's story is the trade-off between them |
| Primary metric | Portability cost. For the agent, the latency of its fastest valid candidate against the emulation baseline, and whether it found a valid candidate at all |
| Verdict | Zero tolerance, compared as raw bytes. Signed zero counts, and NaN payloads count unless the reference GPU itself does not repeat them |
| Near-misses | Invalid candidates are logged and plotted with latency, mismatch rate, and difference in representable steps, plus how many fall within 1e-7 and 1e-6 on FP32 outputs |
| Reference | A pinned A100 Triton capture, which must repeat bit for bit on A100 |
| H200 | Replication with the same probes, kept as a separate reference if the bits differ |
| Configs | The A100 autotuned config first, with the kernel default as the first held-out config, or the second-fastest if the default wins. More held-out configs follow. A held-out config must give different output bits from the tuned config on most test inputs. For capture `a100-ref-381110` this is not yet checked: each config repeated on its own, but the two configs' output hashes have not been compared |
| Kernels | New BF16 kernels in this repo: MatMul (BF16 and FP32 output), ReductionSum (FP32 output), GELU (BF16 and FP32 output). FP16 and Float8 wait |
| ReductionSum design | The reference is `ReductionSumBF16PerRowOutputFloat32` (one tile per row, plain store). The `atomic_add` version is kept only as a kernel that cannot be a reference: with more than two tiles per row it did not repeat on either vendor (19 of 19 repeats differed on A100 and MI210 at block sizes 1024 and 32, 0 of 19 at 2048). Checked 2026-11-08 |
| Test inputs | 16 development and 16 sealed, half random and half adversarial, with different values on each side, stored as data |
| Adversarial inputs | Agreed for now and open to revision. All kernels: subnormals, signed zeros, BF16's largest and smallest normals, and values exactly halfway between two BF16 numbers before rounding. MatMul: cancelling large values, constant rows, dot products near a rounding boundary. ReductionSum: cancelling large values, ascending and descending rows. GELU: values near 0, around ±3 to ±10, and near the output's subnormal-to-normal switch. NaN and infinity go in a separate set |
| Autotuning the reference | Reject a config only if it fails to run. Pick the lowest median latency among the rest, and record every config's latency and output hash. No tolerance filter |
| Candidate freedom | Any tile order or warp layout |
| Portability cost denominator | Autotuned native AMD Triton as the headline, with the reference's forced config as a secondary number |
| Baseline | A deterministic emulation baseline, then the agent on top. The user writes the first probe, and we pair on the rest |
| Code level | LLVM IR with inline assembly, and whole-kernel GFX90A assembly only if a measured bottleneck requires it. A separate learning lab covers whole-kernel assembly and stays out of the results |
| Agent | See [ADR 0002](adr/0002-restricted-claude-code-agent.md) |
| Container | The agent runs inside it, with no `$HOME` or `/vast`, one bound run directory, API-only network if possible, a dedicated token, per-candidate timeouts, and a whole exclusive node. A breakout test must fail before any agent runs |
| Agent budget | 12 active hours per kernel, excluding rate-limit pauses but logging both, with 3 runs per kernel and one pinned model and Claude Code version. The 12 hours is an estimate to correct after the pilot |
| Stop rule | Two weeks of work after the first accepted A100 capture, then preserve the result and let the user decide |
| Code location | This repo. Existing `ptx_gym` kernel files stay untouched |

The input counts, the 12 hours, and the two weeks are judgment calls with no measurement behind them yet.

## Order of work

The user approves steps 3, 4, and 7 before they run.

1. Establish code injection on AMD and pin a ROCm environment on MI210. Done on 2026-11-06 with no Triton patch: `amd/check_injection.py` injects at the `llir` and `amdgcn` stages through `TRITON_KERNEL_OVERRIDE`, on one small kernel. Candidate runs must set `TRITON_ALWAYS_COMPILE=1`, because the default cache returned a stale kernel after a file was edited.
2. Write the BF16 kernels as standalone files in `bf16_kernels/`, using stock Triton 3.6.0 on A100 and `triton-rocm` 3.6.0 on MI210. PTX Gym's patched Triton is not used. Written on 2026-11-06 and sanity-checked on MI210 only with `amd/check_bf16_kernels.py`, not yet on A100.
3. Capture test inputs and references on A100 through a Slurm job, then repeat on H200 in its container. Move captures through the Mac with `sha256sum` checks.
4. Measure native AMD against the references.
5. Do the learning lab, then the emulation baseline. The stop rule applies here.
6. Build the agent container and try to break out of it.
7. Run the agents.

## Parked

- **CPU emulator:** the user wants time to think about it. It is most likely to return if the stop rule triggers without a match.
- **Comparison with the Triton AI Compiler paper:** no longer a priority.
- **Agents without bit diffs:** a possible later experiment on whether the diffs help.
- **Natural NVIDIA kernels that do not repeat** (raised by the user on 2026-11-08): making the reference deterministic, as with the per-row ReductionSum, gives an output that real NVIDIA code does not produce. Matching one fixed output is undefined for a kernel like `atomic_add`, so this needs its own experiment design. One starting idea: call an AMD output valid if it equals one of the outputs NVIDIA can produce, such as one order of the atomic additions.

## Evidence this plan rests on

The user's ICLR work (A100 vs MI210, BF16, one workload with FP32 output) found 15,211 of 15,552 outputs differing for a shared ordered-tile Triton kernel and 0 of 15,552 for a scalar sequential one, and names accumulation inside the matrix instructions as the cause. Those numbers were read from `crossmismatchbench/docs/prior_work/iclr2027_source/` by a subagent and have not been rechecked here. They do not predict the BF16-output kernels, where rounding may hide differences. The A100 raw tensors from that work are gone, and no test yet shows that `mma` output can be reproduced with AMD operations.
