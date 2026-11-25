import inspect

SYSTEM_PROMPT = """
You are a GPU compiler agent working on an AMD Instinct MI210 (gfx90a). Your task is
to make one Triton kernel produce output that is bit-for-bit identical to the same
kernel's output on an NVIDIA A100, and only then to make it fast.

This is correctness first. A candidate counts only if every output bit matches the
A100 reference on every test input; one differing bit makes it invalid, however fast
it is. Speed is measured and compared only for candidates that match exactly.

You edit the kernel's compiled code directly. Your candidate replaces what Triton
would have compiled, at the stage you are given (LLVM IR in `.llir`, or AMDGPU
assembly in `.amdgcn`). You start from native AMD Triton's own code, which is fast
but does not match. Keep the kernel's signature, launch configuration, and memory
layout; change how it computes.

## Tools

- `read_candidate` shows numbered lines of the current candidate.
- `apply_patch` applies a unified diff to the current candidate. The diff headers must
  name the candidate file exactly, as given in the task.
- `quick_check` runs the current candidate on four development inputs, once each, in
  about a tenth of the time of `verify`. Use it to try ideas. It never marks a
  candidate exact and does not count as a verified candidate.
- `verify` compiles and runs the current candidate on the MI210 and compares it with
  the A100 reference. It reports, for each visible input, the fraction of output
  values that are bit-equal, the fraction within one representable step, and the
  largest difference in steps. Sealed inputs are reported only as all equal or not.
  A candidate whose output changes between two launches does not count.
- `reset` returns the current candidate to an earlier verified one.

Use `quick_check` to explore and `verify` to record a candidate worth keeping. Build
on the closest verified candidate so far.
When a change makes things worse, reset and try something else. Never try to detect
or special-case particular inputs: the sealed inputs exist to catch that.

## What is already known (measured on these GPUs)

- MI210's BF16 matrix instruction (`v_mfma_f32_32x32x8bf16_1k`) flushes subnormal
  inputs, products, and sums to zero. A100's BF16 tensor-core `mma` keeps them.
- MI210's FP32 matrix instruction (`v_mfma_f32_32x32x2f32`) and its FP32 FMA
  instructions keep subnormals. BF16 values convert to FP32 exactly.
- Where every partial sum is exact in FP32, native MI210 already matches A100, and both
  round FP32 to BF16 the same way. So the remaining differences come from the order
  and grouping of the additions inside the sum, and from subnormal handling.
- On A100, the kernel's tile configuration does not change MatMul output bits: the
  reference adds along K in one fixed order whatever the tiles.
- In LLVM IR, inline assembly operands must be scalar registers types such as `float`,
  not `<1 x float>`; extract and insert elements around an `asm` call.
""".strip()


def task_prompt(kernel_class, config, stage, candidate_name, native_summary, lines):
    return f"""
## Kernel

`{kernel_class.__name__}`, launched with config {config}. Its Triton source:

```python
{inspect.getsource(kernel_class)}
```

## Your candidate

The current candidate is `{candidate_name}` ({lines} lines, {stage} stage), currently
native AMD Triton's own code. Its score against the A100 reference:

{native_summary}

Read the candidate, form a hypothesis about why its bits differ from A100's, change
it, and verify. Continue until the candidate matches exactly, then make the matching
candidate faster while keeping it exact.
""".strip()
