# Cross-Vendor Bit-Exact Kernels

The research on making a kernel run on an AMD GPU produce the same output bits as a reference kernel on an NVIDIA GPU, as fast as possible. It extends the Triton AI Compiler, which targets NVIDIA only and accepts outputs within a tolerance.

## Language

**Reference capture**:
The stored output of a pinned Triton kernel on a named NVIDIA GPU, for fixed test inputs, with its Triton version, PTX, launch configuration, and tuned parameters recorded.
_Avoid_: Gold standard, NVIDIA output, ground truth

**Reference GPU**:
The NVIDIA GPU that a reference capture was taken on. A100 is the primary one; "NVIDIA" alone does not name a reference.
_Avoid_: NVIDIA, the source device

**Replication capture**:
A reference capture repeated on another NVIDIA GPU, such as H200, to test whether the reference GPU's bits hold across NVIDIA hardware. If the bits differ, it stands as a separate reference and does not change the primary one.
_Avoid_: NVIDIA consensus, second reference

**Tuned config**:
The recorded launch and tuning parameters that a Triton kernel ran with for a reference capture. Tile order can change the output bits, so each capture belongs to exactly one tuned config.
_Avoid_: Autotuner output, best config

**Held-out config**:
A tuned config with a reference capture that was not used to develop a candidate or method, and is used to test whether the method generalizes.
_Avoid_: New config, any config

**Test inputs**:
The fixed random and adversarial inputs over which equality is claimed, stored as data because each vendor's random generator produces different values. They divide into development inputs and sealed inputs.
_Avoid_: Input suite, test data, random inputs

**Development inputs**:
The test inputs where the agent may see full bit diffs.
_Avoid_: Training data, visible inputs

**Sealed inputs**:
The test inputs where the agent sees only pass or fail, used to catch candidates fitted to the development inputs.
_Avoid_: Test set, hidden inputs

**Empirical bit equality**:
Every bit of every final output tensor matches the reference capture over the test inputs, with signed zero counted and NaN payloads counted unless the reference GPU itself does not repeat them. It makes no claim about any other inputs.
_Avoid_: Exact, bitwise correct, proven equal, matches

**Valid candidate**:
An AMD kernel that achieves empirical bit equality on both development and sealed inputs, using any tile order or warp layout. A candidate differing in one bit is invalid, however fast it is.
_Avoid_: Correct kernel, passing kernel

**Portability cost**:
The latency of the fastest known valid candidate relative to native AMD Triton autotuned on the same AMD GPU.
_Avoid_: Overhead, slowdown

**Emulation baseline**:
A valid candidate produced deterministically, without an LLM, which the agent must beat on latency.
_Avoid_: Fallback, scalar version

**Native AMD Triton**:
Triton's own compiled kernel on the AMD GPU, which is fast but not expected to be a valid candidate.
_Avoid_: Baseline, AMD reference
