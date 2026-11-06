# Require bit equality against a stored A100 capture

An AMD kernel counts as valid only if its output bytes equal a stored capture from a pinned Triton kernel on A100, with no tolerance. The existing PTX Gym check compares a candidate with Triton on the same GPU through `torch.allclose` at 1e-3 or looser, which cannot show that two vendors agree, and a live autotuned reference has no fixed target because timing can change the tuned config and with it the bits.

## Considered Options

- **A tolerance of 1e-6 or 1e-7:** rejected as the verdict. Near 1.0 that is about one to eight representable steps in FP32 and has no useful meaning in BF16, where one step is about 0.0078. These thresholds are kept as near-miss reporting on FP32 outputs.
- **Same-device Triton on AMD:** rejected because it says nothing about NVIDIA.
- **H200 as the reference:** H200 is newer, but the existing bit-level evidence is on A100. H200 is a replication, and a separate reference if its bits differ.

## Consequences

Test inputs must be stored as data, because each vendor's random generator produces different values. The AMD verdict lives in this repo rather than in PTX Gym. A pass makes no claim about inputs outside the test inputs.
