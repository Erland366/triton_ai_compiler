# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Follow `AGENTS.md` for code style, tooling, and verification rules. `README.md` documents every agent flag and environment variable. This file covers only what you need from several files at once.

## Environment

- `ptx_gym` and `skills/ptx_skills` are submodules holding the evaluator, the kernels, and the PTX reference. They may be uninitialized, in which case the directories are empty. `import ptx_gym` then still succeeds as an empty namespace package and fails later on attribute access, so check `ls ptx_gym` first. To populate them, run `git submodule update --init --recursive`, which needs GitHub access. If the proxy blocks HTTPS, add `-c url."git@github.com:".insteadOf="https://github.com/"` after `git`, from a shell where your GitHub key works.
- Everything here assumes CUDA: `run_kernel.py` and `run_identity_compare.py` raise without `torch.cuda.is_available()`. The AMD MI210 cluster this checkout was found on (2026-11-05) shows no NVIDIA GPUs in `sinfo`, so these commands cannot run on it as written.
- Use `.venv/bin/python`, per `AGENTS.md`. The README's `.triton_ai_compiler` venv name is the upstream install path. The patched Triton ships inside `ptx_gym`, and `pyproject.toml` blocks stock Triton from replacing it.
- The AMD research in `docs/research-plan.md` runs on MI210 from `amd/requirements-rocm.txt` (`torch 2.10.0+rocm7.0`, `triton-rocm 3.6.0`). That environment does not include `ptx_gym`, so the NVIDIA commands below need their own CUDA environment.
- NVIDIA runs for that research use the Singularity image from `containers/triton-cuda.def` (`torch 2.10.0+cu126`, `triton 3.6.0`). On the A100 cluster, submit `containers/check_a100.sbatch` from a checkout under `/nfs-stor/qirong.ho/erland`, which has no Apptainer or usable Podman store for this project.
- To inject code on AMD, use `TRITON_KERNEL_OVERRIDE`, as `amd/check_injection.py` shows. `ir_override` is NVIDIA-only. Always set `TRITON_ALWAYS_COMPILE=1` for candidates: the cache key ignores override file contents, so an edited candidate can silently run the old kernel.
- Model-generated PTX is untrusted low-level code. Run the agent only on isolated, non-production machines.

## Commands

```bash
# Lint (submodule and skill directories excluded)
.venv/bin/python -m ruff check --exclude ptx_gym --exclude skills .

# Syntax check
.venv/bin/python -m compileall agent.py utils prompts

# Smoke test without an LLM: Triton vs Torch, then benchmark one kernel
.venv/bin/python run_kernel.py SoftmaxFloat16Kernel

# Check that Triton's own PTX, launched as a candidate, reproduces Triton
.venv/bin/python run_identity_compare.py SoftmaxFloat16Kernel

# Run the agent on one kernel
.venv/bin/python -m agent MatrixMultiplicationFloat16 --provider anthropic --max-budget 5

# Re-run compile, correctness, and timing on an archived candidate
.venv/bin/python -m remeasure_candidate \
  compiler_results/sm_100a/Convolution2DFloat8Kernel/kernel_1.result.json \
  --kernel Convolution2DFloat8Kernel --output output/sm_100a/out.json

# Dump TTIR, TTGIR, LLVM IR, and PTX for every kernel
.venv/bin/python -m extract_ptx
```

There is no test suite, and `AGENTS.md` forbids adding tests or a test framework unless asked. The README lists `remeasure_baseline.py`, but that file is not in this repository.

## Architecture

The repository is a client for PTX Gym, split on purpose: **PTX Gym** (`ptx_gym`) owns the kernels and the verdict, and **this client** owns prompting, providers, and search. On NVIDIA, candidate correctness and speedup are decided only by `TritonPTXCandidateEvaluator` in PTX Gym, so do not reimplement or weaken those checks here. The planned AMD work is different: its bit-equality verdict will live in this repo, alongside new BF16 kernels, with `ptx_gym` otherwise used for its base class and Triton source. See `CONTEXT.md` for its terms.

`agent.py:run_agent_loop` drives one run for one kernel:

1. Build the evaluator with `utils/response.py:verifier_for_kernel`. This autotunes the Triton baseline unless `--start-json` supplies saved metrics.
2. Dump Triton's PTX, which is the starting point and the reference to beat.
3. Create a provider session (`utils/providers.py`) and build prompts from the kernel source (`prompts/initial.py`, `prompts/blocks.py`).
4. Generate a candidate. Tool calls such as `launch_verifier` go through `utils/response.py:_run_ptx_tool` to `verifier.evaluate(payload)`. `utils/response.py` holds one tool-calling loop per provider API.
5. Repair failures up to `--max-repair-attempts`.
6. Improve: plan one to three changes from the NCU report, or from source when there is no report (`prompts/improvement.py`, `utils/ncu.py`), apply them as small patches, and evaluate each one.

Every candidate and its cost go to a timestamped trace directory (`utils/traces.py`, `utils/cost.py`).

## Things that are easy to break

- **Kernel files are benchmark inputs.** The prompt embeds kernel source verbatim through `inspect.getsource`, so any edit under `ptx_gym/ptx_gym/kernels/`, even formatting, changes the prompt and invalidates comparisons with earlier runs. Never format, deduplicate, or share code across kernel files. `AGENTS.md` gives the full rule.
- **Correctness has different references in different scripts.** `run_kernel.py` compares Triton with Torch at the kernel's `verification_tolerance`, defaulting to `1e-3`. Candidates are compared with Triton using `torch.allclose` at that same tolerance, as both `rtol` and `atol` (`ptx_gym/ptx_gym/evaluation/evaluate.py:149`). Neither is a bitwise check, so name the reference and tolerance when you report a result.
- **`compiler_results/<target>/<Kernel>/` holds archived results** by GPU architecture: `sm_89` (L40S), `sm_90a` (H100), `sm_100a` (B200). Each `*.result.json` stores the PTX, launch dimensions, `autotune_metrics`, `speedup`, `p50`, and `gpu_type`. Speedups are valid only on the matching GPU.
- **Benchmarks must never run in parallel**, because competing workloads invalidate the timings.
