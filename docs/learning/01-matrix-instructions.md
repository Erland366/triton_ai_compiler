# Lab 1: MI210's matrix instructions, by hand

This lab is for learning, not results. By the end you will have read and edited real GFX90A assembly, and seen with your own edit why MI210's BF16 matrix instruction cannot reproduce A100's small values while its FP32 one can (findings F5 and F6).

Everything runs through `lab/run.py` on one MI210 in your job. Start each shell with:

```bash
cd /vast/users/qirong.ho/erland/Python_project/triton_ai_compiler
export HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD
```

The kernel is the probe's 32x32x32 matrix product, so every output is a sum of 32 products. `lab/work/` is ignored by git, so edit freely there.

## Part 1: Dump and read the two kernels

```bash
.venv/bin/python lab/run.py dump bf16_dot lab/work/bf16_dot
.venv/bin/python lab/run.py dump fp32_dot lab/work/fp32_dot
```

Open both `kernel.amdgcn` files and find the `v_mfma_...` lines. Answer these by reading the files, and write your answers in `docs/learning/notes-01.md`:

1. Which MFMA instruction does each kernel use, and how many times?
2. The instruction name `v_mfma_f32_32x32x8bf16_1k` means a 32x32 output block with 8 values along K per instruction. Given K=32, does the instruction count you found match?
3. The destination is `a[0:15]`. Those are accumulator registers (AGPRs). A wavefront has 64 lanes and the output block has 32x32 = 1024 values. How many values does each lane hold, and does that match 16 registers?
4. The A and B inputs are `v...` registers (VGPRs). In the FP32 kernel each one is a single register; in the BF16 kernel they come in pairs like `v[0:1]`. Work out from the sizes why. Hint: how many A values does one instruction consume, divided by 64 lanes, and how many bits is that per lane?
5. Every MFMA after the first reads `a[0:15]` as its accumulator. What does the first one read instead, and why?
6. Look for `s_nop` lines near the MFMAs. What do you think they are for? This is the hazard problem: the hardware does not stall for some dependencies, so the code must wait explicitly.

## Part 2: See the flush happen

```bash
.venv/bin/python lab/run.py run bf16_dot product_subnormal
.venv/bin/python lab/run.py run fp32_dot product_subnormal
```

Each output is one product of 2^-70 and 2^-70, which is 2^-140, a subnormal FP32 number. Note what each kernel returns, and the FP32 bit pattern it prints. Then try the other cases (`input_subnormal`, `subnormal_products_sum_to_normal`, `normal_plus_subnormal_product`, `normal_control`) and write down which kernel keeps what.

## Part 3: Your first edit

Copy the FP32 kernel and edit the copy:

```bash
cp lab/work/fp32_dot/kernel.amdgcn lab/work/fp32_dot/edited.amdgcn
```

In `edited.amdgcn`, delete one `v_mfma_f32_32x32x2f32` line from the middle of the chain (not the first one: you answered why in question 5). Before running it, **predict** the output for `subnormal_products_sum_to_normal`, where every product is 2^-130 and the unedited kernel gives 32 x 2^-130 = 2^-125. Then run:

```bash
.venv/bin/python lab/run.py run fp32_dot subnormal_products_sum_to_normal \
    --edited lab/work/fp32_dot/edited.amdgcn
```

`run` checks that Triton really used your file, and fails loudly if it did not. If your prediction was right, you have shown that you know how many products one instruction adds. If it was wrong, work out why before moving on.

## Part 4: Why this matters for the project

Write two or three sentences in your notes answering this: to match A100 on small values, which of the two instructions could an MI210 kernel use, and what does it cost? Hint: compare the instruction counts from question 1.

That answer is the first piece of knowledge the agent will be given. Lab 2, the first probe, asks the harder question: in what order does A100's `mma` add its 16 products, and can MI210 reproduce that order?
