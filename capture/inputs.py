"""Test inputs for the BF16 kernels, generated on the CPU from fixed seeds.

Each family has 16 development and 16 sealed inputs (8 random and 8 adversarial each,
with different seeds on each side) and 2 special inputs holding NaN and infinity. The
cases follow docs/research-plan.md. Inputs are regenerated rather than copied between
machines, so each capture records their SHA-256 for the other side to check.
"""

import zlib

import torch

BF16_MIN_SUBNORMAL = 2.0**-133
BF16_MIN_NORMAL = 2.0**-126
MATMUL_SHAPE = (4096, 4096)
ROWS_SHAPE = (256, 4096)
RANDOM_PER_SET = 8
NAN_BITS = (0x7FC0, 0x7FC1, 0x7F81, 0xFFC0, 0xFF81)
INF_BITS = (0x7F80, 0xFF80)


def generator(*parts):
    seed = zlib.crc32("/".join(str(part) for part in parts).encode())
    return torch.Generator().manual_seed(seed)


def randn(shape, gen):
    return torch.randn(shape, generator=gen)


def uniform(shape, gen, low, high):
    return low + (high - low) * torch.rand(shape, generator=gen)


def signs(shape, gen):
    return torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)


def mix(base, replacement, fraction, gen):
    return torch.where(
        torch.rand(base.shape, generator=gen) < fraction, replacement, base
    )


def subnormals(shape, gen):
    steps = torch.randint(1, 128, shape, generator=gen).float()
    return signs(shape, gen) * steps * BF16_MIN_SUBNORMAL


def zeros_and_min_normals(shape, gen):
    choice = torch.randint(0, 6, shape, generator=gen)
    values = torch.tensor([0.0, -0.0, BF16_MIN_NORMAL, -BF16_MIN_NORMAL])
    return torch.where(choice < 4, values[choice.clamp(max=3)], randn(shape, gen))


def log_uniform(shape, gen, low_exponent, high_exponent):
    exponents = uniform(shape, gen, low_exponent, high_exponent)
    return signs(shape, gen) * torch.exp2(exponents)


def all_finite_bf16(shape, gen):
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    count = shape[0] * shape[1]
    tiled = values.repeat(count // values.numel() + 1)
    return tiled[torch.randperm(tiled.numel(), generator=gen)[:count]].view(shape)


def inject_specials(tensor, gen, bit_patterns, fraction):
    bits = tensor.to(torch.bfloat16).view(torch.int16).clone()
    patterns = torch.tensor(
        [
            pattern - 0x10000 if pattern >= 0x8000 else pattern
            for pattern in bit_patterns
        ],
        dtype=torch.int16,
    )
    mask = torch.rand(bits.shape, generator=gen) < fraction
    choice = torch.randint(0, len(bit_patterns), bits.shape, generator=gen)
    return torch.where(mask, patterns[choice], bits).view(torch.bfloat16)


def matmul_random(gen):
    return randn(MATMUL_SHAPE, gen), randn(MATMUL_SHAPE, gen)


def matmul_subnormals(gen):
    a = mix(randn(MATMUL_SHAPE, gen), subnormals(MATMUL_SHAPE, gen), 0.5, gen)
    return a, randn(MATMUL_SHAPE, gen)


def matmul_zeros_and_min_normals(gen):
    return zeros_and_min_normals(MATMUL_SHAPE, gen), zeros_and_min_normals(
        MATMUL_SHAPE, gen
    )


def matmul_cancelling(gen):
    a, b = randn(MATMUL_SHAPE, gen), randn(MATMUL_SHAPE, gen)
    b[1::2] = b[0::2]
    half = (MATMUL_SHAPE[0], MATMUL_SHAPE[1] // 2)
    large = (2.0**40 * uniform(half, gen, 1.0, 2.0)).to(torch.bfloat16).float()
    mask = torch.rand(half, generator=gen) < 0.25
    a[:, 0::2] = torch.where(mask, large, a[:, 0::2])
    a[:, 1::2] = torch.where(mask, -large, a[:, 1::2])
    return a, b


def matmul_constant_rows(gen):
    a = randn((MATMUL_SHAPE[0], 1), gen).expand(MATMUL_SHAPE).contiguous()
    return a, randn(MATMUL_SHAPE, gen)


def sparse_offsets(gen, density, magnitudes):
    choice = torch.randint(0, len(magnitudes), MATMUL_SHAPE, generator=gen)
    values = signs(MATMUL_SHAPE, gen) * torch.tensor(magnitudes)[choice]
    mask = torch.rand(MATMUL_SHAPE, generator=gen) < density
    return torch.where(mask, values, torch.zeros(MATMUL_SHAPE))


def matmul_halfway(gen):
    magnitudes = [1.0, 2.0**-6, 2.0**-7, 2.0**-8]
    a = sparse_offsets(gen, 4 / MATMUL_SHAPE[1], magnitudes)
    return a, signs(MATMUL_SHAPE, gen)


def matmul_halfway_scaled(gen):
    a, b = matmul_halfway(gen)
    exponents = torch.randint(-60, 61, (MATMUL_SHAPE[0], 1), generator=gen).float()
    return a * torch.exp2(exponents), b


def matmul_large(gen):
    return randn(MATMUL_SHAPE, gen) * 2.0**56, randn(MATMUL_SHAPE, gen) * 2.0**56


def matmul_tiny(gen):
    return randn(MATMUL_SHAPE, gen) * 2.0**-63, randn(MATMUL_SHAPE, gen) * 2.0**-63


def matmul_special_nan(gen):
    a = inject_specials(randn(MATMUL_SHAPE, gen), gen, NAN_BITS + INF_BITS, 0.01)
    return a, randn(MATMUL_SHAPE, gen)


def matmul_special_inf(gen):
    a = inject_specials(randn(MATMUL_SHAPE, gen), gen, INF_BITS, 0.001)
    b = inject_specials(randn(MATMUL_SHAPE, gen), gen, INF_BITS, 0.001)
    return a, b


def rows_random(gen):
    return randn(ROWS_SHAPE, gen)


def rows_zeros_and_min_normals(gen):
    return zeros_and_min_normals(ROWS_SHAPE, gen)


def rows_special_nan(gen):
    return inject_specials(randn(ROWS_SHAPE, gen), gen, NAN_BITS + INF_BITS, 0.01)


def rows_special_inf(gen):
    return inject_specials(randn(ROWS_SHAPE, gen), gen, INF_BITS, 0.001)


def reduction_subnormals(gen):
    tiny = randn(ROWS_SHAPE, gen) * 2.0**-120
    return mix(tiny, subnormals(ROWS_SHAPE, gen), 0.5, gen)


def reduction_cancelling(gen):
    x = randn(ROWS_SHAPE, gen)
    half = (ROWS_SHAPE[0], ROWS_SHAPE[1] // 2)
    large = (2.0**100 * uniform(half, gen, 1.0, 2.0)).to(torch.bfloat16).float()
    mask = torch.rand(half, generator=gen) < 1 / 64
    x[:, 0::2] = torch.where(mask, large, x[:, 0::2])
    x[:, 1::2] = torch.where(mask, -large, x[:, 1::2])
    return x


def reduction_ascending(gen):
    return randn(ROWS_SHAPE, gen).sort(dim=1).values


def reduction_descending(gen):
    return randn(ROWS_SHAPE, gen).sort(dim=1, descending=True).values


def reduction_constant_rows(gen):
    return randn((ROWS_SHAPE[0], 1), gen).expand(ROWS_SHAPE).contiguous()


def reduction_wide_range(gen):
    return log_uniform(ROWS_SHAPE, gen, -60, 60)


def reduction_large(gen):
    return randn(ROWS_SHAPE, gen) * 2.0**100


def gelu_near_zero(gen):
    return mix(
        log_uniform(ROWS_SHAPE, gen, -40, -5), subnormals(ROWS_SHAPE, gen), 0.25, gen
    )


def gelu_subnormals(gen):
    tiny = randn(ROWS_SHAPE, gen) * 2.0**-120
    return mix(tiny, subnormals(ROWS_SHAPE, gen), 0.75, gen)


def gelu_saturation(gen):
    return signs(ROWS_SHAPE, gen) * uniform(ROWS_SHAPE, gen, 3.0, 10.0)


def gelu_negative_tail(gen):
    return uniform(ROWS_SHAPE, gen, -12.0, -3.0)


def gelu_tiny_output(gen):
    return uniform(ROWS_SHAPE, gen, -20.0, -12.0)


def gelu_large(gen):
    return log_uniform(ROWS_SHAPE, gen, 4, 127)


def gelu_all_finite_bf16(gen):
    return all_finite_bf16(ROWS_SHAPE, gen)


FAMILIES = {
    "matmul": {
        "random": matmul_random,
        "adversarial": (
            matmul_subnormals,
            matmul_zeros_and_min_normals,
            matmul_cancelling,
            matmul_constant_rows,
            matmul_halfway,
            matmul_halfway_scaled,
            matmul_large,
            matmul_tiny,
        ),
        "special": (matmul_special_nan, matmul_special_inf),
    },
    "reduction": {
        "random": rows_random,
        "adversarial": (
            reduction_subnormals,
            rows_zeros_and_min_normals,
            reduction_cancelling,
            reduction_ascending,
            reduction_descending,
            reduction_constant_rows,
            reduction_wide_range,
            reduction_large,
        ),
        "special": (rows_special_nan, rows_special_inf),
    },
    "gelu": {
        "random": rows_random,
        "adversarial": (
            gelu_near_zero,
            gelu_subnormals,
            rows_zeros_and_min_normals,
            gelu_saturation,
            gelu_negative_tail,
            gelu_tiny_output,
            gelu_large,
            gelu_all_finite_bf16,
        ),
        "special": (rows_special_nan, rows_special_inf),
    },
}


def to_bf16(inputs):
    if isinstance(inputs, tuple):
        return tuple(tensor.to(torch.bfloat16) for tensor in inputs)
    return inputs.to(torch.bfloat16)


def case_name(case):
    return case.__name__.split("_", 1)[1]


def test_inputs(family, smoke=False):
    cases = FAMILIES[family]
    entries = []
    for input_set in ("development", "sealed"):
        chosen = [cases["random"]] * RANDOM_PER_SET + list(cases["adversarial"])
        for index, case in enumerate(chosen):
            if not smoke or index in (0, RANDOM_PER_SET):
                entries.append((input_set, index, case))
    entries += [("special", index, case) for index, case in enumerate(cases["special"])]
    for input_set, index, case in entries:
        gen = generator(family, input_set, index)
        name = f"{input_set}_{index:02d}_{case_name(case)}"
        yield name, input_set, to_bf16(case(gen))


def autotune_input(family):
    return to_bf16(FAMILIES[family]["random"](generator(family, "autotune")))
