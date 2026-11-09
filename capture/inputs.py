"""Test inputs for the BF16 kernels, generated on the CPU from fixed seeds.

Each family has 16 development and 16 sealed inputs (8 random and 8 adversarial each,
with different seeds on each side) and 2 special inputs holding NaN and infinity. The
cases follow docs/research-plan.md. Inputs are regenerated rather than copied between
machines, so each capture records their SHA-256 for the other side to check.
"""

import functools
import zlib

import torch

BF16_MIN_SUBNORMAL = 2.0**-133
BF16_MIN_NORMAL = 2.0**-126
MATMUL_DIMS = (4096, 4096, 4096)
MATMUL_K1024_DIMS = (4096, 1024, 4096)
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


def matmul_random(gen, dims):
    m, k, n = dims
    return randn((m, k), gen), randn((k, n), gen)


def matmul_subnormals(gen, dims):
    m, k, n = dims
    a = mix(randn((m, k), gen), subnormals((m, k), gen), 0.5, gen)
    return a, randn((k, n), gen)


def matmul_zeros_and_min_normals(gen, dims):
    m, k, n = dims
    return zeros_and_min_normals((m, k), gen), zeros_and_min_normals((k, n), gen)


def matmul_cancelling(gen, dims):
    m, k, n = dims
    a, b = randn((m, k), gen), randn((k, n), gen)
    b[1::2] = b[0::2]
    half = (m, k // 2)
    large = (2.0**40 * uniform(half, gen, 1.0, 2.0)).to(torch.bfloat16).float()
    mask = torch.rand(half, generator=gen) < 0.25
    a[:, 0::2] = torch.where(mask, large, a[:, 0::2])
    a[:, 1::2] = torch.where(mask, -large, a[:, 1::2])
    return a, b


def matmul_constant_rows(gen, dims):
    m, k, n = dims
    a = randn((m, 1), gen).expand((m, k)).contiguous()
    return a, randn((k, n), gen)


def sparse_offsets(gen, shape, density, magnitudes):
    choice = torch.randint(0, len(magnitudes), shape, generator=gen)
    values = signs(shape, gen) * torch.tensor(magnitudes)[choice]
    mask = torch.rand(shape, generator=gen) < density
    return torch.where(mask, values, torch.zeros(shape))


def matmul_halfway(gen, dims):
    m, k, n = dims
    magnitudes = [1.0, 2.0**-6, 2.0**-7, 2.0**-8]
    a = sparse_offsets(gen, (m, k), 4 / k, magnitudes)
    return a, signs((k, n), gen)


def matmul_halfway_scaled(gen, dims):
    a, b = matmul_halfway(gen, dims)
    exponents = torch.randint(-60, 61, (dims[0], 1), generator=gen).float()
    return a * torch.exp2(exponents), b


def matmul_large(gen, dims):
    m, k, n = dims
    return randn((m, k), gen) * 2.0**56, randn((k, n), gen) * 2.0**56


def matmul_tiny(gen, dims):
    m, k, n = dims
    return randn((m, k), gen) * 2.0**-63, randn((k, n), gen) * 2.0**-63


def matmul_special_nan(gen, dims):
    m, k, n = dims
    a = inject_specials(randn((m, k), gen), gen, NAN_BITS + INF_BITS, 0.01)
    return a, randn((k, n), gen)


def matmul_special_inf(gen, dims):
    m, k, n = dims
    a = inject_specials(randn((m, k), gen), gen, INF_BITS, 0.001)
    b = inject_specials(randn((k, n), gen), gen, INF_BITS, 0.001)
    return a, b


def matmul_family(dims):
    def bind(case):
        bound = functools.partial(case, dims=dims)
        bound.__name__ = case.__name__
        return bound

    return {
        "random": bind(matmul_random),
        "adversarial": tuple(
            bind(case)
            for case in (
                matmul_subnormals,
                matmul_zeros_and_min_normals,
                matmul_cancelling,
                matmul_constant_rows,
                matmul_halfway,
                matmul_halfway_scaled,
                matmul_large,
                matmul_tiny,
            )
        ),
        "special": (bind(matmul_special_nan), bind(matmul_special_inf)),
    }


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
    "matmul": matmul_family(MATMUL_DIMS),
    "matmul_k1024": matmul_family(MATMUL_K1024_DIMS),
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
