from types import MappingProxyType

import torch
import triton
import triton.language as tl


class MatrixMultiplicationBF16OutputBF16:
    input_family = "matmul"
    tuning_options = MappingProxyType(
        {
            "block_m": (32, 64, 128, 256),
            "block_n": (32, 64, 128, 256),
            "block_k": (32, 64, 128, 256),
            "num_warps": (4, 8, 16),
        }
    )
    default_config = MappingProxyType(
        {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4}
    )

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_cm: tl.constexpr,
        k_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(tl.cdiv(k_dim, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.bfloat16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, c)

    def random_input(self, generator):
        a = torch.randn((4096, 4096), generator=generator, dtype=torch.float32)
        b = torch.randn((4096, 4096), generator=generator, dtype=torch.float32)
        return a.to(torch.bfloat16), b.to(torch.bfloat16)

    def forward(self, inputs):
        a, b = inputs
        c = torch.empty_like(a)
        grid = (
            triton.cdiv(a.shape[0], self.config["block_m"]),
            triton.cdiv(b.shape[1], self.config["block_n"]),
        )
        self.last_launch = self.kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            b.stride(0),
            c.stride(0),
            a.shape[1],
            BLOCK_M=self.config["block_m"],
            BLOCK_N=self.config["block_n"],
            BLOCK_K=self.config["block_k"],
            num_warps=self.config["num_warps"],
        )
        return c

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a.float(), b.float()).to(torch.bfloat16)


class MatrixMultiplicationBF16OutputFloat32:
    input_family = "matmul"
    tuning_options = MappingProxyType(
        {
            "block_m": (32, 64, 128, 256),
            "block_n": (32, 64, 128, 256),
            "block_k": (32, 64, 128, 256),
            "num_warps": (4, 8, 16),
        }
    )
    default_config = MappingProxyType(
        {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4}
    )

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_cm: tl.constexpr,
        k_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(tl.cdiv(k_dim, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, accumulator)

    def random_input(self, generator):
        a = torch.randn((4096, 4096), generator=generator, dtype=torch.float32)
        b = torch.randn((4096, 4096), generator=generator, dtype=torch.float32)
        return a.to(torch.bfloat16), b.to(torch.bfloat16)

    def forward(self, inputs):
        a, b = inputs
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
        grid = (
            triton.cdiv(a.shape[0], self.config["block_m"]),
            triton.cdiv(b.shape[1], self.config["block_n"]),
        )
        self.last_launch = self.kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            b.stride(0),
            c.stride(0),
            a.shape[1],
            BLOCK_M=self.config["block_m"],
            BLOCK_N=self.config["block_n"],
            BLOCK_K=self.config["block_k"],
            num_warps=self.config["num_warps"],
        )
        return c

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a.float(), b.float())


class MatrixMultiplicationBF16OutputBF16K1024(MatrixMultiplicationBF16OutputBF16):
    input_family = "matmul_k1024"

    def random_input(self, generator):
        a = torch.randn((4096, 1024), generator=generator, dtype=torch.float32)
        b = torch.randn((1024, 4096), generator=generator, dtype=torch.float32)
        return a.to(torch.bfloat16), b.to(torch.bfloat16)


class MatrixMultiplicationBF16OutputFloat32K1024(MatrixMultiplicationBF16OutputFloat32):
    input_family = "matmul_k1024"

    def random_input(self, generator):
        a = torch.randn((4096, 1024), generator=generator, dtype=torch.float32)
        b = torch.randn((1024, 4096), generator=generator, dtype=torch.float32)
        return a.to(torch.bfloat16), b.to(torch.bfloat16)
