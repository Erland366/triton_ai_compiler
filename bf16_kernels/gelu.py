from types import MappingProxyType

import torch
import triton
import triton.language as tl


class GELUBF16OutputBF16:
    size = 4096
    batch_size = 256
    tuning_options = MappingProxyType(
        {
            "block_size": (32, 64, 128, 256, 512, 1024, 2048),
            "num_warps": (4, 8, 16),
        }
    )
    default_config = MappingProxyType({"block_size": 1024, "num_warps": 4})

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        output = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, output.to(tl.bfloat16))

    def random_input(self, generator):
        x = torch.randn(
            (self.batch_size, self.size), generator=generator, dtype=torch.float32
        )
        return x.to(torch.bfloat16)

    def forward(self, x):
        output = torch.empty_like(x)
        self.kernel[(triton.cdiv(x.numel(), self.config["block_size"]),)](
            x,
            output,
            BLOCK_SIZE=self.config["block_size"],
            num_warps=self.config["num_warps"],
        )
        return output

    def forward_torch(self, x):
        return torch.nn.functional.gelu(x.float()).to(torch.bfloat16)


class GELUBF16OutputFloat32:
    size = 4096
    batch_size = 256
    tuning_options = MappingProxyType(
        {
            "block_size": (32, 64, 128, 256, 512, 1024, 2048),
            "num_warps": (4, 8, 16),
        }
    )
    default_config = MappingProxyType({"block_size": 1024, "num_warps": 4})

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        output = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, output)

    def random_input(self, generator):
        x = torch.randn(
            (self.batch_size, self.size), generator=generator, dtype=torch.float32
        )
        return x.to(torch.bfloat16)

    def forward(self, x):
        output = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        self.kernel[(triton.cdiv(x.numel(), self.config["block_size"]),)](
            x,
            output,
            BLOCK_SIZE=self.config["block_size"],
            num_warps=self.config["num_warps"],
        )
        return output

    def forward_torch(self, x):
        return torch.nn.functional.gelu(x.float())
