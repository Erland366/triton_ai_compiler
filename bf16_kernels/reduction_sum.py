from types import MappingProxyType

import torch
import triton
import triton.language as tl


class ReductionSumBF16OutputFloat32:
    size = 4096
    batch_size = 256
    tuning_options = MappingProxyType(
        {
            "block_size": (32, 64, 128, 256, 512, 1024, 2048),
            "num_warps": (4, 8, 16),
        }
    )
    default_config = MappingProxyType({"block_size": 1024, "num_warps": 8})

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        batch_index = tl.program_id(axis=1)
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + batch_index * 4096 + offsets).to(tl.float32)
        tl.atomic_add(output_ptr + batch_index, tl.sum(values, axis=0))

    def random_input(self, generator):
        x = torch.randn(
            (self.batch_size, self.size), generator=generator, dtype=torch.float32
        )
        return x.to(torch.bfloat16)

    def forward(self, x):
        output = torch.zeros(self.batch_size, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(self.size, self.config["block_size"]), self.batch_size)
        self.kernel[grid](
            x,
            output,
            BLOCK_SIZE=self.config["block_size"],
            num_warps=self.config["num_warps"],
        )
        return output

    def forward_torch(self, x):
        return x.float().sum(dim=1)


class ReductionSumBF16PerRowOutputFloat32:
    size = 4096
    batch_size = 256
    tuning_options = MappingProxyType({"num_warps": (4, 8, 16)})
    default_config = MappingProxyType({"num_warps": 4})

    def __init__(self, config=None):
        self.config = dict(config or self.default_config)

    @staticmethod
    @triton.jit
    def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(axis=0)
        values = tl.load(x_ptr + row * 4096 + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
        tl.store(output_ptr + row, tl.sum(values, axis=0))

    def random_input(self, generator):
        x = torch.randn(
            (self.batch_size, self.size), generator=generator, dtype=torch.float32
        )
        return x.to(torch.bfloat16)

    def forward(self, x):
        output = torch.empty(self.batch_size, device=x.device, dtype=torch.float32)
        self.kernel[(self.batch_size,)](
            x,
            output,
            BLOCK_SIZE=self.size,
            num_warps=self.config["num_warps"],
        )
        return output

    def forward_torch(self, x):
        return x.float().sum(dim=1)
