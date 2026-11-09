from .gelu import GELUBF16OutputBF16, GELUBF16OutputFloat32
from .matrix_multiplication import (
    MatrixMultiplicationBF16OutputBF16,
    MatrixMultiplicationBF16OutputBF16K1024,
    MatrixMultiplicationBF16OutputFloat32,
    MatrixMultiplicationBF16OutputFloat32K1024,
)
from .reduction_sum import (
    ReductionSumBF16OutputFloat32,
    ReductionSumBF16PerRowOutputFloat32,
)

KERNELS = {
    kernel.__name__: kernel
    for kernel in (
        MatrixMultiplicationBF16OutputBF16,
        MatrixMultiplicationBF16OutputFloat32,
        ReductionSumBF16OutputFloat32,
        ReductionSumBF16PerRowOutputFloat32,
        GELUBF16OutputBF16,
        GELUBF16OutputFloat32,
        MatrixMultiplicationBF16OutputBF16K1024,
        MatrixMultiplicationBF16OutputFloat32K1024,
    )
}

# ReductionSumBF16OutputFloat32 adds tiles with atomic_add and does not repeat on either
# vendor, so it cannot be a reference (docs/research-plan.md).
REFERENCE_KERNELS = (
    "MatrixMultiplicationBF16OutputBF16",
    "MatrixMultiplicationBF16OutputFloat32",
    "ReductionSumBF16PerRowOutputFloat32",
    "GELUBF16OutputBF16",
    "GELUBF16OutputFloat32",
)

# The MatMul kernels at K=1024, a held-out shape: on both vendors MatMul's bits do not
# depend on the config (docs/findings.md F2, F3), so a held-out config cannot test
# whether a method generalizes, but a different summation length can.
HELD_OUT_SHAPE_KERNELS = (
    "MatrixMultiplicationBF16OutputBF16K1024",
    "MatrixMultiplicationBF16OutputFloat32K1024",
)
