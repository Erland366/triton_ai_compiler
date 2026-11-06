from .gelu import GELUBF16OutputBF16, GELUBF16OutputFloat32
from .matrix_multiplication import (
    MatrixMultiplicationBF16OutputBF16,
    MatrixMultiplicationBF16OutputFloat32,
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
    )
}
