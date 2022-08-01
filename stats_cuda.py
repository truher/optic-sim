import cupy as cp  # type:ignore
import numpy as np
from cupyx import jit  # type:ignore


@jit.rawkernel()
def histogram(
    input_data_1: cp.ndarray,
    input_data_2: cp.ndarray,
    mapper: np.int32,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
) -> None:
    """Histogram with a map function.

    That way you can transform coordinates on the fly, you don't need to store the transformed
    version.  For example, store cartesian, but histogram spherical.

    input_data_1: cp.ndarray[np.float32]
    input_data_2: cp.ndarray[np.float32], use for phi, 1:x, 2:y
    mapper: 0 = none, 1 = phi, 2 = theta
    global_histogram: cp.ndarray[np.int32]
    size: of the input
    """
    # Create a histogram for this block.  Size must match bins and threads.
    block_histogram = jit.shared_memory(np.int32, 128)

    # Alloc is not zeroed so do it.
    block_histogram[jit.threadIdx.x] = 0

    # Wait for all threads to set their zeros.
    jit.syncthreads()

    # Fill up the block-local histogram.
    # Adjacent threads get adjacent data:
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    # If there's more than one grid (not recommended) then do that too:
    ntid = jit.gridDim.x * jit.blockDim.x

    for i in range(tid, size, ntid):
        if mapper == 1:  # phi
            mapped_data = cp.arctan2(input_data_2[i], input_data_1[i])
        elif mapper == 2:  # theta
            mapped_data = cp.arccos(input_data_1[i])
        else:
            mapped_data = input_data_1[i]
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above
        jit.atomic_add(block_histogram, bucket_idx, 1)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


@jit.rawkernel()
def select_and_stack(i0, i1, i2, i3, i4, i5, p, d, selection_size, scale):  # Nx3
    """Take six big vectors as input (three position three direction)
    and return two smaller 3d vectors (one position one direction) suitable for
    plotting.
    """
    idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if idx < selection_size:
        s_i = idx * scale
        p[(idx, 0)] = i0[s_i]
        p[(idx, 1)] = i1[s_i]
        p[(idx, 2)] = i2[s_i]
        d[(idx, 0)] = i3[s_i]
        d[(idx, 1)] = i4[s_i]
        d[(idx, 2)] = i5[s_i]
