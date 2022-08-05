import cupy as cp  # type:ignore
import numpy as np
import scipy.constants
from cupyx import jit  # type:ignore


@jit.rawkernel()
def histogram(
    alive: cp.ndarray, # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32
) -> None:
    # Create a histogram for this block.  Size must match bins and threads.
    block_histogram = jit.shared_memory(np.float32, 128)

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
        #if alive[i]:
        mapped_data = dim_1[i]
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        #jit.atomic_add(block_histogram, bucket_idx, energy_per_bundle_j)
        jit.atomic_add(block_histogram, bucket_idx, alive[i]*energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


@jit.rawkernel()
def histogram_phi(
    alive: cp.ndarray, # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    dim_2: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32
) -> None:
    # Create a histogram for this block.  Size must match bins and threads.
    block_histogram = jit.shared_memory(np.float32, 128)

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
        #if alive[i]:
        mapped_data = cp.arctan2(dim_2[i], dim_1[i])
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        #jit.atomic_add(block_histogram, bucket_idx, energy_per_bundle_j)
        jit.atomic_add(block_histogram, bucket_idx, alive[i]*energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])

@jit.rawkernel()
def histogram_theta(
    alive: cp.ndarray, # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32
) -> None:
    # Create a histogram for this block.  Size must match bins and threads.
    block_histogram = jit.shared_memory(np.float32, 128)

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
        #if alive[i]:
        mapped_data = cp.arccos(dim_1[i])
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        #jit.atomic_add(block_histogram, bucket_idx, energy_per_bundle_j)
        jit.atomic_add(block_histogram, bucket_idx, alive[i]*energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


@jit.rawkernel()
def select_and_stack(i0, i1, i2, i3, i4, i5, ialive, p, d, oalive, selection_size, scale):  # Nx3
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
        oalive[idx] = ialive[s_i]

