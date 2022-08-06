import cupy as cp  # type:ignore
import numpy as np
import math
import scipy.constants
from cupyx import jit  # type:ignore


class Histogram:
    def __init__(self):
        self._hist = None
        self._bin_edges = None
        self._title = ""
        self._xlabel = ""
        self._ylabel = ""

    def add(self, hist):
        if self._hist is None:
            self._hist = hist
        else:
            self._hist += hist


def histogram(
    photon_batch,
    stage,
    neighborhood: float,
    theta_min: float = 0,
    theta_max: float = np.pi,
    phi_min: float = -np.pi,
    phi_max: float = np.pi,
):
    """Make and store a set of histograms."""
    # TODO: do bounds automatically

    x_min = -neighborhood / 2
    x_max = neighborhood / 2
    y_min = -neighborhood / 2
    y_max = neighborhood / 2
    

    bins = 128
    size = photon_batch.size()
    # strids = work per thread
    strides = 32

    grid_size = (int(math.ceil(size / (bins * strides))),)
    block_size = (bins,)

    # for an areal histogram, measure radiosity, power per area, w/m^2
    bin_area_m2 = (y_max - y_min) * (x_max - x_min) / bins
    one_histogram(
        grid_size,
        block_size,
        bins,
        size,
        photon_batch.alive,
        photon_batch.wavelength_nm,
        photon_batch.r_x,
        photon_batch.photons_per_bundle,
        x_min,
        x_max,
        "Radiosity",
        r"X dimension $\mathregular{(m^2)}$",
        r"Radiosity $\mathregular{(W/m^2)}$",
        bin_area_m2,
        photon_batch.duration_s,
        stage._histogram_r_x,
    )

    one_histogram(
        grid_size,
        block_size,
        bins,
        size,
        photon_batch.alive,
        photon_batch.wavelength_nm,
        photon_batch.r_y,
        photon_batch.photons_per_bundle,
        y_min,
        y_max,
        "Radiosity",
        r"Y dimension $\mathregular{(m^2)}$",
        r"Radiosity $\mathregular{(W/m^2)}$",
        bin_area_m2,  # happens to be same as above
        photon_batch.duration_s,
        stage._histogram_r_y,
    )

    # for an angular histogram we're measuring
    # radiant intensity, power per solid angle, w/sr
    bin_area_sr = 4 * np.pi / bins
    # note that the radiant intensity varies a lot by *theta* i.e. not the
    # quantity bucketed here (see below)
    one_histogram_phi(
        grid_size,
        block_size,
        bins,
        size,
        photon_batch.alive,
        photon_batch.wavelength_nm,
        photon_batch.ez_y,
        photon_batch.ez_x,
        photon_batch.photons_per_bundle,
        phi_min,
        phi_max,
        "Radiant Intensity",
        r"Azimuth phi $\mathregular{(radians)}$",
        r"Radiant Intensity $\mathregular{(W/sr)}$",
        bin_area_sr,
        photon_batch.duration_s,
        stage._histogram_ez_phi,
    )

    bin_edges = np.linspace(theta_min, theta_max, bins + 1)
    bin_area_sr = (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:])) * 2 * np.pi
    one_histogram_theta(
        grid_size,
        block_size,
        bins,
        size,
        photon_batch.alive,
        photon_batch.wavelength_nm,
        photon_batch.ez_z,
        photon_batch.photons_per_bundle,
        theta_min,
        theta_max,
        "Radiant Intensity",
        r"Polar angle theta $\mathregular{(radians)}$",
        r"Radiant Intensity $\mathregular{(W/sr)}$",
        bin_area_sr,
        photon_batch.duration_s,
        stage._histogram_ez_theta_weighted,
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    projected_area_m2 = (y_max - y_min) * (x_max - x_min) * np.abs(np.cos(bin_centers))
    bin_area_sr_m2 = (
        (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:])) * 2 * np.pi * projected_area_m2
    )
    one_histogram_theta(
        grid_size,
        block_size,
        bins,
        size,
        photon_batch.alive,
        photon_batch.wavelength_nm,
        photon_batch.ez_z,
        photon_batch.photons_per_bundle,
        theta_min,
        theta_max,
        "Radiance",
        "Polar angle theta $\mathregular{(radians)}$",
        "Radiance $\mathregular{(W/sr/m^2)}$",
        bin_area_sr_m2,
        photon_batch.duration_s,
        stage._histogram_ez_theta_radiance,
    )


# there are 3 of these to avoid conditionals and make it simpler to read


def one_histogram(
    grid_size,
    block_size,
    bins,
    size,
    photon_batch_alive,
    photon_batch_wavelength_nm,
    photon_batch_dimension1,
    photons_per_bundle,
    dim_min,
    dim_max,
    title,
    xlabel,
    ylabel,
    bin_area,
    duration_s,
    histogram_output,
):
    h = cp.zeros(bins, dtype=np.float32)  # joules, so this is joules per bucket
    _histogram(
        grid_size,
        block_size,
        (
            photon_batch_alive,
            photon_batch_wavelength_nm,
            photon_batch_dimension1,
            h,
            np.int32(size),
            np.float32(dim_min),
            np.float32((dim_max - dim_min) / bins),
            np.int32(photons_per_bundle),
        ),
    )

    histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h.get() / (bin_area * duration_s))
    histogram_output._title = title
    histogram_output._xlabel = xlabel
    histogram_output._ylabel = ylabel


@jit.rawkernel()
def _histogram(
    alive: cp.ndarray,  # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32,
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
        # if alive[i]:
        mapped_data = dim_1[i]
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        jit.atomic_add(block_histogram, bucket_idx, alive[i] * energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


def one_histogram_phi(
    grid_size,
    block_size,
    bins,
    size,
    photon_batch_alive,
    photon_batch_wavelength_nm,
    photon_batch_dimension1,
    photon_batch_dimension2,
    photons_per_bundle,
    dim_min,
    dim_max,
    title,
    xlabel,
    ylabel,
    bin_area,
    duration_s,
    histogram_output,
):
    h = cp.zeros(bins, dtype=np.float32)  # joules, so this is joules per bucket

    _histogram_phi(
        grid_size,
        block_size,
        (
            photon_batch_alive,
            photon_batch_wavelength_nm,
            photon_batch_dimension1,
            photon_batch_dimension2,
            h,
            np.int32(size),
            np.float32(dim_min),
            np.float32((dim_max - dim_min) / bins),
            np.int32(photons_per_bundle),
        ),
    )

    histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h.get() / (bin_area * duration_s))
    histogram_output._title = title
    histogram_output._xlabel = xlabel
    histogram_output._ylabel = ylabel


@jit.rawkernel()
def _histogram_phi(
    alive: cp.ndarray,  # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    dim_2: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32,
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
        # if alive[i]:
        mapped_data = cp.arctan2(dim_2[i], dim_1[i])
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        jit.atomic_add(block_histogram, bucket_idx, alive[i] * energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


def one_histogram_theta(
    grid_size,
    block_size,
    bins,
    size,
    photon_batch_alive,
    photon_batch_wavelength_nm,
    photon_batch_dimension1,
    photons_per_bundle,
    dim_min,
    dim_max,
    title,
    xlabel,
    ylabel,
    bin_area,
    duration_s,
    histogram_output,
):
    h = cp.zeros(bins, dtype=np.float32)  # joules, so this is joules per bucket
    _histogram_theta(
        grid_size,
        block_size,
        (
            photon_batch_alive,
            photon_batch_wavelength_nm,
            photon_batch_dimension1,
            h,
            np.int32(size),
            np.float32(dim_min),
            np.float32((dim_max - dim_min) / bins),
            np.int32(photons_per_bundle),
        ),
    )

    histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h.get() / (bin_area * duration_s))
    histogram_output._title = title
    histogram_output._xlabel = xlabel
    histogram_output._ylabel = ylabel


@jit.rawkernel()
def _histogram_theta(
    alive: cp.ndarray,  # bool
    wavelength_nm: cp.ndarray,
    dim_1: cp.ndarray,
    global_histogram: cp.ndarray,
    size: np.int32,
    min_value: np.float32,
    bin_width: np.float32,
    photons_per_bundle: np.int32,
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
        # if alive[i]:
        mapped_data = cp.arccos(dim_1[i])
        bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        # histogram adds up whatever the value is (e.g. joules)
        jit.atomic_add(block_histogram, bucket_idx, alive[i] * energy_per_bundle_j)

    # Wait for all the threads to do it.
    jit.syncthreads()

    # Sum the block-local histograms into a global histogram.
    jit.atomic_add(global_histogram, jit.threadIdx.x, block_histogram[jit.threadIdx.x])


@jit.rawkernel()
def select_and_stack(
    i0, i1, i2, i3, i4, i5, ialive, p, d, oalive, selection_size, scale
):  # Nx3
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
