import cupy as cp  # type:ignore
import numpy as np
import math
import scipy.constants
from cupyx import jit  # type:ignore
import scattering


class Histogram:
    def __init__(self):
        self._hist: cp.ndarray = None # might be 1d or multi-d
        self._bin_edges = None # might be cp.ndarray or List[cp.ndarray]
        self._title: str = ""
        self._xlabel: str = ""
        self._ylabel: str = ""

    def add(self, hist):
        if type(hist) != cp.ndarray:
            raise ValueError("histogram must be cp.ndarray")
        if self._hist is None:
            self._hist = hist
        else:
            self._hist += hist


# only useful for small runs since it remembers everything
class Scatter:
    def __init__(self):
        self._x: cp.ndarray = None
        self._y: cp.ndarray = None
        self._title: str = ""
        self._xlabel: str = ""
        self._ylabel: str = ""

    def add(self, x, y):
        if type(x) != cp.ndarray:
            raise ValueError("x must be cp.ndarray")
        if type(y) != cp.ndarray:
            raise ValueError("y must be cp.ndarray")
        if self._x is None:
            self._x = x
        # for now just disable the infinite memory part
        # else:
        # self._x = cp.concatenate((self._x, x))
        if self._y is None:
            self._y = y
        # else:
        # self._y = cp.concatenate((self._y, y))


def histogram(photon_batch, stage):
    """Make and store a set of histograms."""

    neighborhood = stage._size_m
    theta_min = stage._theta_min
    theta_max = stage._theta_max
    phi_min = -np.pi
    phi_max = np.pi

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
        "Count (W)",
        r"Polar angle theta $\mathregular{(radians)}$",
        r"Count $\mathregular{(W)}$",
        np.float32(1),
        photon_batch.duration_s,
        stage._histogram_ez_theta_count,
    )

    bin_edges = cp.linspace(theta_min, theta_max, bins + 1)
    bin_area_sr = (cp.cos(bin_edges[:-1]) - cp.cos(bin_edges[1:])) * 2 * np.pi
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
        stage._histogram_ez_theta_intensity,
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    projected_area_m2 = (y_max - y_min) * (x_max - x_min) * cp.abs(cp.cos(bin_centers))
    bin_area_sr_m2 = (
        (cp.cos(bin_edges[:-1]) - cp.cos(bin_edges[1:])) * 2 * np.pi * projected_area_m2
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

    histogram_4d(
        photon_batch,
        x_min,
        x_max,
        y_min,
        y_max,
        r"Maximum intensity $\mathregular{W/sr)}$",
        r"Maximum radiance $\mathregular{W/sr\, m^2)}$",
        "X (m)",
        "Y (m)",
        stage._histogram_4d_intensity,
        stage._histogram_4d_radiance,
    )
    counter(
        photon_batch,
        x_min,
        x_max,
        y_min,
        y_max,
        "Bundle count, unscaled",
        "X (m)",
        "Y (m)",
        stage._histogram_4d_count,
    )
    scatterplot(photon_batch, "count theta vs x", "x", "theta", stage._scatter)
    spectrum(photon_batch, stage._photons_spectrum)

def spectrum(photon_batch, histogram_output):
    counts, bins = cp.histogram(photon_batch.wavelength_nm,
                                256, range=(360, 780),
                                weights=photon_batch.alive)
    histogram_output._bin_edges = bins
    histogram_output.add(counts)
    histogram_output._title = "spectrum"
    histogram_output._xlabel = "wavelength (nm)"
    histogram_output._ylabel = "count"


# TODO: this isn't very useful, get rid of it.
def counter(
    photons, x_min, x_max, y_min, y_max, title, xlabel, ylabel, histogram_output
):

    points = (
        photons.r_x,
        photons.r_y,
        cp.arccos(photons.ez_z),  # theta
        cp.arctan2(photons.ez_y, photons.ez_x),
    )  # phi

    (ct_per_bin, edges) = cp.histogramdd(
        points,
        bins=(27, 27, 27, 18),
        # bins=(18,18,16,32),
        range=((x_min, x_max), (y_min, y_max), (0.00001, 0.99999*np.pi), (-np.pi, np.pi)),
        weights=photons.alive,
    )

    histogram_output._bin_edges = edges
    histogram_output.add(ct_per_bin)

    histogram_output._title = title
    histogram_output._xlabel = xlabel
    histogram_output._ylabel = ylabel


def scatterplot(photons, title, xlabel, ylabel, scatter_output):
    # just retain a window in x

    ####
    # center
    window_min = -0.001
    window_max = 0.001

    x = photons.r_x
    x = cp.compress(
        cp.logical_and(
            cp.logical_and(photons.alive, photons.r_y < window_max),
            photons.r_y > window_min,
        ),
        x,
        axis=0,
    )
    y = cp.arccos(photons.ez_z)
    y = cp.compress(
        cp.logical_and(
            cp.logical_and(photons.alive, photons.r_y < window_max),
            photons.r_y > window_min,
        ),
        y,
        axis=0,
    )
    scatter_output.add(x, y)
    scatter_output._title = title
    scatter_output._xlabel = xlabel
    scatter_output._ylabel = ylabel


def histogram_4d(
    photons,
    x_min,
    x_max,
    y_min,
    y_max,
    intensity_title,
    radiance_title,
    xlabel,
    ylabel,
    intensity_histogram_output,
    radiance_histogram_output,
):

    points = (
        photons.r_x,
        photons.r_y,
        cp.arccos(photons.ez_z),  # theta
        cp.arctan2(photons.ez_y, photons.ez_x),
    )  # phi

    wavelength_m = photons.wavelength_nm * 1e-9
    frequency_hz = scipy.constants.c / wavelength_m
    energy_per_photon_j = scipy.constants.h * frequency_hz
    energy_per_bundle_j = energy_per_photon_j * photons.photons_per_bundle

    # joules
    (energy_per_bin_j, edges) = cp.histogramdd(
        points,
        bins=(27, 27, 27, 18),
        # there's an intensity singularity at the pole, and a radiance
        # singularity at 90 degrees, so carve those points out
        range=(
            (x_min, x_max),
            (y_min, y_max),
            ###(0.01, 0.99 * np.pi / 2),
            (0.00001, 0.99999 * np.pi),
            (-np.pi, np.pi),
        ),
        weights=photons.alive * energy_per_bundle_j,
    )

    # would this be better as a constant?
    bin_area_m2 = cp.outer(edges[0][1:] - edges[0][:-1], edges[1][1:] - edges[1][:-1])

    bin_area_m2_stretched = bin_area_m2[:, :, None, None]

    bin_area_sr = cp.outer(
        cp.cos(edges[2][:-1]) - cp.cos(edges[2][1:]), edges[3][1:] - edges[3][:-1]
    )

    bin_area_sr_stretched = bin_area_sr[None, None, :, :]

    power_per_bin_w = energy_per_bin_j / photons.duration_s
    intensity_w_sr = power_per_bin_w / bin_area_sr_stretched

    # TODO: the bin edges may not be the same from batch to batch
    intensity_histogram_output._bin_edges = edges
    intensity_histogram_output.add(intensity_w_sr)

    intensity_histogram_output._title = intensity_title
    intensity_histogram_output._xlabel = xlabel
    intensity_histogram_output._ylabel = ylabel

    theta_bin_centers = (edges[2][:-1] + edges[2][1:]) / 2
    theta_bin_centers_stretched = theta_bin_centers[None, None, :, None]

    projected_area_factor = cp.abs(cp.cos(theta_bin_centers_stretched))
    radiance_w_sr_m2 = intensity_w_sr / (bin_area_m2_stretched * projected_area_factor)

    radiance_histogram_output._bin_edges = edges
    radiance_histogram_output.add(radiance_w_sr_m2)

    radiance_histogram_output._title = radiance_title
    radiance_histogram_output._xlabel = xlabel
    radiance_histogram_output._ylabel = ylabel


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

    histogram_output._bin_edges = cp.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h / (bin_area * duration_s))
    histogram_output._title = title
    histogram_output._xlabel = xlabel
    histogram_output._ylabel = ylabel


# TODO just use cp.histogram
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
        #################
        # just ignore out of bounds for now
        # bucket_idx = int((mapped_data - min_value) // bin_width)
        raw_bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(raw_bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        energy_per_bundle_j *= raw_bucket_idx >= 0
        energy_per_bundle_j *= raw_bucket_idx <= 127

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

    histogram_output._bin_edges = cp.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h / (bin_area * duration_s))
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
        raw_bucket_idx = int((mapped_data - min_value) // bin_width)
        #################
        # just ignore out of bounds for now
        # bucket_idx = int(min(max(bucket_idx, 0), 127))  # must match above
        bucket_idx = int(min(max(raw_bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle
        energy_per_bundle_j *= raw_bucket_idx >= 0
        energy_per_bundle_j *= raw_bucket_idx <= 127

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
    bin_area: cp.ndarray,
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

    histogram_output._bin_edges = cp.linspace(dim_min, dim_max, bins + 1)
    histogram_output.add(h / (bin_area * duration_s))
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
        #################
        # just ignore out of bounds for now
        # bucket_idx = int((mapped_data - min_value) // bin_width)
        raw_bucket_idx = int((mapped_data - min_value) // bin_width)
        bucket_idx = int(min(max(raw_bucket_idx, 0), 127))  # must match above

        wavelength_m = wavelength_nm[i] * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle

        energy_per_bundle_j *= raw_bucket_idx >= 0
        energy_per_bundle_j *= raw_bucket_idx <= 127

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


def sample_pdf(size: int, pdf_x: cp.ndarray, pdf_y: cp.ndarray) -> cp.ndarray:
    """Given a pdf, produce samples."""
    cdf = cp.cumsum(pdf_y)
    cdf = cdf / cdf[-1]

    rng = cp.random.default_rng()
    values = rng.random(size)
    value_bin = cp.searchsorted(cdf, values)
    random_from_cdf = pdf_x[value_bin]
    return random_from_cdf
