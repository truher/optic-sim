from typing import Tuple
import cupy as cp  # type: ignore
from cupyx import jit  # type: ignore
import math
import numpy as np
import scipy.constants
import stats_cuda
import scattering


class Photons:
    """wraps cuda arrays"""

    def __init__(self):
        # location
        self.r_x = None
        self.r_y = None
        self.r_z = None
        # direction .. note this is too many degrees of freedom,
        # it might be better to use phi/theta only even though
        # it's a bit more computation
        self.ez_x = None
        self.ez_y = None
        self.ez_z = None
        self.alive = None
        self.wavelength_nm = None  # [int16]
        self.photons_per_bundle = 0  # should be like 1e7 ish?
        self.duration_s = 0  # for computing power

    def size(self):
        return np.int32(self.r_x.size)

    def count_alive(self):
        return cp.count_nonzero(self.alive)

    def retain(self, p):
        """Retain photons with (vector) probability p."""
        cp.logical_and(self.alive, cp.random.random(self.size()) < p, out=self.alive)

    def remove(self, p):
        """Remove photons with (scalar) probability p."""
        # TODO: do this without allocating a giant vector
        if p < 0.001: # shortcut, do i need this?
            return
        cp.logical_and(self.alive, cp.random.random(self.size()) > p, out=self.alive)

    def prune_outliers(self, size):
        """set out-of-bounds photons to dead"""
        cp.logical_and(self.alive, self.r_x >= -size / 2, out=self.alive)
        cp.logical_and(self.alive, self.r_x <= size / 2, out=self.alive)
        cp.logical_and(self.alive, self.r_y >= -size / 2, out=self.alive)
        cp.logical_and(self.alive, self.r_y <= size / 2, out=self.alive)

    @staticmethod
    def compress_dead(alive, x):
        return cp.compress(alive, x, axis=0)

    def debug(self, source_size_m):
        # return
        energy_j = self.energy_j()
        print(f"photon batch energy joules: {energy_j:.3e}")
        power_w = self.power_w()
        print(f"photon batch power watts: {power_w:.3e}")
        emitter_area_m2 = source_size_m * source_size_m
        print(f"emitter area m^2: {emitter_area_m2:.3e}")
        radiosity_w_m2 = power_w / emitter_area_m2
        print(f"batch radiosity w/m^2: {radiosity_w_m2:.3e}")

    def sample(self):
        """Take every N-th for plotting 1024.  Returns
        a type the plotter likes, which is two numpy (N,3) vectors"""
        size = self.size()
        alive_count = self.count_alive()
        alive_ratio = alive_count / size
        # block_size = 64 # 0.45 s
        block_size = 4  # more waves = less sampling
        # choose extra to compensate for deadness
        # allow approximate return count
        grid_size = int(math.ceil(16 / alive_ratio))
        selection_size = min(size, grid_size * block_size)
        scale = np.int32(size // selection_size)

        # mempool = cp.get_default_memory_pool()
        # mempool.free_all_blocks()

        position_3d = cp.zeros((selection_size, 3), dtype=np.float32)
        direction_3d = cp.zeros((selection_size, 3), dtype=np.float32)
        alive_selected = cp.zeros(selection_size, dtype=bool)

        stats_cuda.select_and_stack(
            (grid_size,),
            (block_size,),
            (
                self.r_x,
                self.r_y,
                self.r_z,
                self.ez_x,
                self.ez_y,
                self.ez_z,
                self.alive,
                position_3d,
                direction_3d,
                alive_selected,
                selection_size,
                scale,
            ),
        )

        position_3d = Photons.compress_dead(alive_selected, position_3d)
        direction_3d = Photons.compress_dead(alive_selected, direction_3d)
        return (position_3d.get(), direction_3d.get())

    @staticmethod
    @cp.fuse()
    def energy_j_kernel(wavelength_nm, photons_per_bundle, alive):
        wavelength_m = wavelength_nm * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle * alive
        return cp.sum(energy_per_bundle_j)

    def energy_j(self) -> float:
        """Energy of this photon bundle."""
        return Photons.energy_j_kernel(
            self.wavelength_nm, self.photons_per_bundle, self.alive
        )

    def power_w(self) -> float:
        return self.energy_j() / self.duration_s


class PhotonsStacked:
    def __init__(self):
        self._p = None
        self._d = None

    def add(self, stack):
        (p, d) = stack
        if p is None:
            raise ValueError()
        if d is None:
            raise ValueError()
        if self._p is None:
            self._p = p
            self._d = d
        else:
            self._p = np.concatenate([self._p, p])
            self._d = np.concatenate([self._d, d])


class Source:
    def make_photons(self, bundles: np.int32) -> Photons:
        raise NotImplementedError()


@jit.rawkernel()
def spherical_to_cartesian_raw(theta_z, phi_x, y, size) -> None:
    """In-place calculation reuses the inputs."""
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        y[i] = cp.sin(theta_z[i]) * cp.sin(phi_x[i])
        phi_x[i] = cp.sin(theta_z[i]) * cp.cos(phi_x[i])
        theta_z[i] = cp.cos(theta_z[i])

class PencilSource(Source):
    """Zero area zero divergence."""
    def __init__(self, wavelength_nm: int, photons_per_bundle: float, duration_s: float):
        self._wavelength_nm = wavelength_nm
        self._photons_per_bundle = photons_per_bundle
        self._duration_s = duration_s

    def make_photons(self, bundles: np.int32) -> Photons:
        photons = Photons()
        photons.r_x = cp.zeros(bundles, dtype=np.float32)
        photons.r_y = cp.zeros(bundles, dtype=np.float32)
        photons.r_z = cp.zeros(bundles, dtype=np.float32)
        photons.ez_x = cp.zeros(bundles, dtype=np.float32)
        photons.ez_y = cp.zeros(bundles, dtype=np.float32)
        photons.ez_z = cp.ones(bundles, dtype=np.float32)
        photons.alive = cp.ones(bundles, dtype=bool)
        photons.wavelength_nm = cp.full(bundles, self._wavelength_nm, dtype=np.uint16)
        photons.photons_per_bundle = self._photons_per_bundle
        photons.duration_s = self._duration_s
        return photons


class MonochromaticLambertianSource(Source):
    def __init__(
        self,
        width_m: float,
        height_m: float,
        wavelength_nm: int,
        photons_per_bundle: float,
        duration_s: float,
    ):
        self._width_m = width_m
        self._height_m = height_m
        self._wavelength_nm = wavelength_nm
        self._photons_per_bundle = photons_per_bundle
        self._duration_s = duration_s

    def make_photons(self, bundles: np.int32) -> Photons:
        photons = Photons()
        photons.r_x = cp.random.uniform(
            -0.5 * self._width_m, 0.5 * self._width_m, bundles, dtype=np.float32
        )
        photons.r_y = cp.random.uniform(
            -0.5 * self._height_m, 0.5 * self._height_m, bundles, dtype=np.float32
        )
        photons.r_z = cp.full(bundles, 0.0, dtype=np.float32)
        # phi, reused as x
        photons.ez_x = cp.random.uniform(0, 2 * np.pi, bundles, dtype=np.float32)
        photons.ez_y = cp.empty(bundles, dtype=np.float32)
        # theta, reused as z
        photons.ez_z = (
            cp.arccos(cp.random.uniform(-1, 1, bundles, dtype=np.float32)) / 2
        )
        spherical_to_cartesian_raw(
            (128,), (1024,), (photons.ez_z, photons.ez_x, photons.ez_y, bundles)
        )
        photons.alive = cp.ones(bundles, dtype=bool)
        photons.wavelength_nm = cp.full(bundles, self._wavelength_nm, dtype=np.uint16)
        photons.photons_per_bundle = self._photons_per_bundle
        photons.duration_s = self._duration_s
        return photons


class Lightbox:
    """Represents the box between the source and diffuser.

    Sides are somewhat reflective.
    """

    def __init__(self, height: float, size: float):
        """
        height: top of the box above the source
        size: full length or width, box is square.
        """
        self._height = height
        self._size = size

    def propagate_without_kernel(self, photons: Photons) -> None:
        """Avoid conditionals and cuda kernels.  This ignores the
        xy position of the source, since it's small relative to the box."""
        absorption = np.float32(0.1)  # polished metal inside

        r_x_box_widths = self._height * photons.ez_x / (photons.ez_z * self._size)
        r_y_box_widths = self._height * photons.ez_y / (photons.ez_z * self._size)

        reflection_count_x = cp.abs(cp.round(r_x_box_widths))
        reflection_count_y = cp.abs(cp.round(r_y_box_widths))

        photons.r_x = (
            self._size * (2 * cp.abs(cp.mod(r_x_box_widths - 0.5, 2) - 1) - 1) / 2
        )
        photons.r_y = (
            self._size * (2 * cp.abs(cp.mod(r_y_box_widths - 0.5, 2) - 1) - 1) / 2
        )
        photons.r_z = cp.full(photons.size(), self._height, dtype=np.float32)

        cp.multiply(
            photons.ez_x, (1 - 2 * cp.mod(reflection_count_x, 2)), out=photons.ez_x
        )
        cp.multiply(
            photons.ez_y, (1 - 2 * cp.mod(reflection_count_y, 2)), out=photons.ez_y
        )

        total_reflection_count = reflection_count_x + reflection_count_y
        photon_survival = cp.power((1 - absorption), total_reflection_count)
        photons.alive = cp.logical_and(
            photons.alive,
            cp.logical_and(
                cp.less(cp.random.random(photons.size()), photon_survival),
                cp.greater(photons.ez_z, 0),
            ),
            out=photons.alive,
        )

def schlick_reflection(n_1: float, n_2: float, cos_theta_rad: cp.ndarray):
    """passing cos(theta) is more convenient
    This does not account for total internal reflection, so maybe only useful
    for rough surfaces.
    """
    r_0 = ((n_1 - n_2)/(n_1 + n_2)) ** 2
    r = r_0 + (1 - r_0) * (1 - cos_theta_rad) ** 5
    return r

def schlick_reflection_with_tir(ni: float, nt: float, cosX: cp.ndarray):
    """ni = incident side, nt = transmitted side.
    For rough surfaces, total internal reflection is much reduced,
    do don't use this one.
    """
    r_0 = ((nt - ni)/(nt + ni)) ** 2
    if ni > nt:
        inv_eta = ni/nt
        sinT2 = inv_eta * inv_eta * (1 - cosX * cosX)
        sinT2 = cp.minimum(sinT2, 1)
        cosX = cp.sqrt(1 - sinT2)
    r = r_0 + (1 - r_0) * (1 - cosX) ** 5
    return r


class AcryliteDiffuser:
    """0D010 DF Acrylite Satinice 'optimum light diffusion' colorless.

       Transmission is 84% for a normal pencil beam; some is absorbed
       internally, some is reflected internally.  FWHM is 40 degrees.
    """
    N_AIR = 1.0
    N_ACRYLIC = 1.495
    def __init__(self):
        self._scattering = scattering.AcryliteScattering()
        # internal absorption, calibrated to 84% total transmission
        # for a pencil beam
        self._absorption = 0.0814

    def diffuse(self, photons):
        # remove photons reflected at the entry surface (air -> acrylic)
        photons.retain(1 - schlick_reflection(AcryliteDiffuser.N_AIR,
            AcryliteDiffuser.N_ACRYLIC, photons.ez_z))

        # adjust the angles
        size = np.int32(photons.size())  # TODO eliminate this
        phi = scattering.get_scattering_phi(size)
        theta = self._scattering.get_scattering_theta(size)
        block_size = 1024  # max
        grid_size = int(math.ceil(size / block_size))
        scattering.scatter(
            (grid_size,),
            (block_size,),
            (photons.ez_x, photons.ez_y, photons.ez_z, theta, phi, size),
        )

        # remove photons absorbed internally
        photons.remove(self._absorption)

        # remove photons reflected at the exit surface (acrylic -> air)
        photons.retain(1 - schlick_reflection(AcryliteDiffuser.N_ACRYLIC,
            AcryliteDiffuser.N_AIR, photons.ez_z))

class Diffuser:
    """Something that changes photon direction.

    Examples: diffuser, retroreflector.
    """

    def __init__(self, g: float, absorption: float):
        """
        g: Henyey and Greenstein scattering parameter.
            0 is iso, 1 is no scattering, -1 is reflection.
        absorption: mostly useful for the diffuser
        """
        self._g = np.float32(g)
        self._absorption = np.float32(absorption)
        # TODO: the actual absorption (and reflection) depends on the incident and scattered angle
        # like there should be zero emission at 90 degrees.

    def diffuse(self, photons: Photons) -> None:
        """Adjust propagation direction."""
        # TODO: the actual scattering depends on the incident angle: more thickness
        # means more scattering.  also more oblique angles internally lead to reflection
        # at the far side, eventually absorption.
        # i could just cut it off at the total internal reflection limit.

        photons.remove(self._absorption)

        size = np.int32(photons.size())  # TODO eliminate this
        phi = scattering.get_scattering_phi(size)
        theta = scattering.get_scattering_theta(self._g, size)
        block_size = 1024  # max
        grid_size = int(math.ceil(size / block_size))
        scattering.scatter(
            (grid_size,),
            (block_size,),
            (photons.ez_x, photons.ez_y, photons.ez_z, theta, phi, size),
        )


#        cp.cuda.Device().synchronize()


class ColorFilter:
    """transmits some of the photons depending on their wavelength."""

    def __init__(self):
        pass

    def transfer(self, photons: Photons) -> None:
        pass


def propagate_to_reflector(photons, location):
    # TODO: make a raw kernel for this whole function
    # first get rid of the ones not heading that way
    cp.logical_and(photons.alive, photons.ez_z > 0, out=photons.alive)

    location_v = cp.full(photons.size(), location, dtype=np.float32)
    distance_z = location_v - photons.r_z
    photons.r_x = photons.r_x + distance_z * photons.ez_x / photons.ez_z
    photons.r_y = photons.r_y + distance_z * photons.ez_y / photons.ez_z
    photons.r_z = location_v


def propagate_to_camera(photons, location):
    # prune photons heading the wrong way
    cp.logical_and(photons.alive, photons.ez_z < 0, out=photons.alive)

    location_v = cp.full(photons.size(), location, dtype=np.float32)
    distance_z = location_v - photons.r_z
    photons.r_x = photons.r_x + distance_z * photons.ez_x / photons.ez_z
    photons.r_y = photons.r_y + distance_z * photons.ez_y / photons.ez_z
    photons.r_z = location_v
