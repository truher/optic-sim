from typing import Tuple
import cupy as cp  # type: ignore
from cupyx import jit  # type: ignore
import math
import numpy as np
import scipy.constants
import stats_cuda


# SCATTERING THETA


@jit.rawkernel(device=True)
def _hanley(g: np.float32, r: np.float32) -> np.float32:
    """r: random[0,2g)"""
    # TODO: do random inside
    temp = (1 - g * g) / (1 - g + r)
    cost = (1 + g * g - temp * temp) / (2 * g)
    return cp.arccos(cost)


@jit.rawkernel()
def _hanley_loop(random_inout: cp.ndarray, g: np.float32, size: np.int32) -> None:
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        random_inout[i] = _hanley(g, random_inout[i])


def get_scattering_theta(g: np.float32, size: np.int32) -> cp.ndarray:
    random_input = cp.random.uniform(np.float32(0), np.float32(2.0 * g), np.int32(size), dtype=np.float32)
    _hanley_loop((128,), (1024,), (random_input, np.float32(g), np.int32(size)))
    return random_input


# SCATTERING PHI


def get_scattering_phi(size: np.int32) -> cp.ndarray:
    return cp.random.uniform(0, 2 * np.pi, size, dtype=np.float32)


# SCATTERING


@jit.rawkernel(device=True)
def any_perpendicular(
    vx: np.float32, vy: np.float32, vz: np.float32
) -> Tuple[np.float32, np.float32, np.float32]:
    if vz < vx:
        return (vy, -vx, np.float32(0.0))
    else:
        return  (np.float32(0.0), -vz, vy)


@jit.rawkernel(device=True)
def normalize(
    x: np.float32, y: np.float32, z: np.float32
) -> Tuple[np.float32, np.float32, np.float32]:
    n = cp.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


@jit.rawkernel(device=True)
def unitary_perpendicular(
    vx: np.float32, vy: np.float32, vz: np.float32
) -> Tuple[np.float32, np.float32, np.float32]:
    (ux, uy, uz) = any_perpendicular(vx, vy, vz)
    return normalize(ux, uy, uz)


@jit.rawkernel(device=True)
def do_rotation(
    X: np.float32,
    Y: np.float32,
    Z: np.float32,
    ux: np.float32,
    uy: np.float32,
    uz: np.float32,
    theta: np.float32,
) -> Tuple[np.float32, np.float32, np.float32]:
    """Rotate v around u."""
    cost = cp.cos(theta)
    sint = cp.sin(theta)
    one_cost = 1 - cost

    x = (
        (cost + ux * ux * one_cost) * X
        + (ux * uy * one_cost - uz * sint) * Y
        + (ux * uz * one_cost + uy * sint) * Z
    )
    y = (
        (uy * ux * one_cost + uz * sint) * X
        + (cost + uy * uy * one_cost) * Y
        + (uy * uz * one_cost - ux * sint) * Z
    )
    z = (
        (uz * ux * one_cost - uy * sint) * X
        + (uz * uy * one_cost + ux * sint) * Y
        + (cost + uz * uz * one_cost) * Z
    )
 
    return (x, y, z)


@jit.rawkernel()
def scatter(
    vx: cp.ndarray,
    vy: cp.ndarray,
    vz: cp.ndarray,
    theta: cp.ndarray,
    phi: cp.ndarray,
    size: np.int32,
) -> None:
    """Mutate v according to the angles in theta and phi.

    TODO: do the random part inside, to avoid allocating those huge vectors."""
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        (ux, uy, uz) = unitary_perpendicular(vx[i], vy[i], vz[i])
        # first rotate the perpendicular around the photon axis
        (ux, uy, uz) = do_rotation(ux, uy, uz, vx[i], vy[i], vz[i], phi[i])
        # then rotate the photon around that perpendicular
        (vx[i], vy[i], vz[i]) = do_rotation(vx[i], vy[i], vz[i], ux, uy, uz, theta[i])


def get_phi(y: cp.ndarray, x: cp.ndarray) -> cp.ndarray:
    return cp.arctan2(y, x)


def get_theta(z: cp.ndarray) -> cp.ndarray:
    return cp.arccos(z)


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
        self.wavelength_nm = None # [int16]
        self.photons_per_bundle = 0 # should be like 1e7 ish?
        self.duration_s = 0 # for computing power

    def size(self):
        return np.int32(self.r_x.size)

    def count_alive(self):
        return cp.count_nonzero(self.alive)

    def decimate(self, p):
        """Remove photons with probability p."""
        # TODO: do this without allocating a giant vector
        if p < 0.001:
            return
        cp.logical_and(self.alive, cp.random.random(self.size()) > p, out=self.alive)

    def prune_outliers(self, size):
        """ set out-of-bounds photons to dead """
        cp.logical_and(self.alive, self.r_x >= -size/2, out=self.alive)
        cp.logical_and(self.alive, self.r_x <= size/2, out=self.alive)
        cp.logical_and(self.alive, self.r_y >= -size/2, out=self.alive)
        cp.logical_and(self.alive, self.r_y <= size/2, out=self.alive)

    @staticmethod
    def compress_dead(alive, x):
        return cp.compress(alive, x, axis=0)

    def debug(self, source_size_m):
        #return
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
        block_size = 64 # 0.45
        # choose extra to compensate for deadness
        # allow approximate return count
        grid_size = int(math.ceil(16 / alive_ratio))
        selection_size = min(size, grid_size * block_size)
        scale = np.int32(size // selection_size)

        #mempool = cp.get_default_memory_pool()
        #mempool.free_all_blocks()

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
                scale
            ),
        )

        position_3d = Photons.compress_dead(alive_selected, position_3d)
        direction_3d = Photons.compress_dead(alive_selected, direction_3d)
        return (position_3d.get(), direction_3d.get())

    @staticmethod
    @cp.fuse()
    def energy_j_kernel(wavelength_nm, photons_per_bundle):
        wavelength_m = wavelength_nm * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        energy_per_bundle_j = energy_per_photon_j * photons_per_bundle
        return cp.sum(energy_per_bundle_j)
    

    def energy_j(self) -> float:
        """Energy of this photon bundle."""
        return Photons.energy_j_kernel(self.wavelength_nm, self.photons_per_bundle)

    def power_w(self) -> float:
        return self.energy_j() / self.duration_s


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


class MonochromaticLambertianSource(Source):
    def __init__(self, width_m: float, height_m: float,
                 wavelength_nm: int, photons_per_bundle: float,
                 duration_s: float):
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
        photons.ez_z = cp.arccos(cp.random.uniform(-1, 1, bundles, dtype=np.float32)) / 2
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
        absorption = np.float32(0.1) # polished metal inside

        r_x_box_widths = self._height * photons.ez_x / (photons.ez_z * self._size)
        r_y_box_widths = self._height * photons.ez_y / (photons.ez_z * self._size)

        reflection_count_x = cp.abs(cp.round(r_x_box_widths))
        reflection_count_y = cp.abs(cp.round(r_y_box_widths))

        photons.r_x = self._size * (2 * cp.abs(cp.mod(r_x_box_widths - 0.5, 2) - 1) - 1) / 2
        photons.r_y = self._size * (2 * cp.abs(cp.mod(r_y_box_widths - 0.5, 2) - 1) - 1) / 2
        photons.r_z = cp.full(photons.size(), self._height, dtype=np.float32)

        cp.multiply(photons.ez_x, (1 - 2 * cp.mod(reflection_count_x, 2)), out=photons.ez_x)
        cp.multiply(photons.ez_y, (1 - 2 * cp.mod(reflection_count_y, 2)), out=photons.ez_y)

        total_reflection_count = reflection_count_x + reflection_count_y
        photon_survival = cp.power((1-absorption), total_reflection_count)
        photons.alive = cp.logical_and(
                            photons.alive,
                            cp.logical_and(
                                cp.less(cp.random.random(photons.size()), photon_survival),
                                cp.greater(photons.ez_z, 0)), out=photons.alive)

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
        """ Adjust propagation direction."""
        # TODO: the actual scattering depends on the incident angle: more thickness
        # means more scattering.  also more oblique angles internally lead to reflection
        # at the far side, eventually absorption.
        # i could just cut it off at the total internal reflection limit.

        photons.decimate(self._absorption)
        
        size = np.int32(photons.size()) # TODO eliminate this
        phi = get_scattering_phi(size)
        theta = get_scattering_theta(self._g, size)
        block_size = 1024 # max
        grid_size = int(math.ceil(size/block_size))
        scatter((grid_size,), (block_size,),
                (photons.ez_x, photons.ez_y, photons.ez_z, theta, phi, size))
#        cp.cuda.Device().synchronize()

class ColorFilter:
    """ transmits some of the photons depending on their wavelength."""
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
