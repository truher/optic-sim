# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, protected-access, too-many-branches, too-few-public-methods, no-member, no-value-for-parameter, too-many-arguments
""" Photon simulation using Cuda.

photon parameters are columnar, individual photons are "rows"

"""

# SETUP

import warnings

from typing import Tuple
import cupy as cp  # type: ignore
import math
import numpy as np
from cupyx import jit  # type: ignore
from stats_cuda import *

warnings.filterwarnings("ignore", category=FutureWarning)
print(f"CuPy version {cp.__version__}")


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

    def size(self):
        return np.int32(self.r_x.size)

    def decimate(self, p):
        """Remove photons with probability p."""
        # TODO: do this without allocating a giant vector
        if p < 0.001:
            return
        self.alive = cp.random.random(self.size()) > p
        self.prune()

    def prune(self):
        """ remove dead photons """
        self.r_x = cp.compress(self.alive, self.r_x)
        self.r_y = cp.compress(self.alive, self.r_y)
        self.r_z = cp.compress(self.alive, self.r_z)
        self.ez_x = cp.compress(self.alive, self.ez_x)
        self.ez_y = cp.compress(self.alive, self.ez_y)
        self.ez_z = cp.compress(self.alive, self.ez_z)
        self.alive = cp.compress(self.alive, self.alive)

    def sample(self):
        """Take every N-th for plotting 1024.  Returns
        a type the plotter likes, which is two (N,3) vectors"""
        size = self.size()  # ~35ns
        grid_size = 32
        block_size = 32
        selection_size = min(size, grid_size * block_size)
        scale = np.int32(size // selection_size)

        p = cp.zeros((selection_size, 3), dtype=np.float32)
        d = cp.zeros((selection_size, 3), dtype=np.float32)

        select_and_stack(
            (grid_size,),
            (block_size,),
            (
                self.r_x,
                self.r_y,
                self.r_z,
                self.ez_x,
                self.ez_y,
                self.ez_z,
                p,
                d,
                selection_size,
                scale
            ),
        )

        return (p, d)


class Source:
    def make_photons(self, size: np.int32) -> Photons:
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


class LambertianSource(Source):
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def make_photons(self, size: np.int32) -> Photons:
        photons = Photons()
        photons.r_x = cp.random.uniform(
            -0.5 * self._width, 0.5 * self._width, size, dtype=np.float32
        )
        photons.r_y = cp.random.uniform(
            -0.5 * self._height, 0.5 * self._height, size, dtype=np.float32
        )
        photons.r_z = cp.full(size, 0.0, dtype=np.float32)
        # phi, reused as x
        photons.ez_x = cp.random.uniform(0, 2 * np.pi, size, dtype=np.float32)
        photons.ez_y = cp.empty(size, dtype=np.float32)
        # theta, reused as z
        photons.ez_z = cp.arccos(cp.random.uniform(-1, 1, size, dtype=np.float32)) / 2
        spherical_to_cartesian_raw(
            (128,), (1024,), (photons.ez_z, photons.ez_x, photons.ez_y, size)
        )
        photons.alive = cp.ones(size, dtype=bool)
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


    _PROPAGATE_KERNEL = cp.ElementwiseKernel(
        in_params = 'raw uint64 seed, raw float32 absorption, raw float32 height, raw float32 size',
        out_params = 'float32 r_x, float32 r_y, float32 r_z, float32 ez_x, float32 ez_y, float32 ez_z, bool alive',
        preamble = """
        #include <curand.h>
        #include <curand_kernel.h>""",
        loop_prep = """
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        curandState state;
        curand_init(seed, idx, 0, &state);""",
        operation = """
        if (!alive) continue;
        if (ez_z < 0) {
            alive = false;
            continue;
        }
        r_x = r_x + height * ez_x / ez_z;
        r_y = r_y + height * ez_y / ez_z;
        r_z = height;
        
        bool done_reflecting = false;
        while (!done_reflecting) {
            if (r_x < -size / 2) {
                r_x = -size - r_x;
                ez_x *= -1;
                // er_x *= -1; // no more persistent perpendicular
                if (curand_uniform(&state) < absorption) break;
            } else if (r_x > size / 2) {
                r_x = size - r_x;
                ez_x *= -1;
                // er_x *= -1;
                if (curand_uniform(&state) < absorption) break;
            } else if (r_y < -size / 2) {
                r_y = -size - r_y;
                ez_y *= -1;
                // er_y *= -1;
                if (curand_uniform(&state) < absorption) break;
            } else if (r_y > size / 2) {
                r_y = size - r_y;
                ez_y *= -1;
                // er_y *= -1;
                if (curand_uniform(&state) < absorption) break;
            }
            if (r_x >= -size / 2 && r_x <= size / 2 && r_y >= -size / 2 && r_y <= size / 2)
                done_reflecting = true;
        }
        if (!done_reflecting) {
            alive = false;
        }
        """,
        no_return = True)

    def propagate(self, photons: Photons) -> None:
        """Propagate (mutate) photons through the light box to the top.
        Deletes absorbed photons.
        TODO: make this faster, it takes way too long; maybe a real raw kernel with a smaller grid?
        """
        seed = np.random.default_rng().integers(1, np.iinfo(np.uint64).max, dtype=np.uint64)
        absorption = np.float32(0.1) # polished metal inside
        Lightbox._PROPAGATE_KERNEL(seed, absorption, self._height, self._size,
                                   photons.r_x, photons.r_y, photons.r_z,
                                   photons.ez_x, photons.ez_y, photons.ez_z,
                                   photons.alive,
                                   block_size = 32) # smaller block = less tail latency?  .. nope.
        photons.prune() # remove the absorbed photons


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

    def diffuse(self, photons: Photons) -> None:
        """ Adjust propagation direction."""

        photons.decimate(self._absorption)
        
        size = np.int32(photons.size()) # TODO eliminate this
        phi = get_scattering_phi(size)
        theta = get_scattering_theta(self._g, size)
        block_size = 1024 # max
        grid_size = int(math.ceil(size/block_size))
        scatter((grid_size,), (block_size,), (photons.ez_x, photons.ez_y, photons.ez_z, theta, phi, size))


def prune_outliers(photons, size):
    cp.logical_and(photons.alive, photons.r_x >= -size/2, out=photons.alive)
    cp.logical_and(photons.alive, photons.r_x <= size/2, out=photons.alive)
    cp.logical_and(photons.alive, photons.r_y >= -size/2, out=photons.alive)
    cp.logical_and(photons.alive, photons.r_y <= size/2, out=photons.alive)
    photons.prune()

def propagate_to_reflector(photons, location, size):
    """ size: reflector size """
    # TODO: make a raw kernel for this whole function
    # first get rid of the ones not heading that way
    photons.alive = photons.ez_z > 0
    photons.prune()

    location_v = cp.full(photons.size(), location, dtype=np.float32)
    distance_z = location_v - photons.r_z
    photons.r_x = photons.r_x + distance_z * photons.ez_x / photons.ez_z
    photons.r_y = photons.r_y + distance_z * photons.ez_y / photons.ez_z
    photons.r_z = location_v

def propagate_to_camera(photons, location):
    # prune photons heading the wrong way
    photons.alive = photons.ez_z < 0
    photons.prune()

    location_v = cp.full(photons.size(), location, dtype=np.float32)
    distance_z = location_v - photons.r_z
    photons.r_x = photons.r_x + distance_z * photons.ez_x / photons.ez_z
    photons.r_y = photons.r_y + distance_z * photons.ez_y / photons.ez_z
    photons.r_z = location_v

class ResultStage:
    def __init__(self):
        self._photons_size = 0

class SimulationResult:
    """Additive metrics produced from N waves of simulation."""
    def __init__(self):
        self._source_stage = ResultStage()

class Simulator:
    """The specific geometry etc of this simulation.
    Runs N waves of M photon bundles each.  Each bundle represents a set of photons,
    each photon has a different wavelength (energy).
    """
    def __init__(self, results, waves, bundles, bundle_size):
        """waves: number of full iterations to run
           bundles: number of "Photons" groups (bundles) to run
           bundle_size: photons per bundle, for accounting energy etc
           I think this will all change, specifying something like emitter power or whatever
           but here it is for now.
        """
        self._results = results
        self._waves = waves
        self._bundles = bundles
        self._bundle_size = bundle_size

    def run(self):
        """Run all the waves."""
        # first make some photons
        source_size = np.float32(10)
        photons = LambertianSource(source_size, source_size).make_photons(self._bundles)
        self._results._source_stage._photons_size = photons.size()
        #print(f"LED emitted photons: {photons.size()}")
        #pass

class Study:
    """Sweep some parameters while measuring output."""
    def __init__(self):
        pass
