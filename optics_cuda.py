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
    print("G")
    print(g)
    print(type(g))
    random_input = cp.random.uniform(np.float32(0), np.float32(2.0 * g), np.int32(size), dtype=np.float32)
    print(random_input)
    _hanley_loop((128,), (1024,), (random_input, np.float32(g), np.int32(size)))
    return random_input


# SCATTERING PHI


def get_scattering_phi(size: np.int32) -> cp.ndarray:
    return cp.random.uniform(0, 2 * np.pi, size, dtype=np.float32)


# SCATTERING


#@jit.rawkernel(device=True)
#def any_perpendicular(
#    vx: np.float32, vy: np.float32, vz: np.float32
#) -> Tuple[np.float32, np.float32, np.float32]:
#    if vz < vx:
#        (rx, ry, rz) = (vy, -vx, np.float32(0.0))
#    else:
#        (rx, ry, rz) = (np.float32(0.0), -vz, vy)
#    return np.zeros(3, dtype=np.float32)
#    #return np.array([rx, ry, rz])


#@jit.rawkernel(device=True)
#def normalize(
#    x: np.float32, y: np.float32, z: np.float32
#) -> Tuple[np.float32, np.float32, np.float32]:
#    n = cp.sqrt(x * x + y * y + z * z)
#    return (x / n, y / n, z / n)


#@jit.rawkernel(device=True)
#def unitary_perpendicular(
#    vx: np.float32, vy: np.float32, vz: np.float32
#) -> Tuple[np.float32, np.float32, np.float32]:
## WTF
#
##    (ux, uy, uz) = any_perpendicular(vx, vy, vz)
#
#    if vz < vx:
#        (ux, uy, uz) = (vy, -vx, np.float32(0.0))
#    else:
#        (ux, uy, uz) = (np.float32(0.0), -vz, vy)
#    n = cp.sqrt(ux * ux + uy * uy + uz * uz)
#    return (ux / n, uy / n, uz / n)
#
##    # this works (ux, uy, uz) = (vx, vy, vz)
##    return normalize(ux, uy, uz)


#@jit.rawkernel(device=True)
#def do_rotation(
#    X: np.float32,
#    Y: np.float32,
#    Z: np.float32,
#    ux: np.float32,
#    uy: np.float32,
#    uz: np.float32,
#    theta: np.float32,
#) -> Tuple[np.float32, np.float32, np.float32]:
#    """Rotate v around u."""
#    cost = cp.cos(theta)
#    sint = cp.sin(theta)
#    one_cost = 1 - cost
#
#    x = (
#        (cost + ux * ux * one_cost) * X
#        + (ux * uy * one_cost - uz * sint) * Y
#        + (ux * uz * one_cost + uy * sint) * Z
#    )
#    y = (
#        (uy * ux * one_cost + uz * sint) * X
#        + (cost + uy * uy * one_cost) * Y
#        + (uy * uz * one_cost - ux * sint) * Z
#    )
#    z = (
#        (uz * ux * one_cost - uy * sint) * X
#        + (uz * uy * one_cost + ux * sint) * Y
#        + (cost + uz * uz * one_cost) * Z
#    )
#
#    return (x, y, z)

# this is to avoid returning a tuple
@jit.rawkernel(device=True)
def rotate_x(X, Y, Z, ux, uy, uz, cost, sint, one_cost) -> np.float32:
    return (
        (cost + ux * ux * one_cost) * X
        + (ux * uy * one_cost - uz * sint) * Y
        + (ux * uz * one_cost + uy * sint) * Z
    )
@jit.rawkernel(device=True)
def rotate_y(X, Y, Z, ux, uy, uz, cost, sint, one_cost) -> np.float32:
    return (
        (uy * ux * one_cost + uz * sint) * X
        + (cost + uy * uy * one_cost) * Y
        + (uy * uz * one_cost - ux * sint) * Z
    )
@jit.rawkernel(device=True)
def rotate_z(X, Y, Z, ux, uy, uz, cost, sint, one_cost) -> np.float32:
    return (
        (uz * ux * one_cost - uy * sint) * X
        + (uz * uy * one_cost + ux * sint) * Y
        + (cost + uz * uz * one_cost) * Z
    )


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

# wtf is going on here?
# this is the test code
        # this works (vx[i], vy[i], vz[i]) = (vx[i], vy[i], vz[i])
        #(vx[i], vy[i], vz[i]) = unitary_perpendicular(vx[i], vy[i], vz[i])

        # make "any" perpendicular
        if vz[i] < vx[i]:
            (ux, uy, uz) = (vy[i], -vx[i], np.float32(0.0))
        else:
            (ux, uy, uz) = (np.float32(0.0), -vz[i], vy[i])
        # normalize it
        n = cp.sqrt(ux * ux + uy * uy + uz * uz)

        (ux, uy, uz) = (ux / n, uy / n, uz / n)

# this is the real code
#        (ux, uy, uz) = unitary_perpendicular(vx[i], vy[i], vz[i])
#
#        # first rotate the perpendicular around the photon axis
        cost = cp.cos(phi[i])
        sint = cp.sin(phi[i])
        one_cost = 1 - cost
        tx = rotate_x(ux, uy, uz, vx[i], vy[i], vz[i], cost, sint, one_cost)
        ty = rotate_y(ux, uy, uz, vx[i], vy[i], vz[i], cost, sint, one_cost)
        tz = rotate_z(ux, uy, uz, vx[i], vy[i], vz[i], cost, sint, one_cost)
        (ux, uy, uz) = (tx, ty, tz)
#        (ux, uy, uz) = do_rotation(ux, uy, uz, vx[i], vy[i], vz[i], phi[i])
#
#        # then rotate the photon around that perpendicular
        cost = cp.cos(theta[i])
        sint = cp.sin(theta[i])
        one_cost = 1 - cost
        tx = rotate_x(vx[i], vy[i], vz[i], ux, uy, uz, cost, sint, one_cost)
        ty = rotate_y(vx[i], vy[i], vz[i], ux, uy, uz, cost, sint, one_cost)
        tz = rotate_z(vx[i], vy[i], vz[i], ux, uy, uz, cost, sint, one_cost)
        (vx[i], vy[i], vz[i]) = (tx, ty, tz)
#        (vx[i], vy[i], vz[i]) = do_rotation(vx[i], vy[i], vz[i], ux, uy, uz, theta[i])


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
        selection_size = grid_size * block_size
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

    def make_photons(self, size) -> Photons:
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
        """
        seed = np.random.default_rng().integers(1, np.iinfo(np.uint64).max, dtype=np.uint64)
        absorption = np.float32(0.1) # polished metal inside
        Lightbox._PROPAGATE_KERNEL(seed, absorption, self._height, self._size,
                                   photons.r_x, photons.r_y, photons.r_z,
                                   photons.ez_x, photons.ez_y, photons.ez_z,
                                   photons.alive,
                                   block_size = 32) # smaller block = less tail latency?
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
        print("theta")
        print(theta)
        _diffuse(self, photons, theta, phi, size)

    def _diffuse(self, photons, theta, phi, size) -> None:

        block_size = 1024 # max
        grid_size = int(math.ceil(size/block_size))
        scatter((grid_size,), (block_size,), (photons.ez_x, photons.ez_y, photons.ez_z, theta, phi, size))


