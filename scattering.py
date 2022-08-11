import math
from typing import Tuple
import cupy as cp
import numpy as np
from cupyx import jit
import stats_cuda
from scipy.stats import norm

# SCATTERING THETA


@jit.rawkernel(device=True)
def _hanley(g: np.float32, r: np.float32) -> np.float32:
    """r: random[0,2g)"""
    # TODO: do random inside
    temp = (1 - g * g) / (1 - g + r)
    cost = (1 + g * g - temp * temp) / (2 * g)
    return cp.arccos(cost)

#
@jit.rawkernel()
def _hanley_loop(random_inout: cp.ndarray, g: np.float32, size: np.int32) -> None:
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        random_inout[i] = _hanley(g, random_inout[i])


def get_scattering_theta(g: np.float32, size: np.int32) -> cp.ndarray:
    random_input = cp.random.uniform(
        np.float32(0), np.float32(2.0 * g), np.int32(size), dtype=np.float32
    )
    _hanley_loop((128,), (1024,), (random_input, np.float32(g), np.int32(size)))
    return random_input

class LambertianScattering:
    def __init__(self):
        pass

    def get_scattering_theta(self, size: int) -> cp.ndarray:
        pass

class AcryliteScattering_0d010:
    """ Generate scattering angles that result in intensity distribution
    that matches the Thor shape.  A good match is two gaussians, one one-third
    the height and two times the width of the other, plus a small constant.

    Also matches the Acrylite width, which for 0d010
    is 40 degrees, i.e. half angle of 20 degrees.  They say this is a measurement
    compared to the *input* but i think that's insane, everybody else measures
    the shape of the output independent of the input, so i'll do it that way.
    """
    def __init__(self):
        mu = 0 # mean is normal
        sigma_1_rad = 17.25 * np.pi / 180
        sigma_2_rad = 34.5 * np.pi / 180
        a_1 = 0.75 # peak
        a_2 = 0.25 # tails
        a_3 = 0.008 # Thor data shows this
        a_1_actually = sigma_1_rad * np.sqrt(2 * np.pi) * a_1
        a_2_actually = sigma_2_rad * np.sqrt(2 * np.pi) * a_2

        # make a distribution
        self.pdf_x_theta_rad = cp.linspace(0, np.pi/2, 512)

        gaussian_1 = cp.array(
            a_1_actually * norm.pdf(self.pdf_x_theta_rad.get(), scale = sigma_1_rad))
        gaussian_2 = cp.array(
            a_2_actually * norm.pdf(self.pdf_x_theta_rad.get(), scale = sigma_2_rad))
        # this is the target intensity distribution.
        intensity_distribution = ((gaussian_1 + gaussian_2) * (1-a_3) + a_3)
        sin_term = cp.sin(self.pdf_x_theta_rad)
        # TODO figure out the cos^5 schlick thing
        schlick_term = 1 - (1 - cp.cos(self.pdf_x_theta_rad)) ** 5
        
        self.angle_distribution = intensity_distribution * sin_term * schlick_term

    def get_scattering_theta(self, size: int) -> cp.ndarray:
        """Does not account for absorption.  Please remove TK% of the rows returned."""
        return stats_cuda.sample_pdf(size, self.pdf_x_theta_rad, self.angle_distribution)

class AcryliteScattering_wd008:
    def __init__(self):

        # parameters from Thor, Acrylite etc
        # matches Thor shape, Acrylite FWHM of 40 degrees
        mu = 0 # mean is normal
        ##############
        # these work for the 20 degree 0d010df material
        ##############
        #sigma_1_rad = 17.25 * np.pi / 180
        #sigma_2_rad = 34.5 * np.pi / 180
        # this is for wd008



#        sigma_1_rad = 37 * np.pi / 180
#        sigma_2_rad = 74 * np.pi / 180
#        a_1 = 0.75 # peak
#        a_2 = 0.25 # tails
#        a_3 = 0.008 # Thor data shows this
#        a_1_actually = sigma_1_rad * np.sqrt(2 * np.pi) * a_1
#        a_2_actually = sigma_2_rad * np.sqrt(2 * np.pi) * a_2
#
#        # make a distribution
#        bins = 256
#        theta_rad = np.linspace(0, np.pi/2, bins)
#
#        gaussian_1 = cp.array(a_1_actually * norm.pdf(theta_rad, scale = sigma_1_rad))
#        gaussian_2 = cp.array(a_2_actually * norm.pdf(theta_rad, scale = sigma_2_rad))
#        self.cp_theta_rad = cp.array(theta_rad)
#        self.gaussian_sum = ((gaussian_1 + gaussian_2) * (1-a_3) + a_3)


#    def get_scattering_theta_old(self, size):
#        """Does not account for absorption.  Please remove 16% of the rows returned."""
########
# if i use this again, make it one sided; i think this yields negative numbers
# half the time. :-(
#        return stats_cuda.generate_rand_from_pdf(size, self.gaussian_sum, self.cp_theta_rad)

    def get_scattering_theta(self, size):
        """Lambertian"""
        #return cp.full(size, 7*np.pi/16, dtype=np.float32)
        return cp.arcsin(cp.random.uniform(0,1,size,dtype=np.float32))
        #return cp.arccos(cp.random.uniform(0,1,size,dtype=np.float32))
    


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
        return (np.float32(0.0), -vz, vy)


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

# oh i can just transform without all these multiplications

#     [0, 0, 1]
# m = |0, 1, 0|
#     [-1, 0, 0]

@jit.rawkernel()
def avoid_pole(
    vx: cp.ndarray,
    vy: cp.ndarray,
    vz: cp.ndarray,
    angle: np.int32, # 1 = bend away from pole, -1 = bend back
    size: np.int32,
) -> None:
    """mutate v, rotating it around the y axis, so that the pole ends up
    at xy00, which has no areal singularity for radiance scaling; this is
    also used to rotate back."""

    ux = np.float32(0.0)
    uy = np.float32(1.0)
    uz = np.float32(0.0)
    rot = np.float32(angle * np.pi/2)

    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        (vx[i], vy[i], vz[i]) = do_rotation(vx[i], vy[i], vz[i], ux, uy, uz, rot)

#def do_avoid_pole_old(photons, angle):
#    block_size = 1024
#    size = np.int32(photons.size())
#    grid_size = int(math.ceil(size / block_size))
#    avoid_pole(
#        (grid_size,),
#        (block_size,),
#        (photons.ez_x, photons.ez_y, photons.ez_z, angle, size)
#    )

def do_avoid_pole(photons, angle):
##########
##########
##########
    return
##########
##########
##########
    if angle > 0:
        tmp =  photons.ez_x
        photons.ez_x = photons.ez_z
        photons.ez_z = -tmp
    else:
        tmp =  photons.ez_x
        photons.ez_x = -photons.ez_z
        photons.ez_z = tmp

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
