# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, protected-access, too-many-branches, too-few-public-methods
""" A simple photon simulator.

This is cribbed from https://github.com/DCC-Lab/PyTissueOptics.
"""
# black formatted

from __future__ import annotations
import copy
import math
import random
from typing import List, Tuple
import numpy as np


class Vector:
    """Supports rotation."""

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self._x = x
        self._y = y
        self._z = z

    def any_perpendicular(self) -> Vector:
        """make er for ez"""
        if self._z < self._x:
            return Vector(self._y, -self._x, 0)
        return Vector(0, -self._z, self._y)

    def normalize(self) -> Vector:
        ux = self._x
        uy = self._y
        uz = self._z

        norm = ux * ux + uy * uy + uz * uz
        if norm != 0:
            invLength = norm ** (-0.5)
            self._x *= invLength
            self._y *= invLength
            self._z *= invLength
        else:
            raise ValueError("You cannot normalize the null vector")

        return self

    def rotateAround(self, u: Vector, theta: float) -> Vector:
        u.normalize()

        cost = np.cos(theta)
        sint = np.sin(theta)
        one_cost = 1 - cost

        ux = u._x
        uy = u._y
        uz = u._z

        X = self._x
        Y = self._y
        Z = self._z

        self._x = (
            (cost + ux * ux * one_cost) * X
            + (ux * uy * one_cost - uz * sint) * Y
            + (ux * uz * one_cost + uy * sint) * Z
        )
        self._y = (
            (uy * ux * one_cost + uz * sint) * X
            + (cost + uy * uy * one_cost) * Y
            + (uy * uz * one_cost - ux * sint) * Z
        )
        self._z = (
            (uz * ux * one_cost - uy * sint) * X
            + (uz * uy * one_cost + ux * sint) * Y
            + (cost + uz * uz * one_cost) * Z
        )
        return self

    def __str__(self) -> str:
        return f"Vector: x={self._x:.4f} y={self._y:.4f} z={self._z:.4f})"


class UnitVector(Vector):
    def __init__(self, x: float, y: float, z: float):
        Vector.__init__(self, x, y, z)

    @staticmethod
    def spherical(theta: float, phi: float) -> UnitVector:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return UnitVector(x, y, z)

    @staticmethod
    def cartesian(x: float, y: float, z: float) -> UnitVector:
        v = Vector(x, y, z).normalize()
        return UnitVector(v._x, v._y, v._z)

    def phi(self) -> float:
        """Returns [-pi, pi]."""
        # use np.arctan2 in vectorized version
        return math.atan2(self._y, self._x)

    def theta(self) -> float:
        # use np.arccos in vectorized version
        return math.acos(self._z)


class Photon:
    def __init__(self, R: Vector, EZ: UnitVector):
        self.r = R  # local coordinate position
        self.ez = EZ  # propagation direction
        self.er = (
            self.ez.any_perpendicular()
        )  # perpendicular to direction, used for rotation
        self.weight = 1  # zero means ignore.

    @staticmethod
    def spherical(x: float, y: float, z: float, theta: float, phi: float) -> Photon:
        # local coordinate position
        r = Vector(x, y, z)
        # propagation direction
        ez = UnitVector.spherical(theta, phi)
        return Photon(r, ez)

    def scatterBy(self, theta: float, phi: float) -> None:
        """Mutates this photon."""
        self.er.rotateAround(self.ez, phi)
        self.ez.rotateAround(self.er, theta)

    @property
    def isAlive(self) -> bool:
        return self.weight > 0

    @staticmethod
    def sample(photons: List[Photon], N: int) -> List[Photon]:
        return copy.deepcopy(random.sample(photons, min(N, len(photons))))

    @staticmethod
    def countAlive(photons: List[Photon]) -> int:
        count = 0
        for p in photons:
            if p.isAlive:
                count += 1
        return count

    @staticmethod
    def selectAlive(photons: List[Photon]) -> List[Photon]:
        alive = []
        for p in photons:
            if p.isAlive:
                alive.append(p)
        return alive

    def __str__(self) -> str:
        return (
            f"Photon: x={self.r._x:.2f} y={self.r._y:.2f} z={self.r._z:.2f} "
            f"theta={self.ez.theta():.2f} phi={self.ez.phi():.2f}"
        )


class Source:
    def newPhoton(self) -> Photon:
        raise NotImplementedError()

    def make_photons(self, photon_count: int) -> List[Photon]:
        photons = []
        for pi in range(photon_count):  # pylint: disable=unused-variable
            p = self.newPhoton()
            photons.append(p)
        return photons


class PencilSource(Source):
    """Zero area zero divergence."""

    def newPhoton(self) -> Photon:
        return Photon.spherical(0, 0, 0, 0, 0)


class HalfSphereSource(Source):
    """Isotropic intensity in the upper half-sphere."""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def newPhoton(self) -> Photon:
        x = self._width * (np.random.random() - 0.5)
        y = self._height * (np.random.random() - 0.5)
        phi = np.random.random() * 2 * np.pi
        theta = np.arccos(np.random.random())
        p = Photon.spherical(x, y, 0, 0, 0)
        p.scatterBy(theta, phi)
        return p

    def __str__(self) -> str:
        return f"HalfSphereSource: width={self._width:.2f} height={self._height:.2f}"


class LambertianSource(Source):
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def newPhoton(self) -> Photon:
        x = self._width * (np.random.random() - 0.5)
        y = self._height * (np.random.random() - 0.5)
        phi = np.random.random() * 2 * np.pi
        theta = np.arccos(2 * np.random.random() - 1) / 2
        p: Photon = Photon.spherical(x, y, 0, theta, phi)
        return p


class RealLEDSource(Source):
    """TODO: use the published data from Cree."""

    def __init__(self, width: float, height: float):
        self._width = width  # microns
        self._height = height  # microns

    def newPhoton(self) -> Photon:
        x = self._width * (np.random.random() - 0.5)
        y = self._height * (np.random.random() - 0.5)
        phi = np.random.random() * 2 * np.pi
        theta = np.arccos(2 * np.random.random() - 1) / 2
        p: Photon = Photon.spherical(x, y, 0, theta, phi)
        return p


class Diffuser:
    """Something that changes photon direction.

    Now mutates rather than copying.
    Examples: diffuser, retroreflector.
    """

    def __init__(self, g: float, absorption: float):
        """
        g: Henyey and Greenstein scattering parameter.
            0 is iso, 1 is no scattering, -1 is reflection.
        absorption: mostly useful for the diffuser
        """
        self._g = g
        self._absorption = absorption

    def getScatteringAngles(self) -> Tuple[float, float]:
        """Henyey and Greenstein scattering. This is from material.py."""
        phi = np.random.random() * 2 * np.pi
        temp = (1 - self._g * self._g) / (
            1 - self._g + 2 * self._g * np.random.random()
        )
        cost = (1 + self._g * self._g - temp * temp) / (2 * self._g)
        return np.arccos(cost), phi

    def diffuse(self, photons: List[Photon]) -> None:
        """Adjust propagation direction."""
        for p in photons:
            if not p.isAlive:
                continue
            if np.random.random() < self._absorption:
                p.weight = 0
                continue
            # use the actual diffuser angular distribution here?
            theta_scatter, phi_scatter = self.getScatteringAngles()
            p.scatterBy(theta_scatter, phi_scatter)


class Lightbox:
    """Represents the box between the source and diffuser.

    Sides are somewhat reflective.
    Now mutates rather than copying.
    """

    def __init__(self, height: float, size: float):
        """
        height: top of the box above the source
        size: full length or width, box is square.
        """
        self._height = height
        self._size = size

    def propagate(self, photons: List[Photon]) -> None:
        """Propagate (mutate) photons through the light box to the top."""

        absorption = 0.1  # polished metal inside
        for p in photons:
            # photon starts at p.r(x,y).  assume p.r(z) is zero (todo: fix that)
            if not p.isAlive:
                continue
            if p.ez._z < 0:
                p.weight = 0
                continue  # this shouldn't happen

            p.r._x = p.r._x + self._height * p.ez._x / p.ez._z
            p.r._y = p.r._y + self._height * p.ez._y / p.ez._z
            p.r._z = self._height

            done_reflecting = False
            while not done_reflecting:
                if p.r._x < -self._size / 2:
                    p.r._x = -self._size - p.r._x
                    p.ez._x *= -1
                    p.er._x *= -1
                    if np.random.random() < absorption:
                        break
                elif p.r._x > self._size / 2:
                    p.r._x = self._size - p.r._x
                    p.ez._x *= -1
                    p.er._x *= -1
                    if np.random.random() < absorption:
                        break
                elif p.r._y < -self._size / 2:
                    p.r._y = -self._size - p.r._y
                    p.ez._y *= -1
                    p.er._y *= -1
                    if np.random.random() < absorption:
                        break
                elif p.r._y > self._size / 2:
                    p.r._y = self._size - p.r._y
                    p.ez._y *= -1
                    p.er._y *= -1
                    if np.random.random() < absorption:
                        break
                if (
                    p.r._x >= -self._size / 2
                    and p.r._x <= self._size / 2
                    and p.r._y >= -self._size / 2
                    and p.r._y <= self._size / 2
                ):
                    done_reflecting = True
            if not done_reflecting:  # i.e. early break
                p.weight = 0


def propagateToReflector(photons: List[Photon], location: float, size: float) -> None:
    """just push them up there.
      location: z dimension of the reflector.  it's *far*, if the unit is 100 microns,
      and the reflector is 1-10 meters away, that's 10000-100000.
      size: say 10cm on a side, 1000 units.
    there's no box.
    todo: make this a class, combine it somehow with the other one above
    """
    for p in photons:
        if not p.isAlive:
            continue
        if p.ez._z < 0:
            p.weight = 0
            continue
        distance_z = location - p.r._z
        p.r._x = p.r._x + distance_z * p.ez._x / p.ez._z
        p.r._y = p.r._y + distance_z * p.ez._y / p.ez._z
        p.r._z = location
        if (
            p.r._x < -size / 2
            or p.r._x > size / 2
            or p.r._y < -size / 2
            or p.r._y > size / 2
        ):
            p.weight = 0


def propagateToCamera(photons: List[Photon], location: float) -> None:
    """back down to the camera.

    for now it's just the camera plane. TODO: add a counter.
    """
    for p in photons:
        if not p.isAlive:
            continue
        if p.ez._z > 0:
            p.weight = 0
            continue
        distance_z = location - p.r._z  # negative number in this case
        p.r._x = p.r._x + distance_z * p.ez._x / p.ez._z
        p.r._y = p.r._y + distance_z * p.ez._y / p.ez._z
        p.r._z = location
