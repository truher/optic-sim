# pylint: disable=too-many-statements, too-many-arguments, too-many-locals, consider-using-from-import, protected-access
""" Plotting functions"""
# black formatted

from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt  # type: ignore
import mpl_toolkits.mplot3d.art3d as art3d  # type: ignore
import numpy as np
from matplotlib.patches import Rectangle  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from numpy.typing import NDArray
import optics

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def plot_rays(
    batches: List[List[optics.Photon]],
    elev: float,
    azim: float,
    size: float,
    arrow_length: float,
    rectangles: List[List[float]],
) -> None:
    """3d plot a set of photon batches, each batch a different color.

    Also render interesting rectangles.  [xmin,xmax,ymin,ymax,z]
    """
    fig = plt.figure(figsize=[15, 15])
    axes = fig.gca(projection="3d")
    for i, p_batch in enumerate(batches):
        output_len = int(len(p_batch))
        photon_x = np.zeros([output_len])
        photon_y = np.zeros([output_len])
        photon_z = np.zeros([output_len])
        photon_u = np.zeros([output_len])
        photon_v = np.zeros([output_len])
        photon_w = np.zeros([output_len])
        for pidx, photon in enumerate(p_batch):
            if not photon.isAlive:
                continue
            photon_x[pidx] = photon.r._x
            photon_y[pidx] = photon.r._y
            photon_z[pidx] = photon.r._z
            photon_u[pidx] = arrow_length * photon.ez._x
            photon_v[pidx] = arrow_length * photon.ez._y
            photon_w[pidx] = arrow_length * photon.ez._z
        axes.quiver(
            photon_x,
            photon_y,
            photon_z,
            photon_u,
            photon_v,
            photon_w,
            arrow_length_ratio=0.05,
            color=f"C{i%10}",
        )
    axes.set_xlim([-size / 2, size / 2])
    axes.set_ylim([-size / 2, size / 2])
    axes.set_zlim([0, size])
    axes.view_init(elev, azim)
    axes.set_xlabel("X axis")
    axes.set_ylabel("Y axis")
    axes.set_zlabel("Z axis")
    for ridx, rectangle in enumerate(rectangles):
        xmin = rectangle[0]
        xmax = rectangle[1]
        ymin = rectangle[2]
        ymax = rectangle[3]
        location_z = rectangle[4]
        patch = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=5,
            color=f"C{ridx%10}",
        )
        axes.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch, z=location_z, zdir="z")
    plt.show()


def plot_histogram_slices(
    photon_batch: List[optics.Photon],
    suptitle: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    theta_min: float = 0,
    theta_max: float = np.pi,
    phi_min: float = -np.pi,
    phi_max: float = np.pi,
    bins: int = 100,
) -> None:
    """Slice counts by x, y, phi, and theta, also show intensity per steradian."""
    # radiance is *projected* area, so compute that too?
    # radiosity (W/m^2): all directions, per area (slice in x, total in y)
    x_freq: NDArray[np.int32] = np.zeros(bins, dtype=np.int32)

    # radiosity (W/m^2): all directions, per area (slice in y, total in x)
    y_freq: NDArray[np.int32] = np.zeros(bins, dtype=np.int32)

    # this is to find errant photons
    z_freq: NDArray[np.int32] = np.zeros(bins, dtype=np.int32)

    # radiant intensity (W/sr): whole area, per angle (slice in theta, total in phi)
    theta_freq: NDArray[np.int32] = np.zeros(bins, dtype=np.int32)

    # radiant intensity (W/sr): whole area, per angle (slice in phi, total in theta)
    phi_freq: NDArray[np.int32] = np.zeros(bins, dtype=np.int32)

    photons_per_steridian_by_theta: NDArray[np.float64] = np.zeros(
        bins, dtype=np.float64
    )

    x_range = x_max - x_min
    x_bin_width = x_range / bins
    x_bins = x_range * np.arange(bins) / bins + x_min  # units?

    y_range = y_max - y_min
    y_bin_width = y_range / bins
    y_bins = y_range * np.arange(bins) / bins + y_min  # units?

    z_range = z_max - z_min
    z_bin_width = z_range / bins
    z_bins = z_range * np.arange(bins) / bins + z_min

    theta_range = theta_max - theta_min
    theta_bin_width_rad = theta_range / bins
    theta_bin_width_deg = theta_bin_width_rad * 180 / np.pi

    theta_bins_rad = theta_range * np.arange(bins) / bins + theta_min
    theta_bins_deg = theta_bins_rad * 180 / np.pi

    phi_range = phi_max - phi_min
    phi_bin_width_rad = phi_range / bins
    phi_bin_width_deg = phi_bin_width_rad * 180 / np.pi
    phi_bins_deg = (phi_range * np.arange(bins) / bins + phi_min) * 180 / np.pi

    for photon in photon_batch:
        if not photon.isAlive:
            continue
        # "floor" so that it doesn't double-count near zero.
        # figure out the area thing here?
        x_bin = int((photon.r._x - x_min) // x_bin_width)
        if x_bin < 0 or x_bin > bins - 1:
            continue
        x_freq[x_bin] += 1

        y_bin = int((photon.r._y - y_min) // y_bin_width)
        if y_bin < 0 or y_bin > bins - 1:
            continue
        y_freq[y_bin] += 1

        z_bin = int((photon.r._z - z_min) // z_bin_width)
        if z_bin < 0 or z_bin > bins - 1:
            continue
        z_freq[z_bin] += 1

        theta_bin = int((photon.ez.theta() - theta_min) // theta_bin_width_rad)
        if theta_bin < 0 or theta_bin > bins - 1:
            continue
        theta_freq[theta_bin] += 1

        phi_bin = int((photon.ez.phi() - phi_min) // phi_bin_width_rad)
        if phi_bin < 0 or phi_bin > bins - 1:
            continue
        phi_freq[phi_bin] += 1

        # figure out the area thing here?
        photons_per_steridian_by_theta[theta_bin] += 1 / np.sin(photon.ez.theta())

    # the above histograms are counts per bucket.
    # i want counts per steradian.
    # it's axially symmetric so ignore phi
    # the size of a theta bucket is
    #  h = np.cos(left) - np.cos(right)
    #  area = 2 * np.pi * h
    # the bin width is np.pi / 100

    theta_bucket_angle_steradians: NDArray[np.float64] = np.array(
        list(
            map(
                lambda x: (
                    np.cos((theta_range * x + theta_min) / bins)
                    - np.cos((theta_range * (x + 1) + theta_min) / bins)
                ),
                range(bins),
            )
        )
    )
    photons_per_steradian_by_theta = theta_freq / theta_bucket_angle_steradians

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(suptitle)

    axes = plt.subplot(331)
    plt.bar(x=x_bins, height=x_freq, width=x_bin_width)
    axes.set_title("photons per bucket by x")
    axes.set_xlabel("x dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(332)
    plt.bar(x=y_bins, height=y_freq, width=y_bin_width)
    axes.set_title("photons per bucket by y")
    axes.set_xlabel("y dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(333)
    plt.bar(x=z_bins, height=z_freq, width=z_bin_width)
    axes.set_title("photons per bucket by z")
    axes.set_xlabel("z dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(334)
    plt.bar(x=phi_bins_deg, height=phi_freq, width=phi_bin_width_deg)
    axes.set_title("photons per bucket by phi")
    axes.set_xlabel("azimuth (phi) (degrees)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(335)
    plt.bar(x=theta_bins_deg, height=theta_freq, width=theta_bin_width_deg)
    axes.set_title("photons per bucket by theta")
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(336)
    plt.bar(
        x=theta_bins_deg,
        height=photons_per_steradian_by_theta,
        width=theta_bin_width_deg,
    )
    axes.set_title("radiant intensity photons/sr by theta by bucket area")
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per ... ? (TODO: sr)")
    # notice the higher variance at the poles; N is lower there, which is not good.

    axes = plt.subplot(337)
    plt.bar(
        x=theta_bins_deg,
        height=photons_per_steridian_by_theta,
        width=theta_bin_width_deg,
    )
    axes.set_title("radiant intensity photons/sr by photon weights")
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per ... ? (TODO: sr)")

    plt.show()

    fig = plt.figure(figsize=[15, 12])
    axes = plt.subplot(projection="polar")
    axes.set_theta_zero_location("N")
    # mirror the data so it looks nice .. resulting in the weird bounds here
    axes.set_thetamin(-theta_max * 180 / np.pi)
    axes.set_thetamax(theta_max * 180 / np.pi)
    axes.bar(
        theta_bins_rad,
        photons_per_steridian_by_theta,
        width=theta_bin_width_rad,
        color="black",
    )
    axes.bar(
        -theta_bins_rad,
        photons_per_steridian_by_theta,
        width=theta_bin_width_rad,
        color="black",
    )
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per ... ? (TODO: sr)")
