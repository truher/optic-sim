import math
import warnings
import numpy as np
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import k3d  # type:ignore
import optics_cuda
import stats_cuda
import simulation

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def plot_polar_histogram(data: simulation.Histogram):
    fig = plt.figure(figsize=[15, 12])
    axes = plt.subplot(projection="polar")
    axes.set_theta_zero_location("N")
    # mirror the data so it looks nice
    axes.plot((data._bin_edges[1:]+data._bin_edges[:-1])/2,
              data._hist, color="blue", snap=False)
    axes.plot(-(data._bin_edges[1:]+data._bin_edges[:-1])/2,
              data._hist, color="blue", snap=False)
    axes.set_title(data._title)
    axes.set_xlabel(data._xlabel)
    axes.set_ylabel(data._ylabel)
    plt.show()

def plot_histogram_data(data: simulation.Histogram):
    axes = plt.subplot()
    axes.plot((data._bin_edges[1:]+data._bin_edges[:-1])/2, data._hist, snap=False)
    axes.set_title(data._title)
    axes.set_xlabel(data._xlabel)
    axes.set_ylabel(data._ylabel)
    plt.show()

def plot_all_histograms(stage):
    plot_histogram_data(stage._histogram_r_x)
    plot_histogram_data(stage._histogram_r_y)
    plot_histogram_data(stage._histogram_r_z)
    plot_histogram_data(stage._histogram_ez_phi)
    plot_histogram_data(stage._histogram_ez_theta)
    plot_histogram_data(stage._histogram_ez_theta_weighted)
    plot_polar_histogram(stage._histogram_ez_theta_weighted)

def plot_histogram_slices(
    photon_batch: optics_cuda.Photons,
    # size: np.int32,
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
) -> None:

    size = photon_batch.size()  # ~35ns
    bins = 128  # matches threads etc
    threads_per_block = bins  # because the threads write back
    grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
    block_size = (threads_per_block, 1, 1)

    null_vector = cp.empty(0, dtype=np.float32)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(suptitle)

    h = cp.zeros(bins, dtype=np.int32)
    histogram(
        grid_size,
        block_size,
        (
            photon_batch.r_x,
            null_vector,
            np.int32(0),
            h,
            size,
            np.float32(x_min),
            np.float32((x_max - x_min) / bins),
        ),
    )
    axes = plt.subplot(331)
    plt.plot(np.linspace(x_min, x_max, bins), h.get(), snap=False)
    axes.set_title("photons per bucket by x")
    axes.set_xlabel("x dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    h = cp.zeros(bins, dtype=np.int32)
    histogram(
        grid_size,
        block_size,
        (
            photon_batch.r_y,
            null_vector,
            np.int32(0),
            h,
            size,
            np.float32(y_min),
            np.float32((y_max - y_min) / bins),
        ),
    )
    axes = plt.subplot(332)
    plt.plot(np.linspace(y_min, y_max, bins), h.get(), snap=False)
    axes.set_title("photons per bucket by y")
    axes.set_xlabel("y dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    h = cp.zeros(bins, dtype=np.int32)
    histogram(
        grid_size,
        block_size,
        (
            photon_batch.r_z,
            null_vector,
            np.int32(0),
            h,
            size,
            np.float32(z_min),
            np.float32((z_max - z_min) / bins),
        ),
    )
    axes = plt.subplot(333)
    plt.plot(np.linspace(z_min, z_max, bins), h.get(), snap=False)
    axes.set_title("photons per bucket by z")
    axes.set_xlabel("z dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    h = cp.zeros(bins, dtype=np.int32)
    histogram(
        grid_size,
        block_size,
        (
            photon_batch.ez_y,
            photon_batch.ez_x,
            np.int32(1),
            h,
            size,
            np.float32(phi_min),
            np.float32((phi_max - phi_min) / bins),
        ),
    )
    axes = plt.subplot(334)
    plt.plot(np.linspace(phi_min, phi_max, bins), h.get(), snap=False)
    axes.set_title("photons per bucket by phi")
    axes.set_xlabel("azimuth (phi) (degrees)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    h = cp.zeros(bins, dtype=np.int32)
    histogram(
        grid_size,
        block_size,
        (
            photon_batch.ez_z,
            null_vector,
            np.int32(2),
            h,
            size,
            np.float32(theta_min),
            np.float32((theta_max - theta_min) / bins),
        ),
    )
    axes = plt.subplot(335)
    plt.plot(np.linspace(theta_min, theta_max, bins), h.get(), snap=False)
    axes.set_title("photons per bucket by theta")
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    theta_range = theta_max - theta_min
    theta_bins_rad = theta_range * np.arange(bins) / bins + theta_min
    theta_bin_width_rad = theta_range / bins
    theta_bucket_angle_steradians = np.array(
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
    photons_per_steradian_by_theta = h.get() / theta_bucket_angle_steradians
    axes = plt.subplot(336)
    plt.plot(
        np.linspace(theta_min, theta_max, bins),
        photons_per_steradian_by_theta,
        snap=False,
    )
    axes.set_title("radiant intensity photons/sr by theta by bucket area")
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per ... ? (TODO: sr)")




    fig = plt.figure(figsize=[15, 12])
    axes = plt.subplot(projection="polar")
    axes.set_theta_zero_location("N")
    # mirror the data so it looks nice .. resulting in the weird bounds here
    axes.set_thetamin(-theta_max * 180 / np.pi)
    axes.set_thetamax(theta_max * 180 / np.pi)
    axes.plot(theta_bins_rad, photons_per_steradian_by_theta, color="blue", snap=False)
    axes.plot(-theta_bins_rad, photons_per_steradian_by_theta, color="blue", snap=False)
    axes.set_xlabel("polar angle (theta) (degrees)")
    axes.set_ylabel("photon count per ... ? (TODO: sr)")


def plot_stage_3d(stage):
    plot = k3d.plot()
    plot += k3d.vectors(stage._sample._p,
                        stage._sample._d * stage._ray_length,
                        color=stage._ray_color)
    xmin = stage._box[0]
    xmax = stage._box[1]
    ymin = stage._box[2]
    ymax = stage._box[3]
    z = stage._box[4]
    plot += k3d.mesh([[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]],
                     [[0, 1, 2], [2, 3, 0]],
                     opacity=0.6, color=stage._box_color, side="both")
    plot += k3d.label(stage._label, position=(xmax, ymax, z), label_box=False)
    plot.display()


def plot_3d(bundles, ray_lengths, boxes, labels, colors):
    plot = k3d.plot()
    for ridx, bundle in enumerate(bundles):
        (p, d) = bundle
        ray_length = ray_lengths[ridx]
        plot += k3d.vectors(p, d * ray_length)

    for ridx, box in enumerate(boxes):
        xmin = box[0]
        xmax = box[1]
        ymin = box[2]
        ymax = box[3]
        z = box[4]
        plot += k3d.mesh(
            [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]],
            [[0, 1, 2], [2, 3, 0]],
            opacity=0.5,
            color=colors[ridx],
            side="both",
        )
        plot += k3d.label(labels[ridx], position=(xmax, ymax, z), label_box=False)

    plot.display()
