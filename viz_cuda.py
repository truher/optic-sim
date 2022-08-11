import math
import warnings
import cupy as cp
import numpy as np
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import k3d  # type:ignore
import optics_cuda
import stats_cuda
import simulation

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def summary(stage):
    print(f"photon bundle count: {stage._photons_size}")
    print(f"photon total energy (J): {stage._photons_energy_j}")


def plot_polar_histogram(data: stats_cuda.Histogram):
    # I used to mirror this data but i think it can be deceiving,
    # implying symmetry where there may not be any.
    fig = plt.figure(figsize=[15, 12])
    axes = plt.subplot(projection="polar")
    axes.plot(
        (data._bin_edges[1:] + data._bin_edges[:-1]) / 2,
        data._hist,
        color="blue",
        snap=False,
    )
    axes.set_title(data._title, fontsize=14, fontweight="black")
    axes.set_xlabel(data._xlabel, fontsize=14)
    axes.set_ylabel(data._ylabel, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_histogram_data(data: stats_cuda.Histogram):
    axes = plt.subplot()
    axes.plot((data._bin_edges[1:] + data._bin_edges[:-1]) / 2, data._hist, snap=False)
    axes.set_title(data._title, fontsize=14, fontweight="black")
    axes.set_xlabel(data._xlabel, fontsize=14)
    axes.set_ylabel(data._ylabel, fontsize=14)
    plt.show()

def plot_histogram_4d(data: stats_cuda.Histogram):
    edges = data._bin_edges
    radiance_w_sr_m2 = data._hist

    #max_radiance_w_sr_m2 = cp.amax(radiance_w_sr_m2, axis=(2,3))
    max_radiance_w_sr_m2 = cp.amax(radiance_w_sr_m2, axis=(2,3))
    #max_radiance_w_sr_m2 = cp.sum(radiance_w_sr_m2, axis=(2,3))

    fig=plt.figure(figsize=[15,12])
    plt.imshow(cp.transpose(max_radiance_w_sr_m2).get(), #vmin=0,
               extent=(edges[0][0].item(), edges[0][-1].item(),
                       edges[1][0].item(), edges[1][-1].item())) 
    plt.title(data._title, fontsize=14, fontweight="black")
    plt.xlabel(data._xlabel, fontsize=14)
    plt.ylabel(data._ylabel, fontsize=14)
    plt.colorbar()
    plt.show()

    # pick 9 areas and make polars for each.
    theta_x = (data._bin_edges[2][1:] + data._bin_edges[2][:-1]) / 2
    fig = plt.figure(figsize=[15, 12])
    plt.suptitle(data._title, fontsize=14, fontweight="black")

    N = 3
    for xplot in range(N):
        for yplot in range(N):
            xidx = int(math.floor((xplot * 2 + 1) * data._hist.shape[0]/(N*2)))
            yidx = int(math.floor((yplot * 2 + 1) * data._hist.shape[1]/(N*2)))
            theta_phi = data._hist[xidx, yidx, :, :]
            # TODO: the "sum" here is wrong, need to divide by
            # something like total sr?  i dunno.
            theta_sum_phi = cp.sum(theta_phi, axis=1)
            axes = plt.subplot(N, N, (yplot * N + xplot) + 1, projection='polar')
            axes.plot(
                theta_x.get(),
                theta_sum_phi.get(),
                color="blue",
                snap=False,
            )
    plt.show()
    


def plot_scatter(data):
    fig=plt.figure(figsize=[15,12])
    plt.plot(data._x.get(), data._y.get(), ',', snap=False)
    plt.title(data._title, fontsize=14, fontweight="black")
    plt.xlabel(data._xlabel, fontsize=14)
    plt.ylabel(data._ylabel, fontsize=14)
    plt.show()


def plot_all_histograms(stage):
    plot_histogram_data(stage._histogram_r_x)
    plot_histogram_data(stage._histogram_r_y)
    plot_histogram_data(stage._histogram_ez_phi)
    plot_histogram_data(stage._histogram_ez_theta_count)
    plot_histogram_data(stage._histogram_ez_theta_intensity)
    plot_histogram_data(stage._histogram_ez_theta_radiance)
    plot_polar_histogram(stage._histogram_ez_theta_count)
    plot_polar_histogram(stage._histogram_ez_theta_intensity)
    plot_polar_histogram(stage._histogram_ez_theta_radiance)
    plot_histogram_4d(stage._histogram_4d_count)
    plot_histogram_4d(stage._histogram_4d_intensity)
    plot_histogram_4d(stage._histogram_4d_radiance)
    plot_scatter(stage._scatter)


def _plot_stage_3d(plot, stage):
    scale = 1000
    head_scale = 10
    plot += k3d.vectors(
        stage._sample._p * scale,
        stage._sample._d * stage._ray_length * scale,
        head_size=stage._ray_length * scale / head_scale,
        color=stage._ray_color,
    )
    xmin = stage._box[0] * scale
    xmax = stage._box[1] * scale
    ymin = stage._box[2] * scale
    ymax = stage._box[3] * scale
    z = stage._box[4] * scale
    plot += k3d.mesh(
        [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]],
        [[0, 1, 2], [2, 3, 0]],
        opacity=0.6,
        color=stage._box_color,
        side="both",
    )
    plot += k3d.label(stage._label, position=(xmax, ymax, z), label_box=False)


def plot_stages_3d(stages):
    plot = k3d.plot(axes=["x (mm)", "y (mm)", "z (mm)"])
    for stage in stages:
        _plot_stage_3d(plot, stage)
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
