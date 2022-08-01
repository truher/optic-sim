import optics_cuda 

def plot_histogram_slices(
    photon_batch: optics.Photons,
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
    bins: int = 100) -> None:


    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(suptitle)

    axes = plt.subplot(331)
    plt.hist(photon_batch.r_x.get(), bins = bins)
    axes.set_title("photons per bucket by x")
    axes.set_xlabel("x dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(332)
    plt.hist(photon_batch.r_y.get(), bins=bins)
    axes.set_title("photons per bucket by y")
    axes.set_xlabel("y dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(333)
    plt.hist(photon_batch.r_z, bins=bins)
    axes.set_title("photons per bucket by z")
    axes.set_xlabel("z dimension (TODO: unit)")
    axes.set_ylabel("photon count per bucket (TODO: density)")

    axes = plt.subplot(334)
    plt.hist(x=phi_bins_deg, height=phi_freq, width=phi_bin_width_deg)
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
    axes.set_ylabel("photon count per ... ? (TODO: sr)")) -> None:
