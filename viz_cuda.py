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
    bins: int = 100,
) -> None:
