import optics_cuda
import stats_cuda


class ResultStage:
    def __init__(self):
        self._photons_size = 0
        self._photons_energy_j = 0
        self._histogram_r_x = stats_cuda.Histogram()
        self._histogram_r_y = stats_cuda.Histogram()
        self._histogram_ez_phi = stats_cuda.Histogram()
        self._histogram_ez_theta_weighted = stats_cuda.Histogram()
        self._histogram_ez_theta_radiance = stats_cuda.Histogram()
        self._sample = optics_cuda.PhotonsStacked()
        self._ray_length = None
        self._ray_color = None
        self._box = None
        self._box_color = None
        self._label = None


class SimulationResult:
    """Additive metrics produced from N waves of simulation."""

    def __init__(self):
        # photons as they emerge from the source
        self._source_stage = ResultStage()
        # photons at the top of the light box
        self._box_stage = ResultStage()
        # photons scattered by the diffuser
        self._diffuser_stage = ResultStage()
        # photons indicent at the reflector
        self._outbound_stage = ResultStage()
        # photons reflected by the reflector
        self._inbound_stage = ResultStage()
        # photons arriving at the camera plane
        self._camera_plane_stage = ResultStage()
