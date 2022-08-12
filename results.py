import numpy as np
import optics_cuda
import stats_cuda


class ResultStage:
    """Experiment setup and results.
    Each stage is an xy plane, with a size and a height.
    """

    def __init__(self, label, size_m, height_m):
        self._label = label
        self._size_m = size_m
        self._height_m = height_m
        self._photons_size = 0
        self._photons_energy_j = 0
        self._photons_power_w = 0
        self._histogram_r_x = stats_cuda.Histogram()
        self._histogram_r_y = stats_cuda.Histogram()
        self._histogram_ez_phi = stats_cuda.Histogram()
        self._histogram_ez_theta_count = stats_cuda.Histogram()
        self._histogram_ez_theta_intensity = stats_cuda.Histogram()
        self._histogram_ez_theta_radiance = stats_cuda.Histogram()

        # intensity w/sr by x,y,theta,phi
        self._histogram_4d_intensity = stats_cuda.Histogram()
        # radiance w/sr/m2 by x,y,theta,phi
        self._histogram_4d_radiance = stats_cuda.Histogram()
        # bundles
        self._histogram_4d_count = stats_cuda.Histogram()
        # grr
        self._scatter = stats_cuda.Scatter()
        self._sample = optics_cuda.PhotonsStacked()
        self._ray_length = 0.01
        self._ray_color = 0xFF0000
        # the box is just for illustration
        self._box = [-size_m / 2, size_m / 2, -size_m / 2, size_m / 2, height_m]
        self._box_color = 0x808080
        # to magnify narrow distributions
        #####
        # self._theta_min = 0
        self._theta_min = 0.01
        #        self._theta_min = np.pi/16
        # self._theta_max = np.pi
        #self._theta_max = 0.99 * np.pi / 2
        # TODO: avoid the singularity at pi/2 somehow
        self._theta_max = 0.99 * np.pi


#        self._theta_max = 15*np.pi/16


# TODO: combine the "results" idea with the flow in the Simulation class somehow

class BaseSimulationResult:
    pass

class BackgroundSimulationResult(BaseSimulationResult):
    def __init__(self):
        reflector_size_m = 0.1
        reflector_distance_m = 10
        box_height_m = 0.04

        self._source_stage = ResultStage("Background Source", reflector_size_m,
                                         reflector_distance_m)
        self._source_stage._ray_length = 0.01

        self._camera_plane_stage = ResultStage("Camera", 0.2, box_height_m)
        self._camera_plane_stage._box_color = 0x0000FF
        self._camera_plane_stage._ray_color = 0x00FF00
        # note offset camera, 1 cm square
        # TODO: show both camera and diffuser
        self._camera_plane_stage._box = [
            0.02,
            0.03,
            -0.0050,
            0.0050,
            box_height_m,
        ]

class SimulationResult(BaseSimulationResult):
    """Additive metrics produced from N waves of simulation."""

    def __init__(self):

        source_size_m = 0.001 # the LED die, more or less
        # TODO: make reflector distance a parameter
        # a reasonable min is 1m, max is 10m (actual max is 16m)
        reflector_size_m = 0.1
        reflector_distance_m = 10
        box_size_m = 0.04
        box_height_m = 0.04

        # photons as they emerge from the source
        # source is about a millimeter square
        # TODO: use the actual measurement
        self._source_stage = ResultStage("Source", source_size_m, 0)
        self._source_stage._ray_length = 0.0001
        # self._source_stage._theta_max = np.pi / 2

        # photons at the top of the light box
        self._box_stage = ResultStage("Lightbox", box_size_m, box_height_m)
        # self._box_stage._theta_max = np.pi / 2

        # photons scattered by the diffuser
        self._diffuser_stage = ResultStage("Diffuser", box_size_m, box_height_m)
        ###        self._diffuser_stage = ResultStage("Diffuser", 0.001, 0.001)

        # photons indicent at the reflector
        self._outbound_stage = ResultStage("Outbound", reflector_size_m,
                                           reflector_distance_m)
        # a very narrow beam arrives at the reflector
        # self._outbound_stage._theta_max = 0.01 * np.pi

        # photons reflected by the reflector
        self._inbound_stage = ResultStage("Inbound", reflector_size_m,
                                          reflector_distance_m)
        # self._inbound_stage._theta_min = 0.9 * np.pi

        # photons arriving at the camera plane
        # camera height is the same as the diffuser
        # camera neighborhood is large so we can see the distribution
        self._camera_plane_stage = ResultStage("Camera", 0.2, box_height_m)
        self._camera_plane_stage._box_color = 0x0000FF
        self._camera_plane_stage._ray_color = 0x00FF00
        # note offset camera, 1 cm square
        # TODO: show both camera and diffuser
        self._camera_plane_stage._box = [
            0.02,
            0.03,
            -0.0050,
            0.0050,
            box_height_m,
        ]

