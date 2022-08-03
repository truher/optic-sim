import cupy as cp  # type: ignore
import math
import numpy as np
import stats_cuda
import optics_cuda

class Histogram:
    def __init__(self):
        self._hist = None
        self._bin_edges = None
        self._title = ""
        self._xlabel = ""
        self._ylabel = ""

    def add(self, hist):
        if self._hist is None:
            self._hist = hist
        else:
            self._hist += hist

class PhotonsStacked:
    def __init__(self):
        self._p = None
        self._d = None

    def add(self, stack):
        (p, d) = stack
        if p is None:
            raise ValueError()
        if d is None:
            raise ValueError()
        if self._p is None:
            self._p = p
            self._d = d
        else:
            self._p = np.concatenate([self._p, p])
            self._d = np.concatenate([self._d, d])


class ResultStage:
    def __init__(self):
        self._photons_size = 0
        self._histogram_r_x = Histogram()
        self._histogram_r_y = Histogram()
        self._histogram_r_z = Histogram()
        self._histogram_ez_phi = Histogram()
        self._histogram_ez_theta = Histogram()
        self._histogram_ez_theta_weighted = Histogram()
        self._sample = PhotonsStacked()
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

class Simulator:
    """The specific geometry etc of this simulation.
    Runs N waves of M photon bundles each.  Each bundle represents a set of photons,
    each photon has a different wavelength (energy).
    """
    def __init__(self, results, waves, bundles, bundle_size):
        """waves: number of full iterations to run
           bundles: number of "Photons" groups (bundles) to run
           bundle_size: photons per bundle, for accounting energy etc
           I think this will all change, specifying something like emitter power or whatever
           but here it is for now.
        """
        self._results = results
        self._waves = waves
        self._bundles = bundles
        self._bundle_size = bundle_size

    @staticmethod
    def one_histogram(bins, photon_batch_dimension1, photon_batch_dimension2,
                      mapper, dim_min, dim_max, title, xlabel, ylabel, histogram_output):
        size = photon_batch_dimension1.size  # ~35ns
        threads_per_block = bins  # because the threads write back
        grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
        block_size = (threads_per_block, 1, 1)
        h = cp.zeros(bins, dtype=np.int32)
        stats_cuda.histogram(
            grid_size,
            block_size,
            (
                photon_batch_dimension1,
                photon_batch_dimension2,
                np.int32(mapper),
                h,
                np.int32(size),
                np.float32(dim_min),
                np.float32((dim_max - dim_min) / bins),
            ),
        )
        # TODO this is wrong, it is not edges, it is not lower edges, it is just wrong
        histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
        #histogram_output._hist = h.get()
        histogram_output.add(h.get())
        histogram_output._title = title
        histogram_output._xlabel = xlabel
        histogram_output._ylabel = ylabel

    def histogram(self, photon_batch, stage,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        theta_min: float = 0,
        theta_max: float = np.pi,
        phi_min: float = -np.pi,
        phi_max: float = np.pi):
        """Make and store a set of histograms."""
        # TODO: do bounds automatically

        null_vector = cp.empty(0, dtype=np.float32)
        Simulator.one_histogram(128, 
                photon_batch.r_x,
                null_vector, 0, x_min, x_max, 
                "photons per bucket by x",
                "x dimension (TODO: unit)",
                "photon count per bucket (TODO: density)",
                stage._histogram_r_x)

        Simulator.one_histogram(128, 
                photon_batch.r_y,
                null_vector, 0, y_min, y_max, 
                "photons per bucket by y",
                "y dimension (TODO: unit)",
                "photon count per bucket (TODO: density)",
                stage._histogram_r_y)

        Simulator.one_histogram(128, 
                photon_batch.r_z,
                null_vector, 0, z_min, z_max, 
                "photons per bucket by z",
                "z dimension (TODO: unit)",
                "photon count per bucket (TODO: density)",
                stage._histogram_r_z)

        Simulator.one_histogram(128, 
                photon_batch.ez_y,
                photon_batch.ez_x,
                1, phi_min, phi_max, 
                "photons per bucket by phi",
                "azimuth phi (radians)",
                "photon count per bucket (TODO: density)",
                stage._histogram_ez_phi)

        Simulator.one_histogram(128, 
                photon_batch.ez_z,
                null_vector,
                2, theta_min, theta_max, 
                "photons per bucket by theta",
                "polar angle theta (radians)",
                "photon count per bucket (TODO: density)",
                stage._histogram_ez_theta)

        # TODO: maybe move this somewhere else
        h = stage._histogram_ez_theta._hist
        b = stage._histogram_ez_theta._bin_edges
        area = np.cos(b[:-1]) - np.cos(b[1:])
        stage._histogram_ez_theta_weighted._hist = h / area
        stage._histogram_ez_theta_weighted._bin_edges = b
        stage._histogram_ez_theta_weighted._title = "radiant intensity by theta by bucket area"
        stage._histogram_ez_theta_weighted._xlabel = "polar angle theta (radians)"
        stage._histogram_ez_theta_weighted._ylabel = "photon count per ... ? (TODO: sr)"


    def run_all_waves(self):
        for i in range(self._waves):
            self.run()

    def run(self):
        """Add a run to the results."""
        # make some photons

        source_size = np.float32(10)
        source = optics_cuda.LambertianSource(source_size, source_size)
        photons = source.make_photons(self._bundles)

        self._results._source_stage._photons_size += photons.size()
        self.histogram(photons, self._results._source_stage,
                       x_min = -source_size/2, x_max = source_size/2,
                       y_min = -source_size/2, y_max = source_size/2,
                       z_min = -5, z_max = 5, theta_max = np.pi/2)

        self._results._source_stage._sample.add(photons.sample())
        self._results._source_stage._ray_length = 1
        self._results._source_stage._ray_color = 0xff0000
        self._results._source_stage._box = [-source_size/2, source_size/2, -source_size/2, source_size/2, 0]
        self._results._source_stage._box_color = 0x808080
        self._results._source_stage._label = "Source"

        # propagate through the reflective light box

        lightbox_height = 400
        lightbox_size = 400
        lightbox = optics_cuda.Lightbox(height = lightbox_height, size = lightbox_size)
        lightbox.propagate(photons)

        self._results._box_stage._photons_size += photons.size()
        self.histogram(photons, self._results._box_stage,
                       x_min = -lightbox_size/2, x_max = lightbox_size/2,
                       y_min = -lightbox_size/2, y_max = lightbox_size/2,
                       z_min = 0, z_max = 1000, theta_max = np.pi/2)
        self._results._box_stage._sample.add(photons.sample())
        self._results._box_stage._ray_length = 100
        self._results._box_stage._ray_color = 0xff0000
        self._results._box_stage._box = [-lightbox_size/2, lightbox_size/2,
                                         -lightbox_size/2, lightbox_size/2, lightbox_height]
        self._results._box_stage._box_color = 0x808080
        self._results._box_stage._label = "Lightbox"

        # diffuse through the diffuser

        diffuser = optics_cuda.Diffuser(g = np.float32(0.64), absorption = np.float32(0.16))
        diffuser.diffuse(photons)

        self._results._diffuser_stage._photons_size += photons.size()
        self.histogram(photons, self._results._diffuser_stage,
                       x_min = -lightbox_size/2, x_max = lightbox_size/2,
                       y_min = -lightbox_size/2, y_max = lightbox_size/2,
                       z_min = 0, z_max = 1000)
        self._results._diffuser_stage._sample.add(photons.sample())
        self._results._diffuser_stage._ray_length = 100
        self._results._diffuser_stage._ray_color = 0xff0000
        self._results._diffuser_stage._box = [-lightbox_size/2, lightbox_size/2,
                                              -lightbox_size/2, lightbox_size/2, lightbox_height]
        self._results._diffuser_stage._box_color = 0x808080
        self._results._diffuser_stage._label = "Diffuser"

        # propagate to the reflector

        # TODO: make distance a parameter
        #reflector_distance = np.float32(100000)
        reflector_distance = np.float32(50000)
        #reflector_distance = np.float32(10000)
        reflector_size = 1000
        optics_cuda.propagate_to_reflector(photons, location = reflector_distance, size = np.float32(1000))
        # eliminate photons that miss the reflector
        photons.prune_outliers(reflector_size)

        self._results._outbound_stage._photons_size += photons.size()
        self.histogram(photons, self._results._outbound_stage,
                       x_min = -reflector_size/2, x_max = reflector_size/2,
                       y_min = -reflector_size/2, y_max = reflector_size/2,
                       z_min = 0, z_max = reflector_distance,
                       theta_max=np.pi/100)  # a narrow beam
        self._results._outbound_stage._sample.add(photons.sample())
        self._results._outbound_stage._ray_length = 100
        self._results._outbound_stage._ray_color = 0xff0000
        self._results._outbound_stage._box = [-reflector_size/2, reflector_size/2,
                                              -reflector_size/2, reflector_size/2, reflector_distance]
        self._results._outbound_stage._box_color = 0x808080
        self._results._outbound_stage._label = "Outbound"

        # reflect

        # TODO: guess at absorption
        reflector = optics_cuda.Diffuser(g = np.float32(-0.9925), absorption=np.float32(0.0))
        reflector.diffuse(photons)

        self._results._inbound_stage._photons_size += photons.size()
        self.histogram(photons, self._results._inbound_stage,
                       x_min = -reflector_size/2, x_max = reflector_size/2,
                       y_min = -reflector_size/2, y_max = reflector_size/2,
                       z_min = 0, z_max = reflector_distance, theta_min = np.pi*90/100)
        self._results._inbound_stage._sample.add(photons.sample())
        self._results._inbound_stage._ray_length = 100
        self._results._inbound_stage._ray_color = 0xff0000
        self._results._inbound_stage._box = [-reflector_size/2, reflector_size/2,
                                              -reflector_size/2, reflector_size/2, reflector_distance]
        self._results._inbound_stage._box_color = 0x808080
        self._results._inbound_stage._label = "Inbound"

        # propagate to the camera

        # make the camera height even with the diffuser
        camera_distance = np.float32(lightbox_height)
        optics_cuda.propagate_to_camera(photons, location = camera_distance)
        # eliminate photons that miss the camera by a lot
        camera_neighborhood = 2000
        photons.prune_outliers(camera_neighborhood)

        self._results._camera_plane_stage._photons_size += photons.size()
        self.histogram(photons, self._results._camera_plane_stage,
                       x_min = -camera_neighborhood/2, x_max = camera_neighborhood/2,
                       y_min = -camera_neighborhood/2, y_max = camera_neighborhood/2,
                       z_min = 0, z_max = camera_distance)
        self._results._camera_plane_stage._sample.add(photons.sample())
        self._results._camera_plane_stage._ray_length = 100
        self._results._camera_plane_stage._ray_color = 0xff0000
        # note offset camera
        self._results._camera_plane_stage._box = [200, 300, -50, 50, camera_distance]
        self._results._camera_plane_stage._box_color = 0x0000ff
        self._results._camera_plane_stage._label = "Camera"


class Study:
    """Sweep some parameters while measuring output."""
    def __init__(self):
        pass
