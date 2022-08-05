import cupy as cp  # type: ignore
import math
import numpy as np
import time
import stats_cuda
import optics_cuda
import scipy.constants

class Histogram:
    def __init__(self):
        self._hist = None # np.ndarray, TODO make cp.ndarray
        self._bin_edges = None
        self._title = ""
        self._xlabel = ""
        self._ylabel = ""

    def add(self, hist): # np.array(float32)
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
        self._histogram_ez_phi = Histogram()
        self._histogram_ez_theta_weighted = Histogram()
        self._histogram_ez_theta_radiance = Histogram()
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
    def one_histogram_phi(grid_size, block_size, bins, size, photon_batch_alive, photon_batch_wavelength_nm,
                      photon_batch_dimension1, photon_batch_dimension2,
                      photons_per_bundle,
                      dim_min, dim_max, title, xlabel, ylabel, bin_area, duration_s,
                      histogram_output):
        h = cp.zeros(bins, dtype=np.float32) # joules, so this is joules per bucket

        stats_cuda.histogram_phi(
            grid_size,
            block_size,
            (
                photon_batch_alive,
                photon_batch_wavelength_nm,
                photon_batch_dimension1,
                photon_batch_dimension2,
                h,
                np.int32(size),
                np.float32(dim_min),
                np.float32((dim_max - dim_min) / bins),
                np.int32(photons_per_bundle)
            ),
        )

        histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
        histogram_output.add(h.get()/(bin_area * duration_s))
        histogram_output._title = title
        histogram_output._xlabel = xlabel
        histogram_output._ylabel = ylabel

    @staticmethod
    def one_histogram_theta(grid_size, block_size, bins, size, photon_batch_alive, photon_batch_wavelength_nm,
                      photon_batch_dimension1,
                      photons_per_bundle,
                      dim_min, dim_max, title, xlabel, ylabel, bin_area, duration_s,
                      histogram_output):
        h = cp.zeros(bins, dtype=np.float32) # joules, so this is joules per bucket
        stats_cuda.histogram_theta(
            grid_size,
            block_size,
            (
                photon_batch_alive,
                photon_batch_wavelength_nm,
                photon_batch_dimension1,
                h,
                np.int32(size),
                np.float32(dim_min),
                np.float32((dim_max - dim_min) / bins),
                np.int32(photons_per_bundle)
            ),
        )

        histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
        histogram_output.add(h.get()/(bin_area * duration_s))
        histogram_output._title = title
        histogram_output._xlabel = xlabel
        histogram_output._ylabel = ylabel


    @staticmethod
    def one_histogram(grid_size, block_size, bins, size, photon_batch_alive, photon_batch_wavelength_nm,
                      photon_batch_dimension1,
                      photons_per_bundle,
                      dim_min, dim_max, title, xlabel, ylabel, bin_area, duration_s,
                      histogram_output):
        h = cp.zeros(bins, dtype=np.float32) # joules, so this is joules per bucket
        stats_cuda.histogram(
            grid_size,
            block_size,
            (
                photon_batch_alive,
                photon_batch_wavelength_nm,
                photon_batch_dimension1,
                h,
                np.int32(size),
                np.float32(dim_min),
                np.float32((dim_max - dim_min) / bins),
                np.int32(photons_per_bundle)
            ),
        )

        histogram_output._bin_edges = np.linspace(dim_min, dim_max, bins + 1)
        histogram_output.add(h.get()/(bin_area * duration_s))
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

        bins = 128
        size = photon_batch.size()
        # strids = work per thread 
        strides = 32

        grid_size = (int(math.ceil(size / (bins * strides))), )
        block_size = (bins, )

        # for an areal histogram, measure radiosity, power per area, w/m^2
        bin_area_m2 = (y_max - y_min) * (x_max - x_min) / bins
        Simulator.one_histogram(grid_size, block_size, bins, size,
                photon_batch.alive,
                photon_batch.wavelength_nm,
                photon_batch.r_x,
                photon_batch.photons_per_bundle, x_min, x_max, 
                "Radiosity",
                r"X dimension $\mathregular{(m^2)}$",
                r"Radiosity $\mathregular{(W/m^2)}$",
                bin_area_m2, photon_batch.duration_s,
                stage._histogram_r_x)

        Simulator.one_histogram(grid_size, block_size, bins, size,
                photon_batch.alive,
                photon_batch.wavelength_nm,
                photon_batch.r_y,
                photon_batch.photons_per_bundle, y_min, y_max, 
                "Radiosity",
                r"Y dimension $\mathregular{(m^2)}$",
                r"Radiosity $\mathregular{(W/m^2)}$",
                bin_area_m2, # happens to be same as above
                photon_batch.duration_s,
                stage._histogram_r_y)

        # for an angular histogram we're measuring
        # radiant intensity, power per solid angle, w/sr
        bin_area_sr = 4 * np.pi / bins
        # note that the radiant intensity varies a lot by *theta* i.e. not the
        # quantity bucketed here (see below)
        Simulator.one_histogram_phi(grid_size, block_size, bins, size,
                photon_batch.alive,
                photon_batch.wavelength_nm,
                photon_batch.ez_y,
                photon_batch.ez_x,
                photon_batch.photons_per_bundle, phi_min, phi_max, 
                "Radiant Intensity",
                r"Azimuth phi $\mathregular{(radians)}$",
                r"Radiant Intensity $\mathregular{(W/sr)}$",
                bin_area_sr,
                photon_batch.duration_s,
                stage._histogram_ez_phi)

        bin_edges = np.linspace(theta_min, theta_max, bins + 1)
        bin_area_sr = (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:]) ) * 2 * np.pi
        Simulator.one_histogram_theta(grid_size, block_size, bins, size,
                photon_batch.alive,
                photon_batch.wavelength_nm,
                photon_batch.ez_z,
                photon_batch.photons_per_bundle, theta_min, theta_max, 
                "Radiant Intensity",
                r"Polar angle theta $\mathregular{(radians)}$",
                r"Radiant Intensity $\mathregular{(W/sr)}$",
                bin_area_sr,
                photon_batch.duration_s,
                stage._histogram_ez_theta_weighted)


        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        projected_area_m2 =  (y_max - y_min) * (x_max - x_min)  * np.abs(np.cos(bin_centers))
        bin_area_sr_m2 = (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:]) ) * 2 * np.pi * projected_area_m2
        Simulator.one_histogram_theta(grid_size, block_size, bins, size,
                photon_batch.alive,
                photon_batch.wavelength_nm,
                photon_batch.ez_z,
                photon_batch.photons_per_bundle, theta_min, theta_max, 
                "Radiance",
                "Polar angle theta $\mathregular{(radians)}$",
                "Radiance $\mathregular{(W/sr/m^2)}$",
                bin_area_sr_m2,
                photon_batch.duration_s,
                stage._histogram_ez_theta_radiance)


    def run_all_waves(self):
        for i in range(self._waves):
            self.run()

    def run(self):
        """Add a run to the results."""
        timer = MyTimer()

        # make some photons
        # about a millimeter square
        # TODO: use the actual measurement
        source_size_m = np.float32(0.001)
        # 555 nm is the peak lumens per watt.
        # TODO: make each photon in the bundle choose from a distribution
        source_wavelength_nm = 555
        # used to calculate energy
        # TODO: calculate this number from the published output
        # waves: 100; photons per bundle: 1e7
        # waves:  20; photons per bundle: 5e7
        source_photons_per_bundle = 5e7
        # duration of the strobe, used to calculate power
        duration_s = 0.001
        source = optics_cuda.MonochromaticLambertianSource(source_size_m,
                                                           source_size_m,
                                                           source_wavelength_nm,
                                                           source_photons_per_bundle,
                                                           duration_s)
        photons = source.make_photons(self._bundles)
        timer.tick("duration 1.0")
        photons.debug(source_size_m)
        timer.tick("duration 1.2")

        # make histograms
        self._results._source_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._source_stage,
                       x_min = -source_size_m/2, x_max = source_size_m/2,
                       y_min = -source_size_m/2, y_max = source_size_m/2,
                       z_min = -0.005, z_max = 0.005, theta_max = np.pi/2)

        self._results._source_stage._sample.add(photons.sample())
        self._results._source_stage._ray_length = 0.0002
        self._results._source_stage._ray_color = 0xff0000
        self._results._source_stage._box = [-source_size_m/2, source_size_m/2,
                                            -source_size_m/2, source_size_m/2, 0]
        self._results._source_stage._box_color = 0x808080
        self._results._source_stage._label = "Source"
        timer.tick("duration 1.5")

        # propagate through the reflective light box
        lightbox_height_m = 0.04 # 4 cm
        lightbox_size_m = 0.04 # 4 cm
        lightbox = optics_cuda.Lightbox(height = lightbox_height_m, size = lightbox_size_m)
        lightbox.propagate_without_kernel(photons)
        timer.tick("duration 2.0")

        # make histograms
        self._results._box_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._box_stage,
                       x_min = -lightbox_size_m/2, x_max = lightbox_size_m/2,
                       y_min = -lightbox_size_m/2, y_max = lightbox_size_m/2,
                       z_min = 0, z_max = 0.1, theta_max = np.pi/2)
        self._results._box_stage._sample.add(photons.sample())
        self._results._box_stage._ray_length = 0.01
        self._results._box_stage._ray_color = 0xff0000
        self._results._box_stage._box = [-lightbox_size_m/2, lightbox_size_m/2,
                                         -lightbox_size_m/2, lightbox_size_m/2, lightbox_height_m]
        self._results._box_stage._box_color = 0x808080
        self._results._box_stage._label = "Lightbox"

        timer.tick("duration 3.0")

        # diffuse through the diffuser
        diffuser = optics_cuda.Diffuser(g = np.float32(0.64), absorption = np.float32(0.16))
        diffuser.diffuse(photons)
        timer.tick("duration 3.2")

        # make histograms
        self._results._diffuser_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._diffuser_stage,
                       x_min = -lightbox_size_m/2, x_max = lightbox_size_m/2,
                       y_min = -lightbox_size_m/2, y_max = lightbox_size_m/2,
                       z_min = 0, z_max = 0.1)
        self._results._diffuser_stage._sample.add(photons.sample())
        self._results._diffuser_stage._ray_length = 0.01
        self._results._diffuser_stage._ray_color = 0xff0000
        self._results._diffuser_stage._box = [-lightbox_size_m/2, lightbox_size_m/2,
                                              -lightbox_size_m/2, lightbox_size_m/2, lightbox_height_m]
        self._results._diffuser_stage._box_color = 0x808080
        self._results._diffuser_stage._label = "Diffuser"

        timer.tick("duration 3.5")

        # propagate to the reflector
        # TODO: make distance a parameter
        reflector_distance_m = np.float32(10)
        #reflector_distance_m = np.float32(5)
        #reflector_distance_m = np.float32(1)
        reflector_size_m = 0.1 # 10 cm
        optics_cuda.propagate_to_reflector(photons, location = reflector_distance_m)
        # eliminate photons that miss the reflector
        photons.prune_outliers(reflector_size_m)
        timer.tick("duration 4.0")

        # make histograms
        self._results._outbound_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._outbound_stage,
                       x_min = -reflector_size_m/2, x_max = reflector_size_m/2,
                       y_min = -reflector_size_m/2, y_max = reflector_size_m/2,
                       z_min = 0, z_max = reflector_distance_m,
                       theta_max=np.pi/100)  # a narrow beam
        self._results._outbound_stage._sample.add(photons.sample())
        self._results._outbound_stage._ray_length = 0.01
        self._results._outbound_stage._ray_color = 0xff0000
        self._results._outbound_stage._box = [-reflector_size_m/2, reflector_size_m/2,
                                              -reflector_size_m/2, reflector_size_m/2, reflector_distance_m]
        self._results._outbound_stage._box_color = 0x808080
        self._results._outbound_stage._label = "Outbound"

        timer.tick("duration 4.5")

        # reflect
        # TODO: guess at absorption
        reflector = optics_cuda.Diffuser(g = np.float32(-0.9925), absorption=np.float32(0.0))
        reflector.diffuse(photons)

        timer.tick("duration 5.0")

        # make histograms
        self._results._inbound_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._inbound_stage,
                       x_min = -reflector_size_m/2, x_max = reflector_size_m/2,
                       y_min = -reflector_size_m/2, y_max = reflector_size_m/2,
                       z_min = 0, z_max = reflector_distance_m, theta_min = np.pi*90/100)
        self._results._inbound_stage._sample.add(photons.sample())
        self._results._inbound_stage._ray_length = 0.01
        self._results._inbound_stage._ray_color = 0xff0000
        self._results._inbound_stage._box = [-reflector_size_m/2, reflector_size_m/2,
                                              -reflector_size_m/2, reflector_size_m/2, reflector_distance_m]
        self._results._inbound_stage._box_color = 0x808080
        self._results._inbound_stage._label = "Inbound"

        timer.tick("duration 5.5")

        # propagate to the camera
        # make the camera height even with the diffuser
        camera_distance_m = np.float32(lightbox_height_m)
        optics_cuda.propagate_to_camera(photons, location = camera_distance_m)
        # eliminate photons that miss the camera by a lot
        camera_neighborhood_m = 0.2
        photons.prune_outliers(camera_neighborhood_m)
        timer.tick("duration 6.0")

        # make histograms
        self._results._camera_plane_stage._photons_size += photons.count_alive()
        self.histogram(photons, self._results._camera_plane_stage,
                       x_min = -camera_neighborhood_m/2, x_max = camera_neighborhood_m/2,
                       y_min = -camera_neighborhood_m/2, y_max = camera_neighborhood_m/2,
                       z_min = 0, z_max = camera_distance_m)
        self._results._camera_plane_stage._sample.add(photons.sample())
        self._results._camera_plane_stage._ray_length = 0.01
        self._results._camera_plane_stage._ray_color = 0xff0000
        # note offset camera, 1 cm square
        self._results._camera_plane_stage._box = [0.02, 0.03, -0.0050, 0.0050, camera_distance_m]
        self._results._camera_plane_stage._box_color = 0x0000ff
        self._results._camera_plane_stage._label = "Camera"
        timer.tick("duration 7.0")

class MyTimer:
    def __init__(self):
        self._t0 = time.monotonic_ns()
    def tick(self, label):
        return
        cp.cuda.Device().synchronize()
        t1 = time.monotonic_ns()
        print(f"{label} {t1 -self._t0}")
        self._t0 = t1

class Study:
    """Sweep some parameters while measuring output."""
    def __init__(self):
        pass
