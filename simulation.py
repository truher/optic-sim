import cupy as cp  # type: ignore
import numpy as np
import stats_cuda
import optics_cuda
import util


class Simulator:
    """The specific geometry etc of this simulation.
    Runs N waves of M photon bundles each.  Each bundle represents a
    set of photons, each photon has a different wavelength (energy).
    """

    def __init__(self, results, waves, bundles, bundle_size):
        """waves: number of full iterations to run
        bundles: number of "Photons" groups (bundles) to run
        bundle_size: photons per bundle, for accounting energy etc
        I think this will all change, specifying something like
        emitter power or whatever but here it is for now.
        """
        self._results = results
        self._waves = waves
        self._bundles = bundles
        self._bundle_size = bundle_size

    def run_all_waves(self):
        for i in range(self._waves):
            self.run()

    def run(self):
        """Add a run to the results."""
        timer = util.MyTimer()

        # make some photons
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

        source = optics_cuda.MonochromaticLambertianSource(
            self._results._source_stage._size_m,
            self._results._source_stage._size_m,
            source_wavelength_nm,
            source_photons_per_bundle,
            duration_s,
        )
        photons = source.make_photons(self._bundles)
        timer.tick("duration 1.0")
        photons.debug(self._results._source_stage._size_m)
        timer.tick("duration 1.2")

        stats_cuda.histogram(
            photons,
            self._results._source_stage,
            neighborhood=self._results._source_stage._size_m,
            theta_max=np.pi / 2,
        )

        self.record_results(self._results._source_stage, photons)
        timer.tick("duration 1.5")

        # propagate through the reflective light box
        lightbox_height_m = self._results._box_stage._height_m
        lightbox = optics_cuda.Lightbox(height = self._results._box_stage._height_m,
            size = self._results._box_stage._size_m)
        lightbox.propagate_without_kernel(photons)
        timer.tick("duration 2.0")

        stats_cuda.histogram(
            photons,
            self._results._box_stage,
            neighborhood = self._results._box_stage._size_m,
            theta_max=np.pi / 2,
        )
        self.record_results(self._results._box_stage, photons)
        timer.tick("duration 3.0")

        # diffuse through the diffuser
        diffuser = optics_cuda.Diffuser(g=np.float32(0.64), absorption=np.float32(0.16))
        diffuser.diffuse(photons)
        timer.tick("duration 3.2")

        stats_cuda.histogram(
            photons,
            self._results._diffuser_stage,
            neighborhood = self._results._diffuser_stage._size_m,
        )
        self.record_results(self._results._diffuser_stage, photons)
        timer.tick("duration 3.5")

        # propagate to the reflector
        optics_cuda.propagate_to_reflector(photons,
            location = self._results._outbound_stage._height_m)

        # eliminate photons that miss the reflector
        photons.prune_outliers(self._results._outbound_stage._size_m)
        timer.tick("duration 4.0")

        stats_cuda.histogram(
            photons,
            self._results._outbound_stage,
            neighborhood = self._results._outbound_stage._size_m,
            theta_max=np.pi / 100,
        )  # a narrow beam
        self.record_results(self._results._outbound_stage, photons)
        timer.tick("duration 4.5")

        # reflect
        # TODO: guess at absorption
        reflector = optics_cuda.Diffuser(
            g=np.float32(-0.9925), absorption=np.float32(0.0)
        )
        reflector.diffuse(photons)
        timer.tick("duration 5.0")

        stats_cuda.histogram(
            photons,
            self._results._inbound_stage,
            neighborhood = self._results._inbound_stage._size_m,
            theta_min=np.pi * 90 / 100,
        )
        self.record_results(self._results._inbound_stage, photons)
        timer.tick("duration 5.5")

        # propagate to the camera
        optics_cuda.propagate_to_camera(photons,
            location = self._results._camera_plane_stage._height_m)

        # eliminate photons that miss the camera by a lot
        photons.prune_outliers(self._results._camera_plane_stage._size_m)
        timer.tick("duration 6.0")

        stats_cuda.histogram(
            photons,
            self._results._camera_plane_stage,
            neighborhood = self._results._camera_plane_stage._size_m
        )
        self.record_results(self._results._camera_plane_stage, photons)
        timer.tick("duration 7.0")

    def record_results(self, stage, photons):
        stage._photons_size += photons.count_alive()
        stage._photons_energy_j += photons.energy_j()
        stage._sample.add(photons.sample())


class Study:
    """Sweep some parameters while measuring output."""

    def __init__(self):
        pass

