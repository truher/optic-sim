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
        photons.debug(self._results._source_stage._size_m)
        self.record_results(self._results._source_stage, photons)

        # propagate through the reflective light box
        lightbox = optics_cuda.Lightbox(
            height=self._results._box_stage._height_m,
            size=self._results._box_stage._size_m,
        )
        lightbox.propagate_without_kernel(photons)
        self.record_results(self._results._box_stage, photons)

        # diffuse through the diffuser
        diffuser = optics_cuda.Diffuser(g=0.64, absorption=0.16)
        diffuser.diffuse(photons)
        self.record_results(self._results._diffuser_stage, photons)

        # propagate to the reflector and eliminate photons that miss it
        optics_cuda.propagate_to_reflector(
            photons, location=self._results._outbound_stage._height_m
        )
        photons.prune_outliers(self._results._outbound_stage._size_m)
        self.record_results(self._results._outbound_stage, photons)

        # reflect TODO: guess at absorption
        reflector = optics_cuda.Diffuser(g=-0.9925, absorption=0.0)
        reflector.diffuse(photons)
        self.record_results(self._results._inbound_stage, photons)

        # propagate to the camera and eliminate photons that miss it by a lot
        optics_cuda.propagate_to_camera(
            photons, location=self._results._camera_plane_stage._height_m
        )
        photons.prune_outliers(self._results._camera_plane_stage._size_m)
        self.record_results(self._results._camera_plane_stage, photons)

        timer.tick("one run")

    def record_results(self, stage, photons):
        stage._photons_size += photons.count_alive()
        stage._photons_energy_j += photons.energy_j()
        stage._sample.add(photons.sample())
        # stats_cuda.histogram(photons, stage, neighborhood = stage._size_m,
        #    theta_min = stage._theta_min, theta_max = stage._theta_max)
        stats_cuda.histogram(photons, stage)


class Study:
    """Sweep some parameters while measuring output."""

    def __init__(self):
        pass
