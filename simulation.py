import cupy as cp  # type: ignore
import numpy as np
import optics_cuda
import spectra
import stats_cuda
import viz_cuda
import util
from tqdm.notebook import tqdm, trange
from IPython.display import Markdown, display

class BaseSimulator:
    def __init__(self, results, waves, bundles, bundle_size):
        """waves: number of full iterations to run
        bundles: number of "Photons" groups (bundles) to run
        bundle_size: photons per bundle, for accounting energy etc
        I think this will all change, specifying something like
        emitter power or whatever but here it is for now.
        """
        # TODO: move stuff from results to members here.
        self._results = results
        self._waves = waves
        self._bundles = bundles
        self._bundle_size = bundle_size  # TODO Actually use this?
### for astm e810, move the camera around
# on-axis
        #self._camera = optics_cuda.Camera(-0.005, 0.005, -0.0050, 0.0050)
# 0.1 degrees # this is the FRC setup
        self._camera = optics_cuda.Camera(0.02, 0.03, -0.0050, 0.0050)
# 0.2 degrees
        #self._camera = optics_cuda.Camera(0.045, 0.055, -0.0050, 0.0050)
# 0.33 degrees
        #self._camera = optics_cuda.Camera(0.08, 0.09, -0.0050, 0.0050)
# 0.5 degrees
        #self._camera = optics_cuda.Camera(0.125, 0.135, -0.0050, 0.0050)
# 1 degree
        #self._camera = optics_cuda.Camera(0.255, 0.265, -0.0050, 0.0050)
# 2 degrees
        #self._camera = optics_cuda.Camera(0.515, 0.525, -0.0050, 0.0050)

    def run_all_waves(self):
        # for i in range(self._waves):
        for i in trange(self._waves):
            self.run()
        display(Markdown("# Done"))

    def record_results(self, stage, photons):
        # assume all bundles same?  instead add each bundle count to a total?
        stage._photons_per_bundle = photons.photons_per_bundle
        stage._photons_size += photons.count_alive()
        stage._photons_energy_j += photons.energy_j()
        stage._photons_power_w += photons.power_w()
        stage._luminous_flux_lm += photons.luminous_flux_lm()
        stage._sample.add(photons.sample())
        stats_cuda.histogram(photons, stage)


class LuminaireSimulator(BaseSimulator):
    """Simulate a light fixture in the frame."""

    def run(self):
        #source_wavelength_nm = 555
        #source_photons_per_bundle = 5e7 # led
        #source_photons_per_bundle = 5e4 # background
        # aiming for 1.5 w/sr/m2 source radiance
# luminaires produce 4k lm each
# ok we're guessing that a 4k luminaire has a 0.25m^2 area
# but we want a 0.04 m^2 area to make the math easier
# so divide by about 6
        source_photons_per_bundle = 7e6
        duration_s = 0.001

        source = optics_cuda.LambertianSource(
            self._results._source_stage._size_m,
            #self._results._source_stage._size_m,
            self._results._source_stage._height_m,
            #source_wavelength_nm,
            spectra.SourceSpectrum.LED_COOL_WHITE,
            source_photons_per_bundle,
            duration_s,
        )
        photons = source.make_photons(self._bundles)
        # point the source at the camera
        photons.ez_z *= -1.0
        #photons.debug(self._results._source_stage._size_m)
        self.record_results(self._results._source_stage, photons)

        optics_cuda.propagate_to_camera(
            photons, location=self._results._camera_plane_stage._height_m
        )
        photons.prune_outliers(self._results._camera_plane_stage._size_m)
        self.record_results(self._results._camera_plane_stage, photons)

        # apply color filter
        flt = spectra.FilterSpectrum.FILTER_27
        flt.absorb(photons.wavelength_nm, photons.alive)

        self.record_results(self._results._filter_stage, photons)

        # only look at what the camera will look at
        # TODO: clean this up somehow
        photons.prune_outliers2(
            self._camera._xmin,
            self._camera._xmax,
            self._camera._ymin,
            self._camera._ymax)
        self.record_results(self._results._camera_lens_stage, photons)

        self._camera.count(photons)

class BackgroundSimulator(BaseSimulator):
    """Simulate the path from background objects to the camera."""

    def run(self):
        # TODO: obs this is wrong
        #source_wavelength_nm = 555
        #source_photons_per_bundle = 5e7
        # background radiance
        # aiming for 0.07 w/sr/m2 source radiance
###
#        source_photons_per_bundle = 1.5e4
        # this is for 0.04 source, which the ASTM thing uses so keep it
        #source_photons_per_bundle = 3.0e4
        source_photons_per_bundle = 8.3e4
        duration_s = 0.001

        source = optics_cuda.LambertianSource(
            self._results._source_stage._size_m,
            #self._results._source_stage._size_m,
            self._results._source_stage._height_m,
            #source_wavelength_nm,
            spectra.SourceSpectrum.LED_COOL_WHITE,
            source_photons_per_bundle,
            duration_s,
        )
        photons = source.make_photons(self._bundles)
        # point the source at the camera
        photons.ez_z *= -1.0
        #photons.debug(self._results._source_stage._size_m)
        self.record_results(self._results._source_stage, photons)

        optics_cuda.propagate_to_camera(
            photons, location=self._results._camera_plane_stage._height_m
        )
        photons.prune_outliers(self._results._camera_plane_stage._size_m)
        self.record_results(self._results._camera_plane_stage, photons)

        # apply color filter
        flt = spectra.FilterSpectrum.FILTER_27
        flt.absorb(photons.wavelength_nm, photons.alive)

        self.record_results(self._results._filter_stage, photons)

        # only look at what the camera will look at
        # TODO: clean this up somehow
        photons.prune_outliers2(
            self._camera._xmin,
            self._camera._xmax,
            self._camera._ymin,
            self._camera._ymax)
        self.record_results(self._results._camera_lens_stage, photons)

        self._camera.count(photons)


class Simulator(BaseSimulator):
    """The specific geometry etc of this simulation.
    Runs N waves of M photon bundles each.  Each bundle represents a
    set of photons, each photon has a different wavelength (energy).
    """
    def run(self):
        """Add a run to the results."""
        timer = util.MyTimer()

        # make some photons
        # 555 nm is the peak lumens per watt.
        # TODO: make each photon in the bundle choose from a distribution
        #source_wavelength_nm = 555

        # used to calculate energy
        # TODO: calculate this number from the published output
        # waves: 100; photons per bundle: 1e7
        # waves:  20; photons per bundle: 5e7
###
# for astm e810 for far-away camera, need more resolution
#        source_photons_per_bundle = 5e7
# 3e6 works for one led, produces about 700mW which is the 1 amp number
        #source_photons_per_bundle = 3e6
# far red 8.4w output, which is the absolute max of 6 (3x overdrive <10% duty cycle)
###        source_photons_per_bundle = 4e7
# try photo red 10.6w total for 6 per eye
        source_photons_per_bundle = 4.3e7


        # duration of the strobe, used to calculate power
        duration_s = 0.001

        source = optics_cuda.LambertianSource(
            ###        source = optics_cuda.FatPencil(
            self._results._source_stage._size_m,
            #self._results._source_stage._size_m,
            self._results._source_stage._height_m,
            #source_wavelength_nm,
# far red is 350mw at 350mA
###            spectra.SourceSpectrum.LED_FAR_RED,
# photo  red is 425mw at 350mA
            spectra.SourceSpectrum.LED_PHOTO_RED,
            source_photons_per_bundle,
            duration_s,
        )
        photons = source.make_photons(self._bundles)
        #photons.debug(self._results._source_stage._size_m)
        self.record_results(self._results._source_stage, photons)

        # absorptive light box
        # lightbox = optics_cuda.Iris(
        # reflective light box
        lightbox = optics_cuda.Lightbox(
            height=self._results._box_stage._height_m,
            size=self._results._box_stage._size_m,
        )
        lightbox.propagate_without_kernel(photons)
        self.record_results(self._results._box_stage, photons)

        # Henyey isn't a very good approximation.
        # diffuser = optics_cuda.HenyeyGreensteinDiffuser(g=0.64, absorption=0.16)

        # Lambertian corresponds to white glass or wd008
        # diffuser = optics_cuda.LambertianDiffuser()

        diffuser = optics_cuda.AcryliteDiffuser_0d010()

        diffuser.diffuse(photons)
        self.record_results(self._results._diffuser_stage, photons)

        # propagate to the reflector and eliminate photons that miss it
        optics_cuda.propagate_to_reflector(
            photons, location=self._results._outbound_stage._height_m
        )
        photons.prune_outliers(self._results._outbound_stage._size_m)
        self.record_results(self._results._outbound_stage, photons)

        # reflect TODO: guess at absorption
###
# for ASTM E810, try a bit sharper distribution
#        reflector = optics_cuda.HenyeyGreensteinDiffuser(g=-0.9925, absorption=0.0)
# lots of absorption to make R_A=500 at 0.2deg
        reflector = optics_cuda.HenyeyGreensteinDiffuser(g=-0.995, absorption=0.84)
        reflector.diffuse(photons)
        self.record_results(self._results._inbound_stage, photons)

        # propagate to the camera and eliminate photons that miss it by a lot
        optics_cuda.propagate_to_camera(
            photons, location=self._results._camera_plane_stage._height_m
        )
        photons.prune_outliers(self._results._camera_plane_stage._size_m)

        self.record_results(self._results._camera_plane_stage, photons)

        # apply color filter
###
# skip absorption for ASTM E810 for now
        flt = spectra.FilterSpectrum.FILTER_27
        flt.absorb(photons.wavelength_nm, photons.alive)

        self.record_results(self._results._filter_stage, photons)

        # only look at what the camera will look at
        # TODO: clean this up somehow
        photons.prune_outliers2(
            self._camera._xmin,
            self._camera._xmax,
            self._camera._ymin,
            self._camera._ymax)
        self.record_results(self._results._camera_lens_stage, photons)

        self._camera.count(photons)

        timer.tick("one run")


class Study:
    """Sweep some parameters while measuring output."""

    def __init__(self):
        pass
