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
        source = optics_cuda.MonochromaticLambertianSource(
            source_size_m,
            source_size_m,
            source_wavelength_nm,
            source_photons_per_bundle,
            duration_s,
        )
        photons = source.make_photons(self._bundles)
        timer.tick("duration 1.0")
        photons.debug(source_size_m)
        timer.tick("duration 1.2")

        self._results._source_stage._photons_size += photons.count_alive()
        self._results._source_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._source_stage,
            x_min=-source_size_m / 2,
            x_max=source_size_m / 2,
            y_min=-source_size_m / 2,
            y_max=source_size_m / 2,
            z_min=-0.005,
            z_max=0.005,
            theta_max=np.pi / 2,
        )

        self._results._source_stage._sample.add(photons.sample())
        self._results._source_stage._ray_length = 0.0002
        self._results._source_stage._ray_color = 0xFF0000
        self._results._source_stage._box = [
            -source_size_m / 2,
            source_size_m / 2,
            -source_size_m / 2,
            source_size_m / 2,
            0,
        ]
        self._results._source_stage._box_color = 0x808080
        self._results._source_stage._label = "Source"
        timer.tick("duration 1.5")

        # propagate through the reflective light box
        lightbox_height_m = 0.04  # 4 cm
        lightbox_size_m = 0.04  # 4 cm
        lightbox = optics_cuda.Lightbox(height=lightbox_height_m, size=lightbox_size_m)
        lightbox.propagate_without_kernel(photons)
        timer.tick("duration 2.0")

        self._results._box_stage._photons_size += photons.count_alive()
        self._results._box_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._box_stage,
            x_min=-lightbox_size_m / 2,
            x_max=lightbox_size_m / 2,
            y_min=-lightbox_size_m / 2,
            y_max=lightbox_size_m / 2,
            z_min=0,
            z_max=0.1,
            theta_max=np.pi / 2,
        )
        self._results._box_stage._sample.add(photons.sample())
        self._results._box_stage._ray_length = 0.01
        self._results._box_stage._ray_color = 0xFF0000
        self._results._box_stage._box = [
            -lightbox_size_m / 2,
            lightbox_size_m / 2,
            -lightbox_size_m / 2,
            lightbox_size_m / 2,
            lightbox_height_m,
        ]
        self._results._box_stage._box_color = 0x808080
        self._results._box_stage._label = "Lightbox"

        timer.tick("duration 3.0")

        # diffuse through the diffuser
        diffuser = optics_cuda.Diffuser(g=np.float32(0.64), absorption=np.float32(0.16))
        diffuser.diffuse(photons)
        timer.tick("duration 3.2")

        self._results._diffuser_stage._photons_size += photons.count_alive()
        self._results._diffuser_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._diffuser_stage,
            x_min=-lightbox_size_m / 2,
            x_max=lightbox_size_m / 2,
            y_min=-lightbox_size_m / 2,
            y_max=lightbox_size_m / 2,
            z_min=0,
            z_max=0.1,
        )
        self._results._diffuser_stage._sample.add(photons.sample())
        self._results._diffuser_stage._ray_length = 0.01
        self._results._diffuser_stage._ray_color = 0xFF0000
        self._results._diffuser_stage._box = [
            -lightbox_size_m / 2,
            lightbox_size_m / 2,
            -lightbox_size_m / 2,
            lightbox_size_m / 2,
            lightbox_height_m,
        ]
        self._results._diffuser_stage._box_color = 0x808080
        self._results._diffuser_stage._label = "Diffuser"

        timer.tick("duration 3.5")

        # propagate to the reflector
        # TODO: make distance a parameter
        reflector_distance_m = np.float32(10)
        # reflector_distance_m = np.float32(5)
        # reflector_distance_m = np.float32(1)
        reflector_size_m = 0.1  # 10 cm
        optics_cuda.propagate_to_reflector(photons, location=reflector_distance_m)
        # eliminate photons that miss the reflector
        photons.prune_outliers(reflector_size_m)
        timer.tick("duration 4.0")

        self._results._outbound_stage._photons_size += photons.count_alive()
        self._results._outbound_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._outbound_stage,
            x_min=-reflector_size_m / 2,
            x_max=reflector_size_m / 2,
            y_min=-reflector_size_m / 2,
            y_max=reflector_size_m / 2,
            z_min=0,
            z_max=reflector_distance_m,
            theta_max=np.pi / 100,
        )  # a narrow beam
        self._results._outbound_stage._sample.add(photons.sample())
        self._results._outbound_stage._ray_length = 0.01
        self._results._outbound_stage._ray_color = 0xFF0000
        self._results._outbound_stage._box = [
            -reflector_size_m / 2,
            reflector_size_m / 2,
            -reflector_size_m / 2,
            reflector_size_m / 2,
            reflector_distance_m,
        ]
        self._results._outbound_stage._box_color = 0x808080
        self._results._outbound_stage._label = "Outbound"

        timer.tick("duration 4.5")

        # reflect
        # TODO: guess at absorption
        reflector = optics_cuda.Diffuser(
            g=np.float32(-0.9925), absorption=np.float32(0.0)
        )
        reflector.diffuse(photons)

        timer.tick("duration 5.0")

        self._results._inbound_stage._photons_size += photons.count_alive()
        self._results._inbound_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._inbound_stage,
            x_min=-reflector_size_m / 2,
            x_max=reflector_size_m / 2,
            y_min=-reflector_size_m / 2,
            y_max=reflector_size_m / 2,
            z_min=0,
            z_max=reflector_distance_m,
            theta_min=np.pi * 90 / 100,
        )
        self._results._inbound_stage._sample.add(photons.sample())
        self._results._inbound_stage._ray_length = 0.01
        self._results._inbound_stage._ray_color = 0xFF0000
        self._results._inbound_stage._box = [
            -reflector_size_m / 2,
            reflector_size_m / 2,
            -reflector_size_m / 2,
            reflector_size_m / 2,
            reflector_distance_m,
        ]
        self._results._inbound_stage._box_color = 0x808080
        self._results._inbound_stage._label = "Inbound"

        timer.tick("duration 5.5")

        # propagate to the camera
        # make the camera height even with the diffuser
        camera_distance_m = np.float32(lightbox_height_m)
        optics_cuda.propagate_to_camera(photons, location=camera_distance_m)
        # eliminate photons that miss the camera by a lot
        camera_neighborhood_m = 0.2
        photons.prune_outliers(camera_neighborhood_m)
        timer.tick("duration 6.0")

        self._results._camera_plane_stage._photons_size += photons.count_alive()
        self._results._camera_plane_stage._photons_energy_j += photons.energy_j()
        stats_cuda.histogram(
            photons,
            self._results._camera_plane_stage,
            x_min=-camera_neighborhood_m / 2,
            x_max=camera_neighborhood_m / 2,
            y_min=-camera_neighborhood_m / 2,
            y_max=camera_neighborhood_m / 2,
            z_min=0,
            z_max=camera_distance_m,
        )
        self._results._camera_plane_stage._sample.add(photons.sample())
        self._results._camera_plane_stage._ray_length = 0.01
        self._results._camera_plane_stage._ray_color = 0xFF0000
        # note offset camera, 1 cm square
        self._results._camera_plane_stage._box = [
            0.02,
            0.03,
            -0.0050,
            0.0050,
            camera_distance_m,
        ]
        self._results._camera_plane_stage._box_color = 0x0000FF
        self._results._camera_plane_stage._label = "Camera"
        timer.tick("duration 7.0")


class Study:
    """Sweep some parameters while measuring output."""

    def __init__(self):
        pass
