from enum import Enum
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants


class Spectrum(Enum):
    def __init__(self, name, file_name, converters=None):
        self._name = name
        if converters is None:
            converters={1: lambda x: float(x.strip("%")) / 100}
        df = pd.read_csv(
            f"spectra/{file_name}",
            delimiter="\t",
            converters=converters
        )
        self._x = cp.arange(360, 780, 0.1)
        # _y is expressed in energy terms
        self._y = cp.interp(self._x, cp.array(df.iloc[:,0].to_numpy()), cp.array(df.iloc[:,1].to_numpy()))

        # the above data are expressed in terms of power, not quanta (with
        # the exception of quantum efficiency which is handled below).
        # so convert the pdf to quanta:
        nm = self._x
        spectral_flux_j_s_nm = self._y

        wavelength_m = nm * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        # actually per bundle
        self._spectral_photon_flux_photons_s_nm = spectral_flux_j_s_nm / energy_per_photon_j


    def pdf(self):
        return (self._x, self._y)


class Emitter(Spectrum):
    def emit(self, size) -> cp.array:
        # photon emission is scaled by photon *energy*
        #return cp.array(Emitter.generate_rand_from_pdf(self._y, self._x, size))
        # so use the photon flux here instead.
        return cp.array(Emitter.generate_rand_from_pdf(self._spectral_photon_flux_photons_s_nm,
                                                       self._x, size))

    @staticmethod
    def generate_rand_from_pdf(pdf: cp.ndarray, x_grid: cp.ndarray, size) -> pd.Series:
        cdf = pdf.cumsum()
        cdf = cdf / cdf[-1]
        values = cp.random.rand(size)
        value_bins = cp.searchsorted(cdf, values)
        return x_grid[value_bins]

    def plot(self):
        (given_x, given_pdf) = self.pdf()
        scale_for_comparison = given_pdf.max()
        plt.plot(given_x.get(), given_pdf.get(), "-")
        samples = self.emit(1000)
        counts, bins = cp.histogram(samples, 256)
        weights = counts.get() * scale_for_comparison.item() / counts.max().item()
        plt.hist(bins[:-1].get(), bins.get(), weights = weights)
        plt.title(self._name)
        plt.xlabel("wavelength (nm)")
        plt.xlim(360, 780)
        plt.ylim(0, 1)
        plt.show()


class Absorber(Spectrum):
    def __init__(self, name, file_name, converters = None):
        super().__init__(name, file_name, converters)
        self._rng = cp.random.default_rng()

    def absorb(self, photon_wavelength:cp.ndarray, photon_alive:cp.ndarray):
        filter_idx = cp.searchsorted(self._x, photon_wavelength)
        # even though _y measures energy it actually measures the (spectral) ratio of output to input
        # which is the same no matter how you measure it.
        filter_value = self._y[filter_idx]
        fate = self._rng.random(photon_alive.size) < filter_value
        cp.logical_and(photon_alive, fate, out=photon_alive)

    def plot(self):
        (given_x, given_pdf) = self.pdf()
        plt.plot(given_x.get(), given_pdf.get(), "-")
        plt.title(self._name)
        plt.xlabel("wavelength (nm)")
        plt.xlim(360, 780)
        plt.ylim(0, 1)
        plt.show()

class LuminousEfficiency(Absorber):
    def __init__(self, name, file_name):
        super().__init__(name, file_name, converters={})
        
        wavelength_m = self._x * 1e-9
        frequency_hz = scipy.constants.c / wavelength_m
        energy_per_photon_j = scipy.constants.h * frequency_hz
        lumen_seconds_per_joule_per_nm = self._y * 683
        self._lumen_seconds_per_photon_per_nm = lumen_seconds_per_joule_per_nm * energy_per_photon_j

    def lumen_seconds(self, photon_wavelength_nm, photons_per_bundle, photon_alive):
        # actually per bundle
        idx = cp.searchsorted(self._x, photon_wavelength_nm)
        lumen_seconds_per_nm = self._lumen_seconds_per_photon_per_nm[idx] * photon_alive
        return cp.sum(lumen_seconds_per_nm) * photons_per_bundle


class SourceSpectrum(Emitter):
    # these curves describe spectral flux in terms of power produced per wavelength
    LED_AMBER = ("Amber", "Spectra - Cree xp-e2 amber.tsv")
    LED_BLUE = ("Blue", "Spectra - Cree xp-e2 blue.tsv")
    LED_COOL_WHITE = ("Cool White", "Spectra - Cree xp-e2 cool white.tsv")
    LED_FAR_RED = ("Far Red", "Spectra - Cree xp-e2 far red.tsv")
    LED_GREEN = ("Green", "Spectra - Cree xp-e2 green.tsv")
    LED_PC_AMBER = ("PC Amber", "Spectra - Cree xp-e2 PC amber.tsv")
    LED_PHOTO_RED = ("Photo Red", "Spectra - Cree xp-e2 photo red.tsv")
    LED_RED_ORANGE = ("Red-Orange", "Spectra - Cree xp-e2 red-orange.tsv")
    LED_RED = ("Red", "Spectra - Cree xp-e2 red.tsv")
    LED_ROYAL_BLUE = ("Royal Blue", "Spectra - Cree xp-e2 royal blue.tsv")
    LED_WARM_WHITE = ("Warm White", "Spectra - Cree xp-e2 warm white.tsv")


class FilterSpectrum(Absorber):
    # these curves describe spectral flux in terms of power absorbed per wavelength
    FILTER_27 = ("27 Medium Red", "Spectra - Roscolux 27.tsv")
    FILTER_389 = ("389 Chroma Green", "Spectra - Roscolux 389.tsv")
    FILTER_4600 = ("4600 Medium Red Blue", "Spectra - Roscolux 4600.tsv")
    FILTER_4954 = ("4954 Primary Green", "Spectra - Roscolux 4954.tsv")
    FILTER_6500 = ("6500 Primary Red", "Spectra - Roscolux 6500.tsv")
    FILTER_74 = ("74 Night Blue", "Spectra - Roscolux 74.tsv")

class Photopic(LuminousEfficiency):
    # This is the Sharpe, Stockman, Jagla & Gagle 2005 version, not the 1931 CIE version
    # so it is slightly more sensitive to blue: wikipedia.org/wiki/Luminous_efficiency_function
    # it is expressed as power per wavelength.
    PHOTOPIC = ("Photopic Response", 'Spectra - Photopic.tsv')

class CameraSpectrum(Absorber):
    # QE is the ratio of photons absorbed to electrons produced
    CAMERA_SEE_3_CAM = ("Quantum Efficiency", "Spectra - QE.tsv")




def compare_all():
    size = 10000
    scale = 200
    for src in SourceSpectrum:
        idx = 0
        for flt in FilterSpectrum:
            idx += 1
            if idx > 3:
                idx = 1
            if idx == 1: # new row
                fig = plt.figure(figsize=[30,5])
            ax = plt.subplot(1, len(FilterSpectrum), idx)

            """plot the filtered and unfiltered camera response to this source"""
            # TODO: scale size to source brightness?
            src_samples: cp.ndarray = src.emit(size)
            src_alive: cp.ndarray = cp.ones(size).astype(bool)
            emit_count = cp.sum(src_alive)

            src_alive = cp.ones(size).astype(bool)
            src_filter = flt
            src_filter.absorb(src_samples, src_alive)
            CameraSpectrum.CAMERA_SEE_3_CAM.absorb(src_samples, src_alive)
            qe_filter_count = cp.sum(src_alive)

            # show the filter shape
            (given_x, given_pdf) = flt.pdf()
            ax.plot(given_x.get(), scale * given_pdf.get(), "-", color="red", label="transmission")

            # before filtering
            h, b = cp.histogram(src_samples, bins=256)
            ax.hist(b[:-1].get(), b.get(), weights=h.get(), color="orange", label="source")

            # after filtering
            h, b = cp.histogram(src_samples, bins=256, weights=1.0 * src_alive)
            ax.hist(b[:-1].get(), b.get(), weights=h.get(), color="blue", label="filtered")
            ax.set_title(
                f"{src._name} + {flt._name} = {100*qe_filter_count/emit_count:.2f}%",
            )
            ax.set_xlabel("wavelength (nm)")
            ax.set_xlim(360, 780)
            ax.set_ylim(0, scale)
        plt.show()
