from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class Spectrum(Enum):
    def __init__(self, name, file_name):
        self._name = name
        df = pd.read_csv(
            f"spectra/{file_name}",
            delimiter="\t",
            converters={1: lambda x: float(x.strip("%")) / 100},
        )
        interpolated = interpolate.interp1d(
            df.iloc[:, 0],
            df.iloc[:, 1],
            fill_value=(df.iloc[0, 1], df.iloc[-1, 1]),
            bounds_error=False,
        )
        x_new = np.arange(360, 780, 0.1)
        y_new = interpolated(x_new)
        self._interpolated_df = pd.DataFrame({"nm": x_new, df.columns[1]: y_new})

    def pdf(self):
        return (self._interpolated_df.iloc[:, 0], self._interpolated_df.iloc[:, 1])


class Emitter(Spectrum):
    def emit(self, size) -> pd.Series:
        given_x: pd.Series = self._interpolated_df.iloc[:, 0]
        given_pdf: pd.Series = self._interpolated_df.iloc[:, 1]
        return SourceSpectrum.generate_rand_from_pdf(given_pdf, given_x, size)

    @staticmethod
    def generate_rand_from_pdf(pdf: pd.Series, x_grid: pd.Series, size) -> pd.Series:
        cdf = pdf.cumsum()
        cdf = cdf / cdf.iloc[-1]
        values = np.random.rand(size)
        value_bins = np.searchsorted(cdf, values)
        return x_grid[value_bins]

    def plot(self):
        (given_x, given_pdf) = self.pdf()
        scale_for_comparison = given_pdf.max()
        plt.plot(given_x, given_pdf, "-")
        samples = self.emit(1000)
        counts, bins = np.histogram(samples, 256)
        plt.hist(bins[:-1], bins, weights=counts * scale_for_comparison / counts.max())
        plt.title(self._name)
        plt.xlabel("wavelength (nm)")
        plt.xlim(360, 780)
        plt.ylim(0, 1)
        plt.show()

    def compare(self, flt):
        """plot the filtered and unfiltered camera response to this source"""
        # TODO: scale size to source brightness?
        size = 10000
        src_samples = self.emit(size)
        src_alive = np.ones(size).astype(bool)
        emit_count = np.sum(src_alive)

        src_alive = np.ones(size).astype(bool)
        src_filter = flt
        src_filter.absorb(src_samples, src_alive)
        CameraSpectrum.CAMERA_SEE_3_CAM.absorb(src_samples, src_alive)
        qe_filter_count = np.sum(src_alive)

        # fig = plt.figure(figsize=[15,10])
        # show the filter shape
        (given_x, given_pdf) = flt.pdf()
        plt.plot(given_x, 200 * given_pdf, "-", color="red", label="transmission")
        # before filtering
        h, b = np.histogram(src_samples, bins=256)
        plt.hist(b[:-1], b, weights=h, color="orange", label="source")
        # after filtering
        h, b = np.histogram(src_samples, bins=256, weights=1.0 * src_alive)
        plt.hist(b[:-1], b, weights=h, color="blue", label="filtered")
        plt.title(
            f"{self._name} + {flt._name} = {100*qe_filter_count/emit_count:.2f}%",
            fontsize=12,
        )
        plt.xlabel("wavelength (nm)")
        plt.xlim(360, 780)
        plt.legend()
        plt.show()


class Absorber(Spectrum):
    def __init__(self, name, file_name):
        super().__init__(name, file_name)
        self._rng = np.random.default_rng()

    def absorb(self, photon_wavelength, photon_alive):
        filter_idx = self._interpolated_df.iloc[:, 0].searchsorted(photon_wavelength)
        filter_value = self._interpolated_df.iloc[filter_idx, 1]
        fate = self._rng.random(photon_alive.size) < filter_value
        np.logical_and(photon_alive, fate, out=photon_alive)

    def plot(self):
        (given_x, given_pdf) = self.pdf()
        plt.plot(given_x, given_pdf, "-")
        plt.title(self._name)
        plt.xlabel("wavelength (nm)")
        plt.xlim(360, 780)
        plt.ylim(0, 1)
        plt.show()


class SourceSpectrum(Emitter):
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
    FILTER_27 = ("27 Medium Red", "Spectra - Roscolux 27.tsv")
    FILTER_389 = ("389 Chroma Green", "Spectra - Roscolux 389.tsv")
    FILTER_4600 = ("4600 Medium Red Blue", "Spectra - Roscolux 4600.tsv")
    FILTER_4954 = ("4954 Primary Green", "Spectra - Roscolux 4954.tsv")
    FILTER_6500 = ("6500 Primary Red", "Spectra - Roscolux 6500.tsv")
    FILTER_74 = ("74 Night Blue", "Spectra - Roscolux 74.tsv")


class CameraSpectrum(Absorber):
    """quantum efficiency of the camera"""

    CAMERA_SEE_3_CAM = ("Quantum Efficiency", "Spectra - QE.tsv")
