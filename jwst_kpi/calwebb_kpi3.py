from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


"""
JWST stage 3 pipeline for kernel phase imaging.

Authors: Jens Kammerer, Thomas Vandal, Katherine Thibault, Frantz Martinache
Supported instruments: NIRCam, NIRISS, MIRI
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects

from astroquery.svo_fps import SvoFps
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter
from xara import core, kpo

from . import utils as ut
from . import pupil_data

PUPIL_DIR = pupil_data.__path__[0]


# =============================================================================
# MAIN
# =============================================================================

# Load the NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile
# Service.
# http://svo2.cab.inta-csic.es/theory/fps/
wave_nircam = {}
weff_nircam = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRCAM")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".")+1:]
    wave_nircam[name] = filter_list["WavelengthMean"][i]/1e4 # micron
    weff_nircam[name] = filter_list["WidthEff"][i]/1e4 # micron
wave_niriss = {}
weff_niriss = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRISS")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".")+1:]
    wave_niriss[name] = filter_list["WavelengthMean"][i]/1e4 # micron
    weff_niriss[name] = filter_list["WidthEff"][i]/1e4 # micron
wave_miri = {}
weff_miri = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="MIRI")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".")+1:]
    wave_miri[name] = filter_list["WavelengthMean"][i]/1e4 # micron
    weff_miri[name] = filter_list["WidthEff"][i]/1e4 # micron
del filter_list

# Bad pixel flags.
# https://jwst-reffiles.stsci.edu/source/data_quality.html
pxdq_flags = {"DO_NOT_USE": 1,
              "SATURATED": 2,
              "JUMP_DET": 4,
              "DROPOUT": 8,
              "UNRELIABLE_ERROR": 256,
              "NON_SCIENCE": 512,
              "DEAD": 1024,
              "HOT": 2048,
              "WARM": 4096,
              "LOW_QE": 8192,
              "RC": 16384,
              "TELEGRAPH": 32768,
              "NONLINEAR": 65536,
              "BAD_REF_PIXEL": 131072,
              "NO_FLAT_FIELD": 262144,
              "NO_GAIN_VALUE": 524288,
              "NO_LIN_CORR": 1048576,
              "NO_SAT_CHECK": 2097152,
              "UNRELIABLE_BIAS": 4194304,
              "UNRELIABLE_DARK": 8388608,
              "UNRELIABLE_SLOPE": 16777216,
              "UNRELIABLE_FLAT": 33554432,
              "OPEN": 67108864,
              "ADJ_OPEN": 134217728,
              "UNRELIABLE_RESET": 268435456,
              "MSA_FAILED_OPEN": 536870912,
              "OTHER_BAD_PIXEL": 1073741824,
              }

# Detector pixel scales.
# TODO: assumes that NIRISS pixels are square but they are slightly
#       rectangular.
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview
pscale = {"NIRCAM_SHORT": 31.0, # mas
          "NIRCAM_LONG": 63.0, # mas
          "NIRISS": 65.55, # mas
          "MIRI": 110.0, # mas
          }

# Detector gains.
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview/niriss-detector-performance
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-performance
gain = {"NIRCAM_SHORT": 2.05, # e-/ADU
        "NIRCAM_LONG": 1.82, # e-/ADU
        "NIRISS": 1.61, # e-/ADU
        "MIRI": 4., # e-/ADU
        }


class Kpi3Pipeline:
    """
    JWST stage 3 pipeline for kernel phase imaging.

    ..Notes:: AMI skips the ipc, photom, and resample steps in the stage 1 & 2
              pipelines. It is recommended to also skip these steps for kernel
              phase imaging.
    """

    def __init__(self):
        """
        Initialize the pipeline.
        """

        # Initialize the pipeline steps.
        self.trim_frames = trim_frames()
        self.fix_bad_pixels = fix_bad_pixels()
        self.recenter_frames = recenter_frames()
        self.window_frames = window_frames()
        self.extract_kerphase = extract_kerphase()
        self.empirical_uncertainties = empirical_uncertainties()

        # Initialize the pipeline parameters.
        self.output_dir = None
        self.show_plots = False
        self.good_frames = None

    def run(self,
            file,
            ):
        """
        Run the pipeline.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        """

        # Make the output directory if it does not exist.
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # Run the pipeline steps if they are not skipped. For the kernel phase
        # extraction, run the recentering and windowing internally in complex
        # visibility space.
        suffix = ""
        if not self.trim_frames.skip:
            suffix = self.trim_frames.step(
                file,
                suffix,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )
        if not self.fix_bad_pixels.skip:
            suffix = self.fix_bad_pixels.step(
                file,
                suffix,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )
        if not self.recenter_frames.skip:
            suffix = self.recenter_frames.step(
                file,
                suffix,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )
        if not self.window_frames.skip:
            suffix = self.window_frames.step(
                file,
                suffix,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )
        if not self.extract_kerphase.skip:
            suffix = self.extract_kerphase.step(
                file,
                suffix,
                self.recenter_frames,
                self.window_frames,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )
        if not self.empirical_uncertainties.skip:
            suffix = self.empirical_uncertainties.step(
                file,
                suffix,
                output_dir=self.output_dir,
                show_plots=self.show_plots,
                good_frames=self.good_frames,
            )


class trim_frames:
    """
    Trim frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.trim_cent = None
        self.trim_halfsize = 32 # pix

    def step(self,
             file,
             suffix,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running trim frames step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        pxdq = hdul["DQ"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning("List of good frames needs to contain at least one element")
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning("List of good frames may only contain integer elements")
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning("Some of the provided good frames are outside the data range")

        # Suffix for the file path from the current step.
        suffix_out = "_trimmed"

        # Trim frames. Need to trim all frames (also bad ones) to match their
        # shapes.
        if self.trim_cent is None:
            ww_max = []
            for i in range(nf):
                ww_max += [np.unravel_index(np.nanargmax(median_filter(data[i], size=3)), data[i].shape)]
            ww_max = np.array(ww_max)
        else:
            ww_max = np.array([[self.trim_cent[0], self.trim_cent[1]]]*nf)
        if good_frames is None:
            if np.max(np.abs(ww_max-np.mean(ww_max, axis=0))) > 3.:
                raise UserWarning("Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position")
            ww_max = [int(round(np.median(ww_max[:, 0]))), int(round(np.median(ww_max[:, 1])))]
        else:
            if np.max(np.abs(ww_max[good_frames]-np.mean(ww_max[good_frames], axis=0))) > 3.:
                raise UserWarning("Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position")
            ww_max = [int(round(np.median(ww_max[good_frames, 0]))), int(round(np.median(ww_max[good_frames, 1])))]
        xh_max = min(sx - ww_max[1], ww_max[1])
        yh_max = min(sy - ww_max[0], ww_max[0])
        sh_max = min(xh_max, yh_max)
        self.trim_halfsize = min(sh_max, self.trim_halfsize)

        print("--> Trimming frames around (x, y) = (%.0f, %.0f) to a size of %.0fx%.0f pixels" % (ww_max[1], ww_max[0], 2 * self.trim_halfsize, 2 * self.trim_halfsize))

        for i in range(nf):
            data_trimmed = data[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()
            erro_trimmed = erro[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()
            pxdq_trimmed = pxdq[
                :,
                ww_max[0] - self.trim_halfsize : ww_max[0] + self.trim_halfsize,
                ww_max[1] - self.trim_halfsize : ww_max[1] + self.trim_halfsize,
            ].copy()

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(2, 2, figsize=(1.50 * 6.4, 1.50 * 4.8))
            if good_frames is None:
                p00 = ax[0, 0].imshow(np.log10(np.abs(data[0])), origin="lower")
            else:
                p00 = ax[0, 0].imshow(np.log10(np.abs(data[good_frames[0]])), origin="lower")
            plt.colorbar(p00, ax=ax[0, 0])
            r00 = Rectangle((ww_max[1] - self.trim_halfsize - 0.5, ww_max[0] - self.trim_halfsize - 0.5), 2 * self.trim_halfsize, 2 * self.trim_halfsize, facecolor='none', edgecolor='red')
            ax[0, 0].add_patch(r00)
            t00 = ax[0, 0].text(
                0.01,
                0.01,
                "center = %.0f, %.0f" % (ww_max[1], ww_max[0]),
                color="white",
                ha="left",
                va="bottom",
                transform=ax[0, 0].transAxes,
                size=12,
            )
            t00.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[0, 0].set_title(
                "Full frame (log)",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p01 = ax[0, 1].imshow(pxdq[0] & 1 == 1, origin="lower")
            else:
                p01 = ax[0, 1].imshow(pxdq[good_frames[0]] & 1 == 1, origin="lower")
            plt.colorbar(p01, ax=ax[0, 1])
            r01 = Rectangle((ww_max[1] - self.trim_halfsize - 0.5, ww_max[0] - self.trim_halfsize - 0.5), 2 * self.trim_halfsize, 2 * self.trim_halfsize, facecolor='none', edgecolor='red')
            ax[0, 1].add_patch(r01)
            t01 = ax[0, 1].text(
                0.01,
                0.01,
                "DO_NOT_USE",
                color="white",
                ha="left",
                va="bottom",
                transform=ax[0, 1].transAxes,
                size=12,
            )
            t01.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[0, 1].set_title(
                "Bad pixel map",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p10 = ax[1, 0].imshow(np.log10(np.abs(data_trimmed[0])), origin="lower")
            else:
                p10 = ax[1, 0].imshow(np.log10(np.abs(data_trimmed[good_frames[0]])), origin="lower")
            plt.colorbar(p10, ax=ax[1, 0])
            t10 = ax[1, 0].text(
                0.01,
                0.01,
                "center = %.0f, %.0f" % (ww_max[1], ww_max[0]),
                color="white",
                ha="left",
                va="bottom",
                transform=ax[1, 0].transAxes,
                size=12,
            )
            t10.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[1, 0].set_title(
                "Trimmed frame (log)",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p11 = ax[1, 1].imshow(pxdq_trimmed[0] & 1 == 1, origin="lower")
            else:
                p11 = ax[1, 1].imshow(pxdq_trimmed[good_frames[0]] & 1 == 1, origin="lower")
            plt.colorbar(p11, ax=ax[1, 1])
            t11 = ax[1, 1].text(
                0.01,
                0.01,
                "DO_NOT_USE",
                color="white",
                ha="left",
                va="bottom",
                transform=ax[1, 1].transAxes,
                size=12,
            )
            t11.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[1, 1].set_title(
                "Trimmed bad pixel map",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Trim frames step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data_trimmed = data_trimmed[0]
            erro_trimmed = erro_trimmed[0]
            pxdq_trimmed = pxdq_trimmed[0]
        hdul["SCI"].data = data_trimmed
        hdul["SCI"].header["CENT_X"] = ww_max[1]
        hdul["SCI"].header["CENT_Y"] = ww_max[0]
        hdul["ERR"].data = erro_trimmed
        hdul["ERR"].header["CENT_X"] = ww_max[1]
        hdul["ERR"].header["CENT_Y"] = ww_max[0]
        hdul["DQ"].data = pxdq_trimmed
        hdul["DQ"].header["CENT_X"] = ww_max[1]
        hdul["DQ"].header["CENT_Y"] = ww_max[0]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Trim frames step done")

        return suffix_out


class fix_bad_pixels:
    """
    Fix bad pixels.

    ..Notes:: References for the Fourier method:
              https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
              https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.bad_bits = ["DO_NOT_USE"]
        self.bad_bits_allowed = pxdq_flags.keys()
        self.method = "medfilt"
        self.method_allowed = ["medfilt", "fourier"]

    def step(self,
             file,
             suffix,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running fix bad pixels step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        pxdq = hdul["DQ"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning("List of good frames needs to contain at least one element")
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning("List of good frames may only contain integer elements")
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning("Some of the provided good frames are outside the data range")

        # Suffix for the file path from the current step.
        suffix_out = "_bpfixed"

        # Make bad pixel map.
        mask = pxdq < 0
        for i in range(len(self.bad_bits)):
            if self.bad_bits[i] not in self.bad_bits_allowed:
                raise UserWarning("Unknown data quality flag")
            else:
                pxdq_flag = pxdq_flags[self.bad_bits[i]]
                mask = mask | (pxdq & pxdq_flag == pxdq_flag)
                if i == 0:
                    bb = self.bad_bits[i]
                else:
                    bb += ", " + self.bad_bits[i]

        print("--> Found %.0f bad pixels (%.2f%%)" % (np.sum(mask[good_frames]), np.sum(mask[good_frames]) / np.prod(mask[good_frames].shape) * 100.))

        # Fix bad pixels.
        data_bpfixed = data.copy()
        erro_bpfixed = erro.copy()
        if self.method not in self.method_allowed:
            raise UserWarning("Unknown bad pixel cleaning method")
        else:
            if self.method == "medfilt":
                for i in range(nf):
                    if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                        data_bpfixed[i][mask[i]] = median_filter(data_bpfixed[i], size=5)[mask[i]]
                        erro_bpfixed[i][mask[i]] = median_filter(erro_bpfixed[i], size=5)[mask[i]]
            elif self.method == "fourier":
                raise NotImplementedError("Fourier bad pixel cleaning method not implemented yet")

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 3, figsize=(2.25 * 6.4, 0.75 * 4.8))
            if good_frames is None:
                p0 = ax[0].imshow(mask[0], origin="lower")
            else:
                p0 = ax[0].imshow(mask[good_frames[0]], origin="lower")
            plt.colorbar(p0, ax=ax[0])
            t0 = ax[0].text(
                0.01,
                0.01,
                bb,
                color="white",
                ha="left",
                va="bottom",
                transform=ax[0].transAxes,
                size=12,
            )
            t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[0].set_title(
                "Bad pixel map",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p1 = ax[1].imshow(np.log10(np.abs(data[0])), origin="lower")
            else:
                p1 = ax[1].imshow(np.log10(np.abs(data[good_frames[0]])), origin="lower")
            plt.colorbar(p1, ax=ax[1])
            ax[1].set_title(
                "Frame (log)",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed[0])), origin="lower")
            else:
                p2 = ax[2].imshow(np.log10(np.abs(data_bpfixed[good_frames[0]])), origin="lower")
            plt.colorbar(p2, ax=ax[2])
            t2 = ax[2].text(
                0.01,
                0.01,
                "method = " + self.method,
                color="white",
                ha="left",
                va="bottom",
                transform=ax[2].transAxes,
                size=12,
            )
            t2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[2].set_title(
                "Fixed frame (log)",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Fix bad pixels step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data_bpfixed = data_bpfixed[0]
            erro_bpfixed = erro_bpfixed[0]
            mask = mask[0]
        hdul["SCI"].data = data_bpfixed
        hdul["SCI"].header["FIX_METH"] = self.method
        hdul["ERR"].data = erro_bpfixed
        hdul["ERR"].header["FIX_METH"] = self.method
        hdu_dq_mod = pyfits.ImageHDU(mask.astype("uint32"))
        hdu_dq_mod.header["EXTNAME"] = "DQ-MOD"
        hdu_dq_mod.header["BAD_BITS"] = bb
        hdul += [hdu_dq_mod]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Fix bad pixels step done")

        return suffix_out


class recenter_frames:
    """
    Recenter frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.method = "FPNM"
        self.method_allowed = ["BCEN", "COGI", "FPNM"]
        self.instrume_allowed = ["NIRCAM", "NIRISS", "MIRI"]
        self.bmax = 6. # m
        self.pupil_path = None
        self.verbose = False

    def step(self,
             file,
             suffix,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running recenter frames step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning("List of good frames needs to contain at least one element")
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning("List of good frames may only contain integer elements")
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning("Some of the provided good frames are outside the data range")
        INSTRUME = hdul[0].header["INSTRUME"]
        if INSTRUME not in self.instrume_allowed:
            raise UserWarning("Unsupported instrument")
        FILTER = hdul[0].header["FILTER"]

        # Suffix for the file path from the current step.
        suffix_out = "_recentered"

        # Simple background subtraction to avoid discontinuity when zero-padding the data in XARA.
        data -= np.median(data, axis=(1, 2), keepdims=True)

        # Get detector pixel scale and position angle.
        if INSTRUME == "NIRCAM":
            CHANNEL = hdul[0].header["CHANNEL"]
            PSCALE = pscale[INSTRUME + "_" + CHANNEL] # mas
        else:
            PSCALE = pscale[INSTRUME] # mas
        V3I_YANG = -hdul["SCI"].header["V3I_YANG"] * hdul["SCI"].header["VPARITY"] # deg, counter-clockwise

        if "FPNM" in self.method:

            # Check if the filter is known.
            if INSTRUME == "NIRCAM":
                filter_allowed = wave_nircam.keys()
            elif INSTRUME == "NIRISS":
                filter_allowed = wave_niriss.keys()
            elif INSTRUME == "MIRI":
                filter_allowed = wave_miri.keys()
            if FILTER not in filter_allowed:
                raise UserWarning("Unknown filter")

            # Get pupil model path and filter properties.
            if INSTRUME == "NIRCAM":
                if self.pupil_path is None:
                    if hdul[0].header["PUPIL"] == "MASKRND":
                        default_pupil_model = "nircam_rnd_pupil.fits"
                    elif hdul[0].header["PUPIL"] == "MASKBAR":
                        default_pupil_model = "nircam_bar_pupil.fits"
                    else:
                        default_pupil_model = "nircam_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_nircam[FILTER] * 1e-6 # m
                weff = weff_nircam[FILTER] * 1e-6 # m
            elif INSTRUME == "NIRISS":
                if self.pupil_path is None:
                    if hdul[0].header["PUPIL"] == "NRM":
                        default_pupil_model = "niriss_nrm_pupil.fits"
                    else:
                        default_pupil_model = "niriss_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_niriss[FILTER] * 1e-6 # m
                weff = weff_niriss[FILTER] * 1e-6 # m
            elif INSTRUME == "MIRI":
                if self.pupil_path is None:
                    default_pupil_model = "miri_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_miri[FILTER] * 1e-6 # m
                weff = weff_miri[FILTER] * 1e-6 # m
            
            # print("Rotating pupil model by %.2f deg (counter-clockwise)" % V3I_YANG)

            # # Rotate pupil model.
            # theta = np.deg2rad(V3I_YANG) # rad
            # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # hdul_pup = pyfits.open(self.pupil_path)
            # xxc = hdul_pup["APERTURE"].data["XXC"]
            # yyc = hdul_pup["APERTURE"].data["YYC"]
            # trm = hdul_pup["APERTURE"].data["TRM"]
            # hdul_pup.close()
            # txt = ""
            # for i in range(len(trm)):
            #     temp = rot.dot(np.array([xxc[i], yyc[i]]))
            #     txt += "%+.5f %+.5f %.5f\n" % (temp[0], temp[1], trm[i])
            # txtfile = open("pupil_model.txt", "w")
            # txtfile.write(txt)
            # txtfile.close()

            # Load pupil model.
            KPO = kpo.KPO(
                fname=self.pupil_path,
                array=None,
                ndgt=5,
                bmax=self.bmax,
                hexa=False,
                ID="",
            )
            m2pix = core.mas2rad(PSCALE) * sx / wave
            # KPO.kpi.plot_pupil_and_uv()
            # plt.show()

        else:
            KPO = None
            m2pix = None

        # Recenter frames.
        if self.method not in self.method_allowed:
            raise UserWarning("Unknown recentering method")
        else:
            data_recentered = []
            erro_recentered = []
            dx = [] # pix
            dy = [] # pix
            for i in range(nf):
                if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                    temp = core.recenter(
                        data[i],
                        algo=self.method,
                        subpix=True,
                        between=False,
                        verbose=False,
                        return_center=True,
                        dxdy=None,
                        mykpo=KPO,
                        m2pix=m2pix,
                        bmax=self.bmax,
                    )
                    data_recentered += [temp[0]]
                    dx += [temp[1]]
                    dy += [temp[2]]
                    temp = core.recenter(
                        erro[i],
                        algo=self.method,
                        subpix=True,
                        between=False,
                        verbose=False,
                        return_center=False,
                        dxdy=(dx[-1], dy[-1]),
                        mykpo=KPO,
                        m2pix=m2pix,
                        bmax=self.bmax,
                    )
                    erro_recentered += [temp]
                else:
                    data_recentered += [data[i]]
                    dx += [0.]
                    dy += [0.]
                    erro_recentered += [erro[i]]

                if self.verbose:
                    print("Image shift = (%.2f, %.2f)" % (dx[-1], dy[-1]))
            data_recentered = np.array(data_recentered)
            erro_recentered = np.array(erro_recentered)

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 2, figsize=(1.50 * 6.4, 0.75 * 4.8))
            if good_frames is None:
                p0 = ax[0].imshow(data[0], origin="lower")
            else:
                p0 = ax[0].imshow(data[good_frames[0]], origin="lower")
            plt.colorbar(p0, ax=ax[0])
            if good_frames is None:
                ax[0].axvline(sx // 2 + dx[0], color="red")
                ax[0].axhline(sy // 2 + dy[0], color="red")
                t0 = ax[0].text(
                    0.01,
                    0.01,
                    "center = %.2f, %.2f" % (sx // 2 + dx[0], sy // 2 + dy[0]),
                    color="white",
                    ha="left",
                    va="bottom",
                    transform=ax[0].transAxes,
                    size=12,
                )
            else:
                ax[0].axvline(sx // 2 + dx[good_frames[0]], color="red")
                ax[0].axhline(sy // 2 + dy[good_frames[0]], color="red")
                t0 = ax[0].text(
                    0.01,
                    0.01,
                    "center = %.2f, %.2f" % (sx // 2 + dx[good_frames[0]], sy // 2 + dy[good_frames[0]]),
                    color="white",
                    ha="left",
                    va="bottom",
                    transform=ax[0].transAxes,
                    size=12,
                )
            t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[0].set_title(
                "Frame",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p1 = ax[1].imshow(data_recentered[0], origin="lower")
            else:
                p1 = ax[1].imshow(data_recentered[good_frames[0]], origin="lower")
            plt.colorbar(p1, ax=ax[1])
            ax[1].axvline(sx // 2, color="red")
            ax[1].axhline(sy // 2, color="red")
            t1 = ax[1].text(
                0.01,
                0.01,
                "center = %.2f, %.2f" % (sx // 2, sy // 2),
                color="white",
                ha="left",
                va="bottom",
                transform=ax[1].transAxes,
                size=12,
            )
            t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[1].set_title(
                "Recentered frame",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Recenter frames step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data = data[0]
            erro = erro[0]
            data_recentered = data_recentered[0]
            erro_recentered = erro_recentered[0]
        hdu_sci_org = pyfits.ImageHDU(data)
        hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
        hdu_err_org = pyfits.ImageHDU(erro)
        hdu_err_org.header["EXTNAME"] = "ERR-ORG"
        hdul += [hdu_sci_org, hdu_err_org]
        hdul["SCI"].data = data_recentered
        hdul["ERR"].data = erro_recentered
        if is2d:
            xsh = pyfits.Column(name="XSHIFT", format="D", array=np.array([dx])) # pix
            ysh = pyfits.Column(name="YSHIFT", format="D", array=np.array([dy])) # pix
        else:
            xsh = pyfits.Column(name="XSHIFT", format="D", array=np.array(dx)) # pix
            ysh = pyfits.Column(name="YSHIFT", format="D", array=np.array(dy)) # pix
        hdu_ims = pyfits.BinTableHDU.from_columns([xsh, ysh])
        hdu_ims.header["EXTNAME"] = "IMSHIFT"
        hdul += [hdu_ims]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Recenter frames step done")

        return suffix_out


class window_frames:
    """
    Window frames.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.wrad = 24 # pix

    def step(self,
             file,
             suffix,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running window frames step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning("List of good frames needs to contain at least one element")
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning("List of good frames may only contain integer elements")
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning("Some of the provided good frames are outside the data range")

        # Suffix for the file path from the current step.
        suffix_out = "_windowed"

        # Window.
        if self.wrad is None:
            self.wrad = 24

        print("--> Windowing frames with super-Gaussian mask with radius %.0f pixels" % self.wrad)

        data_windowed = data.copy()
        erro_windowed = erro.copy()
        sgmask = core.super_gauss(sy, sx, self.wrad, between_pix=False)
        if good_frames is None:
            data_windowed *= sgmask
            erro_windowed *= sgmask
        else:
            for i in range(nf):
                if i in good_frames or (nf - i) * (-1) in good_frames:
                    data_windowed[i] *= sgmask
                    erro_windowed[i] *= sgmask

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            plt.ioff()
            f, ax = plt.subplots(1, 2, figsize=(1.50 * 6.4, 0.75 * 4.8))
            if good_frames is None:
                p0 = ax[0].imshow(data_windowed[0], origin="lower")
            else:
                p0 = ax[0].imshow(data_windowed[good_frames[0]], origin="lower")
            plt.colorbar(p0, ax=ax[0])
            c0 = plt.Circle(
                (sx // 2, sy // 2),
                self.wrad,
                color="red",
                ls="--",
                fill=False,
            )
            ax[0].add_patch(c0)
            t0 = ax[0].text(
                0.01,
                0.01,
                "wrad = %.0f pix" % self.wrad,
                color="white",
                ha="left",
                va="bottom",
                transform=ax[0].transAxes,
                size=12,
            )
            t0.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[0].set_title(
                "Windowed frame",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            if good_frames is None:
                p1 = ax[1].imshow(np.log10(np.abs(data_windowed[0])), origin="lower")
            else:
                p1 = ax[1].imshow(np.log10(np.abs(data_windowed[good_frames[0]])), origin="lower")
            plt.colorbar(p1, ax=ax[1])
            c1 = plt.Circle(
                (sx // 2, sy // 2),
                self.wrad,
                color="red",
                ls="--",
                fill=False,
            )
            ax[1].add_patch(c1)
            t1 = ax[1].text(
                0.01,
                0.01,
                "wrad = %.0f pix" % self.wrad,
                color="white",
                ha="left",
                va="bottom",
                transform=ax[1].transAxes,
                size=12,
            )
            t1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])
            ax[1].set_title(
                "Windowed frame (log)",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Window frames step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data = data[0]
            erro = erro[0]
            data_windowed = data_windowed[0]
            erro_windowed = erro_windowed[0]
        try:
            hdul.index_of("SCI-ORG")
        except:
            hdu_sci_org = pyfits.ImageHDU(data)
            hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
            hdu_err_org = pyfits.ImageHDU(erro)
            hdu_err_org.header["EXTNAME"] = "ERR-ORG"
            hdul += [hdu_sci_org, hdu_err_org]
        hdul["SCI"].data = data_windowed
        hdul["SCI"].header["WRAD"] = self.wrad
        hdul["ERR"].data = erro_windowed
        hdul["ERR"].header["WRAD"] = self.wrad
        hdu_win = pyfits.ImageHDU(sgmask)
        hdu_win.header["EXTNAME"] = "WINMASK"
        hdu_win.header["WRAD"] = self.wrad
        hdul += [hdu_win]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Window frames step done")

        return suffix_out


class extract_kerphase:
    """
    Extract the kernel phase while re-centering in complex visibility space.

    The KPFITS file structure has been agreed upon by the participants of
    Steph Sallum's masking & kernel phase hackathon in 2021 and is defined
    here:
        https://ui.adsabs.harvard.edu/abs/2022arXiv221017528K/abstract
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.instrume_allowed = ["NIRCAM", "NIRISS", "MIRI"]
        self.bmax = None # m
        self.pupil_path = None
        self.verbose = False

    def step(self,
             file,
             suffix,
             recenter_frames_obj,
             window_frames_obj,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        recenter_frames_obj: obj
            Object of recenter_frames class.
        window_frames_obj: obj
            Object of window_frames class.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running extract kerphase step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = hdul["SCI"].data
        erro = hdul["ERR"].data
        pxdq = hdul["DQ"].data
        try:
            data_org = hdul["SCI-ORG"].data
            erro_org = hdul["ERR-ORG"].data
        except:
            pass
        try:
            pxdq_mod = hdul["DQ-MOD"].data
        except:
            pass
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
            try:
                data_org = data_org[np.newaxis]
                erro_org = erro_org[np.newaxis]
            except:
                pass
            try:
                pxdq_mod = pxdq_mod[np.newaxis]
            except:
                pass
        else:
            is2d = False
        nf, sy, sx = data.shape
        if good_frames is not None:
            if len(good_frames) < 1:
                raise UserWarning("List of good frames needs to contain at least one element")
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning("List of good frames may only contain integer elements")
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning("Some of the provided good frames are outside the data range")
        INSTRUME = hdul[0].header["INSTRUME"]
        if INSTRUME not in self.instrume_allowed:
            raise UserWarning("Unsupported instrument")
        FILTER = hdul[0].header["FILTER"]

        # Suffix for the file path from the current step.
        suffix_out = "_kpfits"

        # Get detector pixel scale and position angle.
        if INSTRUME == "NIRCAM":
            CHANNEL = hdul[0].header["CHANNEL"]
            PSCALE = pscale[INSTRUME + "_" + CHANNEL] # mas
        else:
            PSCALE = pscale[INSTRUME] # mas
        V3I_YANG = -hdul["SCI"].header["V3I_YANG"] * hdul["SCI"].header["VPARITY"] # deg, counter-clockwise

        # Check if the filter is known.
        if INSTRUME == "NIRCAM":
            filter_allowed = wave_nircam.keys()
        elif INSTRUME == "NIRISS":
            filter_allowed = wave_niriss.keys()
        elif INSTRUME == "MIRI":
            filter_allowed = wave_miri.keys()
        if FILTER not in filter_allowed:
            raise UserWarning("Unknown filter")

        # Get pupil model path and filter properties.
        if INSTRUME == "NIRCAM":
            if self.pupil_path is None:
                if hdul[0].header["PUPIL"] == "MASKRND":
                    default_pupil_model = "nircam_rnd_pupil.fits"
                elif hdul[0].header["PUPIL"] == "MASKBAR":
                    default_pupil_model = "nircam_bar_pupil.fits"
                else:
                    default_pupil_model = "nircam_clear_pupil.fits"
                self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
            wave = wave_nircam[FILTER] * 1e-6 # m
            weff = weff_nircam[FILTER] * 1e-6 # m
        elif INSTRUME == "NIRISS":
            if self.pupil_path is None:
                if hdul[0].header["PUPIL"] == "NRM":
                    default_pupil_model = "niriss_nrm_pupil.fits"
                else:
                    default_pupil_model = "niriss_clear_pupil.fits"
                self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
            wave = wave_niriss[FILTER] * 1e-6 # m
            weff = weff_niriss[FILTER] * 1e-6 # m
        elif INSTRUME == "MIRI":
            if self.pupil_path is None:
                default_pupil_model = "miri_clear_pupil.fits"
                self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
            wave = wave_miri[FILTER] * 1e-6 # m
            weff = weff_miri[FILTER] * 1e-6 # m

        # print("Rotating pupil model by %.2f deg (counter-clockwise)" % V3I_YANG)

        # # Rotate pupil model.
        # theta = np.deg2rad(V3I_YANG) # rad
        # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # hdul_pup = pyfits.open(self.pupil_path)
        # xxc = hdul_pup["APERTURE"].data["XXC"]
        # yyc = hdul_pup["APERTURE"].data["YYC"]
        # trm = hdul_pup["APERTURE"].data["TRM"]
        # hdul_pup.close()
        # txt = ""
        # for i in range(len(trm)):
        #     temp = rot.dot(np.array([xxc[i], yyc[i]]))
        #     txt += "%+.5f %+.5f %.5f\n" % (temp[0], temp[1], trm[i])
        # txtfile = open("pupil_model.txt", "w")
        # txtfile.write(txt)
        # txtfile.close()

        # Load pupil model.
        KPO = kpo.KPO(
            fname=self.pupil_path,
            array=None,
            ndgt=5,
            bmax=self.bmax,
            hexa=False,
            ID="",
        )
        m2pix = core.mas2rad(PSCALE) * sx / wave
        # KPO.kpi.plot_pupil_and_uv()
        # plt.show()

        # Recenter frames, window frames, and extract kernel phase.
        if recenter_frames_obj.skip and window_frames_obj.skip:
            if "SCI-ORG" in hdul:
                raise UserWarning("There is a SCI-ORG FITS file extension although both the recentering and the windowing steps were skipped")
            for i in range(nf):
                if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                    KPO.extract_KPD_single_frame(
                        data[i],
                        PSCALE,
                        wave,
                        target=None,
                        recenter=False,
                        wrad=None,
                        method="LDFT1",
                    )
        elif recenter_frames_obj.skip and not window_frames_obj.skip:
            for i in range(nf):
                if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                    KPO.extract_KPD_single_frame(
                        data_org[i],
                        PSCALE,
                        wave,
                        target=None,
                        recenter=False,
                        wrad=window_frames_obj.wrad,
                        method="LDFT1",
                    )
        elif not recenter_frames_obj.skip and window_frames_obj.skip:
            dx = [] # pix
            dy = [] # pix
            for i in range(nf):
                if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                    temp = KPO.extract_KPD_single_frame(
                        data_org[i],
                        PSCALE,
                        wave,
                        target=None,
                        recenter=True,
                        wrad=None,
                        method="LDFT1",
                        algo_cent=recenter_frames_obj.method,
                        bmax_cent=recenter_frames_obj.bmax,
                    )
                    dx += [temp[0]]
                    dy += [temp[1]]
        else:
            dx = [] # pix
            dy = [] # pix
            for i in range(nf):
                if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                    temp = KPO.extract_KPD_single_frame(
                        data_org[i],
                        PSCALE,
                        wave,
                        target=None,
                        recenter=True,
                        wrad=window_frames_obj.wrad,
                        method="LDFT1",
                        algo_cent=recenter_frames_obj.method,
                        bmax_cent=recenter_frames_obj.bmax,
                    )
                    dx += [temp[0]]
                    dy += [temp[1]]

        # Extract kernel phase covariance. See:
        # https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
        kpcov = []
        kpsig = []
        kpcor = []
        for i in range(nf):
            if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                frame = data[i].copy()
                varframe = erro[i].copy() ** 2
                B = KPO.kpi.KPM.dot(np.divide(KPO.FF.imag.T, np.abs(KPO.FF.dot(frame.flatten()))).T)
                kpcov += [np.multiply(B, varframe.flatten()).dot(B.T)]
                kpsig += [np.sqrt(np.diag(kpcov[-1]))]
                kpcor += [np.true_divide(kpcov[-1], kpsig[-1][:, None] * kpsig[-1][None, :])]
        kpcov = np.array(kpcov)
        kpsig = np.array(kpsig)
        kpcor = np.array(kpcor)

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.ioff()
            f, ax = plt.subplots(2, 2, figsize=(1.50 * 6.4, 1.50 * 4.8))
            if good_frames is None:
                d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data[0])))
            else:
                d00 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data[good_frames[0]])))
            p00 = ax[0, 0].imshow(np.angle(d00), origin="lower", vmin=-np.pi, vmax=np.pi)
            c00 = plt.colorbar(p00, ax=ax[0, 0])
            c00.set_label("Fourier phase [rad]", rotation=270, labelpad=20)
            xx = KPO.kpi.UVC[:, 0] * m2pix + sx // 2
            yy = KPO.kpi.UVC[:, 1] * m2pix + sy // 2
            ax[0, 0].scatter(xx, yy, s=0.2, c="red")
            xx = -KPO.kpi.UVC[:, 0] * m2pix + sx // 2
            yy = -KPO.kpi.UVC[:, 1] * m2pix + sy // 2
            ax[0, 0].scatter(xx, yy, s=0.2, c="red")
            ax[0, 0].set_title(
                "Fourier phase",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            p01 = ax[0, 1].imshow(kpcor[0], origin="lower", cmap="RdBu", vmin=-1., vmax=1.)
            c01 = plt.colorbar(p01, ax=ax[0, 1])
            c01.set_label("Correlation", rotation=270, labelpad=20)
            ax[0, 1].set_title(
                "Kernel phase correlation",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ww = np.argsort(KPO.kpi.BLEN)
            ax[1, 0].plot(np.angle(KPO.CVIS[0][0, ww]))
            ax[1, 0].axhline(0., ls="--", color="black")
            ax[1, 0].set_ylim([-np.pi, np.pi])
            ax[1, 0].grid(axis="y")
            ax[1, 0].set_xlabel("Index sorted by baseline length")
            ax[1, 0].set_ylabel("Fourier phase [rad]")
            ax[1, 0].set_title(
                "Fourier phase",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ax[1, 1].errorbar(np.arange(KPO.KPDT[0][0, :].shape[0]), KPO.KPDT[0][0, :], yerr=kpsig[0], color=colors[0], alpha=1. / 3.)
            ax[1, 1].plot(np.arange(KPO.KPDT[0][0, :].shape[0]), KPO.KPDT[0][0, :], color=colors[0])
            ax[1, 1].axhline(0., ls="--", color="black")
            ylim = ax[1, 1].get_ylim()
            ax[1, 1].set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
            ax[1, 1].grid(axis="y")
            ax[1, 1].set_xlabel("Index")
            ax[1, 1].set_ylabel("Kernel phase [rad]")
            ax[1, 1].set_title(
                "Kernel phase",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            plt.suptitle("Extract kerphase step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        data_good = []
        erro_good = []
        pxdq_good = []
        data_org_good = []
        erro_org_good = []
        pxdq_mod_good = []
        for i in range(nf):
            if good_frames is None or i in good_frames or (nf - i) * (-1) in good_frames:
                data_good += [data[i]]
                erro_good += [erro[i]]
                pxdq_good += [pxdq[i]]
                try:
                    data_org_good += [data_org[i]]
                    erro_org_good += [erro_org[i]]
                except:
                    pass
                try:
                    pxdq_mod_good += [pxdq_mod[i]]
                except:
                    pass
        data_good = np.array(data_good)
        erro_good = np.array(erro_good)
        pxdq_good = np.array(pxdq_good)
        data_org_good = np.array(data_org_good)
        erro_org_good = np.array(erro_org_good)
        pxdq_mod_good = np.array(pxdq_mod_good)
        hdul[0].data = data_good[:, np.newaxis, :, :]
        hdul["SCI"].data = data_good[:, np.newaxis, :, :]
        hdul["ERR"].data = erro_good[:, np.newaxis, :, :]
        hdul["DQ"].data = pxdq_good[:, np.newaxis, :, :]
        try:
            hdul["SCI-ORG"].data = data_org_good[:, np.newaxis, :, :]
            hdul["ERR-ORG"].data = erro_org_good[:, np.newaxis, :, :]
        except:
            pass
        try:
            hdul["DQ-MOD"].data = pxdq_mod_good[:, np.newaxis, :, :]
        except:
            pass
        hdul[0].header["PSCALE"] = PSCALE # mas
        if INSTRUME == "NIRCAM":
            hdul[0].header["GAIN"] = gain[INSTRUME + "_" + CHANNEL] # e-/ADU
        else:
            hdul[0].header["GAIN"] = gain[INSTRUME] # e-/ADU
        hdul[0].header["DIAM"] = 6.559348 # m (flat-to-flat)
        hdul[0].header["EXPTIME"] = hdul[0].header["EFFINTTM"] # s
        hdul[0].header["DATEOBS"] = hdul[0].header["DATE-OBS"] + "T" + hdul[0].header["TIME-OBS"] # YYYY-MM-DDTHH:MM:SS.MMM
        hdul[0].header["PROCSOFT"] = "CALWEBB_KPI3"
        try:
            hdul[0].header["WRAD"] = hdul["WINMASK"].header["WRAD"] # pix
        except:
            hdul[0].header["WRAD"] = "NONE"
        hdul[0].header["CALFLAG"] = "False"
        hdul[0].header["CONTENT"] = "KPFITS1"
        xy1 = pyfits.Column(name="XXC", format="D", array=KPO.kpi.VAC[:, 0]) # m
        xy2 = pyfits.Column(name="YYC", format="D", array=KPO.kpi.VAC[:, 1]) # m
        trm = pyfits.Column(name="TRM", format="D", array=KPO.kpi.TRM)
        hdu_ape = pyfits.BinTableHDU.from_columns([xy1, xy2, trm])
        hdu_ape.header["EXTNAME"] = "APERTURE"
        hdu_ape.header["TTYPE1"] = ("XXC", "Virtual aperture x-coord (meter)")
        hdu_ape.header["TTYPE2"] = ("YYC", "Virtual aperture y-coord (meter)")
        hdu_ape.header["TTYPE3"] = ("TRM", "Virtual aperture transmission (0 < t <= 1)")
        hdul += [hdu_ape]
        uv1 = pyfits.Column(name="UUC", format="D", array=KPO.kpi.UVC[:, 0]) # m
        uv2 = pyfits.Column(name="VVC", format="D", array=KPO.kpi.UVC[:, 1]) # m
        red = pyfits.Column(name="RED", format="I", array=KPO.kpi.RED)
        hdu_uvp = pyfits.BinTableHDU.from_columns([uv1, uv2, red])
        hdu_uvp.header["EXTNAME"] = "UV-PLANE"
        hdu_uvp.header["TTYPE1"] = ("UUC", "Baseline u-coord (meter)")
        hdu_uvp.header["TTYPE2"] = ("VVC", "Baseline v-coord (meter)")
        hdu_uvp.header["TTYPE3"] = ("RED", "Baseline redundancy (int)")
        hdul += [hdu_uvp]
        hdu_kpm = pyfits.ImageHDU(KPO.kpi.KPM)
        hdu_kpm.header["EXTNAME"] = "KER-MAT"
        hdul += [hdu_kpm]
        hdu_blm = pyfits.ImageHDU(np.diag(KPO.kpi.RED).dot(KPO.kpi.TFM))
        hdu_blm.header["EXTNAME"] = "BLM-MAT"
        hdul += [hdu_blm]
        hdu_kpd = pyfits.ImageHDU(np.array(KPO.KPDT)) # rad
        hdu_kpd.header["EXTNAME"] = "KP-DATA"
        hdul += [hdu_kpd]
        hdu_kpe = pyfits.ImageHDU(kpsig[:, np.newaxis, :]) # rad
        hdu_kpe.header["EXTNAME"] = "KP-SIGM"
        hdul += [hdu_kpe]
        hdu_kpc = pyfits.ImageHDU(kpcov[:, np.newaxis, :]) # rad^2
        hdu_kpc.header["EXTNAME"] = "KP-COV"
        hdul += [hdu_kpc]
        cwavel = pyfits.Column(name="CWAVEL", format="D", array=np.array([wave])) # m
        bwidth = pyfits.Column(name="BWIDTH", format="D", array=np.array([weff])) # m
        hdu_lam = pyfits.BinTableHDU.from_columns([cwavel, bwidth])
        hdu_lam.header["EXTNAME"] = "CWAVEL"
        hdul += [hdu_lam]
        hdu_ang = pyfits.ImageHDU(np.array([hdul["SCI"].header["ROLL_REF"]+V3I_YANG] * data_good.shape[0])) # deg
        hdu_ang.header["EXTNAME"] = "DETPA"
        hdul += [hdu_ang]
        temp = np.zeros((2, data_good.shape[0], 1, hdul["KER-MAT"].data.shape[1]))
        temp[0] = np.real(np.array(KPO.CVIS))
        temp[1] = np.imag(np.array(KPO.CVIS))
        hdu_vis = pyfits.ImageHDU(temp)
        hdu_vis.header["EXTNAME"] = "CVIS-DATA"
        hdul += [hdu_vis]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Extract kerphase step done")

        return suffix_out


class empirical_uncertainties:
    """
    Compute empirical uncertainties for the kernel phase.
    """

    def __init__(self):
        """
        Initialize the pipeline step.
        """

        # Initialize the step parameters.
        self.skip = False
        self.plot = True
        self.get_emp_err = True
        self.get_emp_cor = False

    def step(self,
             file,
             suffix,
             output_dir=None,
             show_plots=False,
             good_frames=None,
             ):
        """
        Run the pipeline step.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        suffix: str
            Suffix for the file path to find the product from the previous
            step.
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        print("--> Running empirical uncertainties step...")

        # Open file.
        if suffix == "":
            hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        else:
            hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        kpdat = hdul["KP-DATA"].data
        kpsig = hdul["KP-SIGM"].data
        kpcov = hdul["KP-COV"].data
        if kpdat.ndim != 3 or kpsig.ndim != 3 or kpcov.ndim != 4:
            raise UserWarning("Input file is not a valid KPFITS file")
        if kpdat.shape[0] < 3:
            print("--> Not enough frames to estimate uncertainties empirically, skipping step!")
            return suffix

        # Suffix for the file path from the current step.
        suffix_out = "_emp_kpfits"

        #
        nf = kpdat.shape[0]

        print("--> Estimating empirical uncertainties from %.0f frames" % nf)

        nwave = kpdat.shape[1]
        nkp = kpdat.shape[2]
        wmdat = np.zeros((1, nwave, nkp))
        wmsig = np.zeros((1, nwave, nkp))
        wmcov = np.zeros((1, nwave, nkp, nkp))
        emsig = np.zeros((1, nwave, nkp)) # empirical uncertainties
        emcov = np.zeros((1, nwave, nkp, nkp)) # empirical uncertainties
        emcor = np.zeros((1, nwave, nkp, nkp)) # empirical uncertainties
        emsig_sample = np.zeros((1, nwave, nkp)) # empirical uncertainties
        emcov_sample = np.zeros((1, nwave, nkp, nkp)) # empirical uncertainties
        emcor_sample = np.zeros((1, nwave, nkp, nkp)) # empirical uncertainties
        for i in range(nwave):
            invcov = []
            invcovdat = []
            for j in range(nf):
                invcov += [np.linalg.inv(kpcov[j, i])]
                invcovdat += [invcov[-1].dot(kpdat[j, i])]
            wmcov[0, i] = np.linalg.inv(np.sum(np.array(invcov), axis=0))
            wmsig[0, i] = np.sqrt(np.diag(wmcov[0, i]))
            wmdat[0, i] = wmcov[0, i].dot(np.sum(np.array(invcovdat), axis=0))
            emsig[0, i] = np.std(kpdat[:, i], axis=0)/np.sqrt(nf)
            wmcor = np.true_divide(wmcov[0, i], wmsig[0, i][:, None] * wmsig[0, i][None, :])
            emcov[0, i] = np.multiply(wmcor, emsig[0, i][:, None] * emsig[0, i][None, :])
            emcor[0, i] = wmcor.copy()
            emcov_sample[0, i] = np.cov(kpdat[:, i].T)
            emsig_sample[0, i] = np.sqrt(np.diag(emcov_sample[0, i]))
            emcor_sample[0, i] = np.true_divide(emcov_sample[0, i], emsig_sample[0, i][:, None] * emsig_sample[0, i][None, :])

        # Get output file path.
        path = ut.get_output_base(file, output_dir=output_dir)

        # Plot.
        if self.plot:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            plt.ioff()
            gs = gridspec.GridSpec(2, 2)
            f = plt.figure(figsize=(2.00 * 6.4, 2.00 * 4.8))
            ax = plt.subplot(gs[0, 0])
            p00 = ax.imshow(emcor[0, 0], origin="lower", cmap="RdBu", vmin=-1., vmax=1.)
            c00 = plt.colorbar(p00, ax=ax)
            c00.set_label("Kernel phase correlation", rotation=270, labelpad=20)
            ax.set_title(
                "Theoretical estimate",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ax = plt.subplot(gs[0, 1])
            p01 = ax.imshow(emcor_sample[0, 0], origin="lower", cmap="RdBu", vmin=-1., vmax=1.)
            c01 = plt.colorbar(p01, ax=ax)
            c01.set_label("Kernel phase correlation", rotation=270, labelpad=20)
            ax.set_title(
                "Empirical estimate",
                y=1.,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ax = plt.subplot(gs[1, :])
            ax.fill_between(np.arange(nkp), np.min(kpdat[:, 0], axis=0), np.max(kpdat[:, 0], axis=0), edgecolor="None", facecolor=colors[0], alpha=1. / 3., label="Min/max range")
            if self.get_emp_err:
                ax.errorbar(np.arange(nkp), wmdat[0, 0], yerr=emsig[0, 0], color=colors[0], label="Weighted mean")
            else:
                ax.errorbar(np.arange(nkp), wmdat[0, 0], yerr=wmsig[0, 0], color=colors[0], label="Weighted mean")
            ax.axhline(0., ls="--", color="black")
            ylim = ax.get_ylim()
            ax.set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
            ax.grid(axis="y")
            ax.set_xlabel("Index")
            ax.set_ylabel("Kernel phase [rad]")
            ax.legend(loc="upper right")
            plt.suptitle("Empirical uncertainties step", size=18)
            plt.tight_layout()
            plt.savefig(path + suffix_out + ".pdf")
            if show_plots:
                plt.show()
            plt.close()

        # Save file.
        hdul["KP-DATA"].data = wmdat
        if self.get_emp_err:
            if self.get_emp_cor:
                hdul["KP-SIGM"].data = emsig_sample
                hdul["KP-COV"].data = emcov_sample
            else:
                hdul["KP-SIGM"].data = emsig
                hdul["KP-COV"].data = emcov
        else:
            hdul["KP-SIGM"].data = wmsig
            hdul["KP-COV"].data = wmcov
        hdul["DETPA"].data = np.array([np.mean(hdul["DETPA"].data)])
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Empirical uncertainties step done")

        return suffix_out
