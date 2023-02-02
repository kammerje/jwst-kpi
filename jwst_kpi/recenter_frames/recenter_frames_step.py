import os

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astroquery.svo_fps import SvoFps
from xara import core, kpo

from . import pupil_data
from . import utils as ut

PUPIL_DIR = pupil_data.__path__[0]

# Detector pixel scales.
# TODO: assumes that NIRISS pixels are square but they are slightly
#       rectangular.
# TODO: Move to access from many files
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview
pscale = {
    "NIRCAM_SHORT": 31.0,  # mas
    "NIRCAM_LONG": 63.0,  # mas
    "NIRISS": 65.55,  # mas
    "MIRI": 110.0,  # mas
}

# TODO: Encapsulate this somewhere else and call when needed?
# TODO: Move to access from many files
# Load the NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile
# Service.
# http://svo2.cab.inta-csic.es/theory/fps/
wave_nircam = {}
weff_nircam = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRCAM")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_nircam[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_nircam[name] = filter_list["WidthEff"][i] / 1e4  # micron
wave_niriss = {}
weff_niriss = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRISS")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_niriss[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_niriss[name] = filter_list["WidthEff"][i] / 1e4  # micron
wave_miri = {}
weff_miri = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="MIRI")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_miri[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_miri[name] = filter_list["WidthEff"][i] / 1e4  # micron
del filter_list


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
        self.bmax = 6.0  # m
        self.pupil_path = None
        self.verbose = False

    def step(
        self,
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
                raise UserWarning(
                    "List of good frames needs to contain at least one element"
                )
            elif not all(isinstance(item, int) for item in good_frames):
                raise UserWarning(
                    "List of good frames may only contain integer elements"
                )
            elif np.max(good_frames) >= nf or np.min(good_frames) < nf * (-1):
                raise UserWarning(
                    "Some of the provided good frames are outside the data range"
                )
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
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = (
            -hdul["SCI"].header["V3I_YANG"] * hdul["SCI"].header["VPARITY"]
        )  # deg, counter-clockwise

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
                wave = wave_nircam[FILTER] * 1e-6  # m
                weff = weff_nircam[FILTER] * 1e-6  # m
            elif INSTRUME == "NIRISS":
                if self.pupil_path is None:
                    if hdul[0].header["PUPIL"] == "NRM":
                        default_pupil_model = "niriss_nrm_pupil.fits"
                    else:
                        default_pupil_model = "niriss_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_niriss[FILTER] * 1e-6  # m
                weff = weff_niriss[FILTER] * 1e-6  # m
            elif INSTRUME == "MIRI":
                if self.pupil_path is None:
                    default_pupil_model = "miri_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_miri[FILTER] * 1e-6  # m
                weff = weff_miri[FILTER] * 1e-6  # m

            # print("Rotating pupil model by %.2f deg (counter-clockwise)" % V3I_YANG)

            # # Rotate pupil model.
            # theta = np.deg2rad(V3I_YANG) # rad
            # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # hdul_pup = fits.open(self.pupil_path)
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
            dx = []  # pix
            dy = []  # pix
            for i in range(nf):
                if (
                    good_frames is None
                    or i in good_frames
                    or (nf - i) * (-1) in good_frames
                ):
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
                    dx += [0.0]
                    dy += [0.0]
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
                    "center = %.2f, %.2f"
                    % (sx // 2 + dx[good_frames[0]], sy // 2 + dy[good_frames[0]]),
                    color="white",
                    ha="left",
                    va="bottom",
                    transform=ax[0].transAxes,
                    size=12,
                )
            t0.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="black")]
            )
            ax[0].set_title(
                "Frame",
                y=1.0,
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
            t1.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="black")]
            )
            ax[1].set_title(
                "Recentered frame",
                y=1.0,
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
        hdu_sci_org = fits.ImageHDU(data)
        hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
        hdu_err_org = fits.ImageHDU(erro)
        hdu_err_org.header["EXTNAME"] = "ERR-ORG"
        hdul += [hdu_sci_org, hdu_err_org]
        hdul["SCI"].data = data_recentered
        hdul["ERR"].data = erro_recentered
        if is2d:
            xsh = fits.Column(name="XSHIFT", format="D", array=np.array([dx]))  # pix
            ysh = fits.Column(name="YSHIFT", format="D", array=np.array([dy]))  # pix
        else:
            xsh = fits.Column(name="XSHIFT", format="D", array=np.array(dx))  # pix
            ysh = fits.Column(name="YSHIFT", format="D", array=np.array(dy))  # pix
        hdu_ims = fits.BinTableHDU.from_columns([xsh, ysh])
        hdu_ims.header["EXTNAME"] = "IMSHIFT"
        hdul += [hdu_ims]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Recenter frames step done")

        return suffix_out
