import os

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
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview
pscale = {
    "NIRCAM_SHORT": 31.0,  # mas
    "NIRCAM_LONG": 63.0,  # mas
    "NIRISS": 65.55,  # mas
    "MIRI": 110.0,  # mas
}
# TODO: Move to more generic file
# Detector gains.
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview/niriss-detector-performance
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-performance
gain = {
    "NIRCAM_SHORT": 2.05,  # e-/ADU
    "NIRCAM_LONG": 1.82,  # e-/ADU
    "NIRISS": 1.61,  # e-/ADU
    "MIRI": 4.0,  # e-/ADU
}

# TODO: Encapsulate this somewhere else and call when needed?
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
        self.bmax = None  # m
        self.pupil_path = None
        self.verbose = False

    def step(
        self,
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
        except Exception:
            pass
        try:
            pxdq_mod = hdul["DQ-MOD"].data
        except Exception:
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
            except Exception:
                pass
            try:
                pxdq_mod = pxdq_mod[np.newaxis]
            except Exception:
                pass
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
        suffix_out = "_kpfits"

        # Get detector pixel scale and position angle.
        if INSTRUME == "NIRCAM":
            CHANNEL = hdul[0].header["CHANNEL"]
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = (
            -hdul["SCI"].header["V3I_YANG"] * hdul["SCI"].header["VPARITY"]
        )  # deg, counter-clockwise

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

        # Recenter frames, window frames, and extract kernel phase.
        if recenter_frames_obj.skip and window_frames_obj.skip:
            if "SCI-ORG" in hdul:
                raise UserWarning(
                    "There is a SCI-ORG FITS file extension although both the recentering and the windowing steps were skipped"
                )
            for i in range(nf):
                if (
                    good_frames is None
                    or i in good_frames
                    or (nf - i) * (-1) in good_frames
                ):
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
                if (
                    good_frames is None
                    or i in good_frames
                    or (nf - i) * (-1) in good_frames
                ):
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
            dx = []  # pix
            dy = []  # pix
            for i in range(nf):
                if (
                    good_frames is None
                    or i in good_frames
                    or (nf - i) * (-1) in good_frames
                ):
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
            dx = []  # pix
            dy = []  # pix
            for i in range(nf):
                if (
                    good_frames is None
                    or i in good_frames
                    or (nf - i) * (-1) in good_frames
                ):
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
            if (
                good_frames is None
                or i in good_frames
                or (nf - i) * (-1) in good_frames
            ):
                frame = data[i].copy()
                varframe = erro[i].copy() ** 2
                B = KPO.kpi.KPM.dot(
                    np.divide(KPO.FF.imag.T, np.abs(KPO.FF.dot(frame.flatten()))).T
                )
                kpcov += [np.multiply(B, varframe.flatten()).dot(B.T)]
                kpsig += [np.sqrt(np.diag(kpcov[-1]))]
                kpcor += [
                    np.true_divide(kpcov[-1], kpsig[-1][:, None] * kpsig[-1][None, :])
                ]
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
                d00 = np.fft.fftshift(
                    np.fft.fft2(np.fft.fftshift(data[good_frames[0]]))
                )
            p00 = ax[0, 0].imshow(
                np.angle(d00), origin="lower", vmin=-np.pi, vmax=np.pi
            )
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
                y=1.0,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            p01 = ax[0, 1].imshow(
                kpcor[0], origin="lower", cmap="RdBu", vmin=-1.0, vmax=1.0
            )
            c01 = plt.colorbar(p01, ax=ax[0, 1])
            c01.set_label("Correlation", rotation=270, labelpad=20)
            ax[0, 1].set_title(
                "Kernel phase correlation",
                y=1.0,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ww = np.argsort(KPO.kpi.BLEN)
            ax[1, 0].plot(np.angle(KPO.CVIS[0][0, ww]))
            ax[1, 0].axhline(0.0, ls="--", color="black")
            ax[1, 0].set_ylim([-np.pi, np.pi])
            ax[1, 0].grid(axis="y")
            ax[1, 0].set_xlabel("Index sorted by baseline length")
            ax[1, 0].set_ylabel("Fourier phase [rad]")
            ax[1, 0].set_title(
                "Fourier phase",
                y=1.0,
                pad=-20,
                bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round"),
            )
            ax[1, 1].errorbar(
                np.arange(KPO.KPDT[0][0, :].shape[0]),
                KPO.KPDT[0][0, :],
                yerr=kpsig[0],
                color=colors[0],
                alpha=1.0 / 3.0,
            )
            ax[1, 1].plot(
                np.arange(KPO.KPDT[0][0, :].shape[0]),
                KPO.KPDT[0][0, :],
                color=colors[0],
            )
            ax[1, 1].axhline(0.0, ls="--", color="black")
            ylim = ax[1, 1].get_ylim()
            ax[1, 1].set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
            ax[1, 1].grid(axis="y")
            ax[1, 1].set_xlabel("Index")
            ax[1, 1].set_ylabel("Kernel phase [rad]")
            ax[1, 1].set_title(
                "Kernel phase",
                y=1.0,
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
            if (
                good_frames is None
                or i in good_frames
                or (nf - i) * (-1) in good_frames
            ):
                data_good += [data[i]]
                erro_good += [erro[i]]
                pxdq_good += [pxdq[i]]
                try:
                    data_org_good += [data_org[i]]
                    erro_org_good += [erro_org[i]]
                except Exception:
                    pass
                try:
                    pxdq_mod_good += [pxdq_mod[i]]
                except Exception:
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
        except Exception:
            pass
        try:
            hdul["DQ-MOD"].data = pxdq_mod_good[:, np.newaxis, :, :]
        except Exception:
            pass
        hdul[0].header["PSCALE"] = PSCALE  # mas
        if INSTRUME == "NIRCAM":
            hdul[0].header["GAIN"] = gain[INSTRUME + "_" + CHANNEL]  # e-/ADU
        else:
            hdul[0].header["GAIN"] = gain[INSTRUME]  # e-/ADU
        hdul[0].header["DIAM"] = 6.559348  # m (flat-to-flat)
        hdul[0].header["EXPTIME"] = hdul[0].header["EFFINTTM"]  # s
        hdul[0].header["DATEOBS"] = (
            hdul[0].header["DATE-OBS"] + "T" + hdul[0].header["TIME-OBS"]
        )  # YYYY-MM-DDTHH:MM:SS.MMM
        hdul[0].header["PROCSOFT"] = "CALWEBB_KPI3"
        try:
            hdul[0].header["WRAD"] = hdul["WINMASK"].header["WRAD"]  # pix
        except Exception:
            hdul[0].header["WRAD"] = "NONE"
        hdul[0].header["CALFLAG"] = "False"
        hdul[0].header["CONTENT"] = "KPFITS1"
        xy1 = fits.Column(name="XXC", format="D", array=KPO.kpi.VAC[:, 0])  # m
        xy2 = fits.Column(name="YYC", format="D", array=KPO.kpi.VAC[:, 1])  # m
        trm = fits.Column(name="TRM", format="D", array=KPO.kpi.TRM)
        hdu_ape = fits.BinTableHDU.from_columns([xy1, xy2, trm])
        hdu_ape.header["EXTNAME"] = "APERTURE"
        hdu_ape.header["TTYPE1"] = ("XXC", "Virtual aperture x-coord (meter)")
        hdu_ape.header["TTYPE2"] = ("YYC", "Virtual aperture y-coord (meter)")
        hdu_ape.header["TTYPE3"] = ("TRM", "Virtual aperture transmission (0 < t <= 1)")
        hdul += [hdu_ape]
        uv1 = fits.Column(name="UUC", format="D", array=KPO.kpi.UVC[:, 0])  # m
        uv2 = fits.Column(name="VVC", format="D", array=KPO.kpi.UVC[:, 1])  # m
        red = fits.Column(name="RED", format="I", array=KPO.kpi.RED)
        hdu_uvp = fits.BinTableHDU.from_columns([uv1, uv2, red])
        hdu_uvp.header["EXTNAME"] = "UV-PLANE"
        hdu_uvp.header["TTYPE1"] = ("UUC", "Baseline u-coord (meter)")
        hdu_uvp.header["TTYPE2"] = ("VVC", "Baseline v-coord (meter)")
        hdu_uvp.header["TTYPE3"] = ("RED", "Baseline redundancy (int)")
        hdul += [hdu_uvp]
        hdu_kpm = fits.ImageHDU(KPO.kpi.KPM)
        hdu_kpm.header["EXTNAME"] = "KER-MAT"
        hdul += [hdu_kpm]
        hdu_blm = fits.ImageHDU(np.diag(KPO.kpi.RED).dot(KPO.kpi.TFM))
        hdu_blm.header["EXTNAME"] = "BLM-MAT"
        hdul += [hdu_blm]
        hdu_kpd = fits.ImageHDU(np.array(KPO.KPDT))  # rad
        hdu_kpd.header["EXTNAME"] = "KP-DATA"
        hdul += [hdu_kpd]
        hdu_kpe = fits.ImageHDU(kpsig[:, np.newaxis, :])  # rad
        hdu_kpe.header["EXTNAME"] = "KP-SIGM"
        hdul += [hdu_kpe]
        hdu_kpc = fits.ImageHDU(kpcov[:, np.newaxis, :])  # rad^2
        hdu_kpc.header["EXTNAME"] = "KP-COV"
        hdul += [hdu_kpc]
        cwavel = fits.Column(name="CWAVEL", format="D", array=np.array([wave]))  # m
        bwidth = fits.Column(name="BWIDTH", format="D", array=np.array([weff]))  # m
        hdu_lam = fits.BinTableHDU.from_columns([cwavel, bwidth])
        hdu_lam.header["EXTNAME"] = "CWAVEL"
        hdul += [hdu_lam]
        hdu_ang = fits.ImageHDU(
            np.array([hdul["SCI"].header["ROLL_REF"] + V3I_YANG] * data_good.shape[0])
        )  # deg
        hdu_ang.header["EXTNAME"] = "DETPA"
        hdul += [hdu_ang]
        temp = np.zeros((2, data_good.shape[0], 1, hdul["KER-MAT"].data.shape[1]))
        temp[0] = np.real(np.array(KPO.CVIS))
        temp[1] = np.imag(np.array(KPO.CVIS))
        hdu_vis = fits.ImageHDU(temp)
        hdu_vis.header["EXTNAME"] = "CVIS-DATA"
        hdul += [hdu_vis]
        hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        hdul.close()

        print("--> Extract kerphase step done")

        return suffix_out
