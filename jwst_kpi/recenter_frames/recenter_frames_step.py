import os

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
# from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step
from xara import core, kpo

from . import pupil_data
from ..constants import pscale, wave_nircam, wave_niriss, wave_miri, weff_nircam, weff_niriss, weff_miri
# from . import utils as ut

PUPIL_DIR = pupil_data.__path__[0]


class RecenterFramesStep(Step):
    """
    Recenter frames.
    """

    class_alias = "recenter_frames"

    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        method = string(default='FPNM')
        method_allowed = string_list(default=['BCEN', 'COGI', 'FPNM'])
        instrume_allowed = string_list(default=['NIRCAM', 'NIRISS', 'MIRI'])
        bmax = float(default=6.0)
        pupil_path = string(default=None)
        verbose = boolean(default=False)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(
        self,
        input_data,
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

        self.log.info("--> Running recenter frames step...")

        good_frames = self.good_frames

        # Open file. Use default pipeline way unless "previous suffix is used"
        # TODO: Mention this in PR
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        # if suffix == "":
        #     hdul = ut.open_fits(file, suffix=suffix, file_dir=None)
        # else:
        #     hdul = ut.open_fits(file, suffix=suffix, file_dir=output_dir)
        data = input_models.data
        erro = input_models.err
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
        INSTRUME = input_models.meta.instrument.name
        if INSTRUME not in self.instrume_allowed:
            raise UserWarning("Unsupported instrument")
        FILTER = input_models.meta.instrument.filter

        # Suffix for the file path from the current step.
        suffix_out = f"_{self.suffix or self.default_suffix()}"

        # Simple background subtraction to avoid discontinuity when zero-padding the data in XARA.
        data -= np.median(data, axis=(1, 2), keepdims=True)

        # Get detector pixel scale and position angle.
        if INSTRUME == "NIRCAM":
            CHANNEL = input_models.meta.instrument.channel
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = (
            - input_models.meta.wcsinfo.v3yangle * data.meta.wcsinfo.vparity
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
            pupil_name = input_models.meta.instrument.pupil
            if INSTRUME == "NIRCAM":
                if self.pupil_path is None:
                    if pupil_name == "MASKRND":
                        default_pupil_model = "nircam_rnd_pupil.fits"
                    elif pupil_name == "MASKBAR":
                        default_pupil_model = "nircam_bar_pupil.fits"
                    else:
                        default_pupil_model = "nircam_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
                wave = wave_nircam[FILTER] * 1e-6  # m
                weff = weff_nircam[FILTER] * 1e-6  # m
            elif INSTRUME == "NIRISS":
                if self.pupil_path is None:
                    if pupil_name == "NRM":
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
                    self.log.info("Image shift = (%.2f, %.2f)" % (dx[-1], dy[-1]))
            data_recentered = np.array(data_recentered)
            erro_recentered = np.array(erro_recentered)

        # TODO: Test and cleanup
        # Get output file path.
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        path = os.path.splitext(mk_path)[0]

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
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        # TODO: Mark step completed
        # TODO: How add keywords in pipeline?
        # TODO: Might want to just save this direclty here instead of using default saving mech
        if is2d:
            data = data[0]
            erro = erro[0]
            data_recentered = data_recentered[0]
            erro_recentered = erro_recentered[0]
        output_models = input_models.copy()
        output_models.data = data_recentered
        output_models.err = erro_recentered
        # hdu_sci_org = fits.ImageHDU(data)
        # hdu_sci_org.header["EXTNAME"] = "SCI-ORG"
        # hdu_err_org = fits.ImageHDU(erro)
        # hdu_err_org.header["EXTNAME"] = "ERR-ORG"
        # hdul += [hdu_sci_org, hdu_err_org]
        # hdul["SCI"].data = data_recentered
        # hdul["ERR"].data = erro_recentered
        # if is2d:
        #     xsh = fits.Column(name="XSHIFT", format="D", array=np.array([dx]))  # pix
        #     ysh = fits.Column(name="YSHIFT", format="D", array=np.array([dy]))  # pix
        # else:
        #     xsh = fits.Column(name="XSHIFT", format="D", array=np.array(dx))  # pix
        #     ysh = fits.Column(name="YSHIFT", format="D", array=np.array(dy))  # pix
        # hdu_ims = fits.BinTableHDU.from_columns([xsh, ysh])
        # hdu_ims.header["EXTNAME"] = "IMSHIFT"
        # hdul += [hdu_ims]
        # hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        # hdul.close()

        self.log.info("--> Recenter frames step done")

        return output_models