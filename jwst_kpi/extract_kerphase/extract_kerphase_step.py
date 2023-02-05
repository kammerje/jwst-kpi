import os

import matplotlib.pyplot as plt
import numpy as np
# from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step
from xara import core, kpo

from ..constants import (gain, pscale, wave_miri, wave_nircam, wave_niriss,
                         weff_miri, weff_nircam, weff_niriss)
from ..datamodels import KPFitsModel
from .. import pupil_data
from .extract_kerphase_plots import plot_kerphase
from .. import utils as ut

PUPIL_DIR = pupil_data.__path__[0]


class ExtractKerphaseStep(Step):
    """
    Extract the kernel phase while re-centering in complex visibility space.

    The KPFITS file structure has been agreed upon by the participants of
    Steph Sallum's masking & kernel phase hackathon in 2021 and is defined
    here:
        https://ui.adsabs.harvard.edu/abs/2022arXiv221017528K/abstract
    """

    class_alias = "extract_kerphase"

    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        instrume_allowed = string_list(default=list('NIRCAM', 'NIRISS', 'MIRI'))
        bmax = float(default=None)
        pupil_path = string(default=None)
        verbose = boolean(default=False)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(
        self,
        input_data,
        # recenter_frames_obj,
        # window_frames_obj,
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

        self.log.info("--> Running extract kerphase step...")

        good_frames = self.good_frames

        # Open file.
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        data = input_models.data
        erro = input_models.err
        # TODO: Fix pxdq
        # pxdq = input_models.dq
        pxdq = np.zeros_like(erro)
        # TODO: Handle ORG/MOD
        # try:
        #     data_org = hdul["SCI-ORG"].data
        #     erro_org = hdul["ERR-ORG"].data
        # except Exception:
        #     pass
        # try:
        #     pxdq_mod = hdul["DQ-MOD"].data
        # except Exception:
        #     pass
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
            # try:
            #     data_org = data_org[np.newaxis]
            #     erro_org = erro_org[np.newaxis]
            # except Exception:
            #     pass
            # try:
            #     pxdq_mod = pxdq_mod[np.newaxis]
            # except Exception:
            #     pass
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

        # Get detector pixel scale and position angle.
        # TODO: Dup code with recentering step
        if INSTRUME == "NIRCAM":
            CHANNEL = input_models.meta.instrument.channel
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = (
            -input_models.meta.wcsinfo.v3yangle * input_models.meta.wcsinfo.vparity
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

        # TODO: Up code lasts until this block
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
        # TODO: Clean up or set-up with T/F switch
        # KPO.kpi.plot_pupil_and_uv()
        # plt.show()

        # Recenter frames, window frames, and extract kernel phase.
        # TODO: Does this mean recentering is always done twice?
        # TODO: Have single loop, do setup before with if statements of following:
        # - data vs data_org
        # - wrad from window_frames or None
        # - centering algo and bmax values
        # - recenter=True or False
        # TODO: this is only for code to work. Probably yields wrong results when recenter is True
        data_org = data.copy()
        # if recenter_frames_obj.skip and window_frames_obj.skip:
        if True:
            # TODO: Update
            # if "SCI-ORG" in hdul:
            #     raise UserWarning(
            #         "There is a SCI-ORG FITS file extension although both the recentering and the windowing steps were skipped"
            #     )
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
        # elif recenter_frames_obj.skip and not window_frames_obj.skip:
        #     for i in range(nf):
        #         if (
        #             good_frames is None
        #             or i in good_frames
        #             or (nf - i) * (-1) in good_frames
        #         ):
        #             KPO.extract_KPD_single_frame(
        #                 data_org[i],
        #                 PSCALE,
        #                 wave,
        #                 target=None,
        #                 recenter=False,
        #                 wrad=window_frames_obj.wrad,
        #                 method="LDFT1",
        #             )
        # elif not recenter_frames_obj.skip and window_frames_obj.skip:
        #     dx = []  # pix
        #     dy = []  # pix
        #     for i in range(nf):
        #         if (
        #             good_frames is None
        #             or i in good_frames
        #             or (nf - i) * (-1) in good_frames
        #         ):
        #             temp = KPO.extract_KPD_single_frame(
        #                 data_org[i],
        #                 PSCALE,
        #                 wave,
        #                 target=None,
        #                 recenter=True,
        #                 wrad=None,
        #                 method="LDFT1",
        #                 algo_cent=recenter_frames_obj.method,
        #                 bmax_cent=recenter_frames_obj.bmax,
        #             )
        #             dx += [temp[0]]
        #             dy += [temp[1]]
        # else:
        #     dx = []  # pix
        #     dy = []  # pix
        #     for i in range(nf):
        #         if (
        #             good_frames is None
        #             or i in good_frames
        #             or (nf - i) * (-1) in good_frames
        #         ):
        #             temp = KPO.extract_KPD_single_frame(
        #                 data_org[i],
        #                 PSCALE,
        #                 wave,
        #                 target=None,
        #                 recenter=True,
        #                 wrad=window_frames_obj.wrad,
        #                 method="LDFT1",
        #                 algo_cent=recenter_frames_obj.method,
        #                 bmax_cent=recenter_frames_obj.bmax,
        #             )
        #             dx += [temp[0]]
        #             dy += [temp[1]]

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

        # TODO: Test and cleanup
        # Get output file path.
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_kerphase(data, KPO, m2pix, kpcor, kpsig, good_frames=good_frames)
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        # TODO: Move creation of KPFITS to separate function
        # TODO: JWST data model?
        data_good = []
        erro_good = []
        pxdq_good = []
        # data_org_good = []
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
                # TODO: Update
                # try:
                #     data_org_good += [data_org[i]]
                #     erro_org_good += [erro_org[i]]
                # except Exception:
                #     pass
                # try:
                #     pxdq_mod_good += [pxdq_mod[i]]
                # except Exception:
                #     pass
        data_good = np.array(data_good)
        erro_good = np.array(erro_good)
        pxdq_good = np.array(pxdq_good)
        # data_org_good = np.array(data_org_good)
        erro_org_good = np.array(erro_org_good)
        pxdq_mod_good = np.array(pxdq_mod_good)
        # TODO: Kpfits model for output and udpate with input
        output_models = KPFitsModel()
        output_models.update(input_models, extra_fits=True)
        output_models.data = data_good[:, np.newaxis, :, :]
        output_models.err = erro_good[:, np.newaxis, :, :]
        output_models.dq = pxdq_good[:, np.newaxis, :, :]
        # TODO: Update ORG and MOD
        # TODO: Simpler handling of repeated try/except's?
        # try:
        #     output_models.data_org = data_org_good[:, np.newaxis, :, :]
        #     output_models.err_org = erro_org_good[:, np.newaxis, :, :]
        # except AttributeError:
        #     pass
        try:
            output_models.dq_mod = pxdq_mod_good[:, np.newaxis, :, :]
        except Exception:
            pass
        output_models.meta.pscale = PSCALE
        # TODO: Handle gain_kwd earlier and then use this to get without if/else
        if INSTRUME == "NIRCAM":
            output_models.meta.gain = gain[INSTRUME + "_" + CHANNEL]  # e-/ADU
        else:
            output_models.meta.gain = gain[INSTRUME]  # e-/ADU
        # TODO: Store in constants instead of hardcoding
        output_models.meta.diam = 6.559348  # m (flat-to-flat)
        output_models.meta.exptime = input_models.meta.exposure.integration_time
        output_models.meta.dateobs = (
            input_models.meta.observation.date
            + "T"
            + input_models.meta.observation.time
        )  # YYYY-MM-DDTHH:MM:SS.MMM
        output_models.meta.procsoft = "CALWEBB_KPI3"
        try:
            output_models.meta.wrad = input_models.meta.wrad  # pix
        except AttributeError:
            try:
                output_models.extra_fits.PRIMARY = input_models
            except AttributeError:
                # TODO: type check will prob raise error
                output_models.meta.wrad = "NONE"
        output_models.meta.calflag = False
        output_models.meta.content = "KPFITS1"
        # Aperture coordinates
        # TODO: Can transfer ttypes with datamodels/schemas?
        # TODO: Check this shape
        output_models.aperture = np.recarray(KPO.kpi.VAC.shape[0], output_models.aperture.dtype)
        output_models.aperture["XXC"] = KPO.kpi.VAC[:, 0]  # m
        output_models.aperture["YYC"] = KPO.kpi.VAC[:, 1]  # m
        output_models.aperture["TRM"] = KPO.kpi.TRM  # (0 <= 1 <= 1)
        # hdu_ape.header["TTYPE1"] = ("XXC", "Virtual aperture x-coord (meter)")
        # hdu_ape.header["TTYPE2"] = ("YYC", "Virtual aperture y-coord (meter)")
        # hdu_ape.header["TTYPE3"] = ("TRM", "Virtual aperture transmission (0 < t <= 1)")
        # UV plane
        output_models.uv_plane = np.recarray(KPO.kpi.UVC.shape[0], output_models.uv_plane.dtype)
        output_models.uv_plane["UUC"] = KPO.kpi.UVC[:, 0]  # m
        output_models.uv_plane["VVC"] = KPO.kpi.UVC[:, 1]  # m
        output_models.uv_plane["RED"] = KPO.kpi.RED  # int
        # hdu_uvp.header["TTYPE1"] = ("UUC", "Baseline u-coord (meter)")
        # hdu_uvp.header["TTYPE2"] = ("VVC", "Baseline v-coord (meter)")
        # hdu_uvp.header["TTYPE3"] = ("RED", "Baseline redundancy (int)")
        output_models.ker_mat = KPO.kpi.KPM
        output_models.blm_mat = np.diag(KPO.kpi.RED).dot(KPO.kpi.TFM)
        output_models.kp_data = np.array(KPO.KPDT)  # rad
        output_models.kp_sigm = kpsig[:, np.newaxis, :]  # rad
        output_models.kp_cov = kpcov[:, np.newaxis, :]  # rad^2
        wave_arr = np.array([wave])
        output_models.cwavel = np.recarray(wave_arr.shape, output_models.cwavel.dtype)
        output_models.cwavel["CWAVEL"] = wave_arr  # m
        output_models.cwavel["BWIDTH"] = np.array([weff])  # m
        # TODO: 1 or 2 dimensions?
        output_models.detpa = np.array(
            [output_models.meta.wcsinfo.roll_ref + V3I_YANG] * data_good.shape[0]
        )  # deg
        temp = np.zeros((2, data_good.shape[0], 1, output_models.ker_mat.shape[1]))
        temp[0] = np.real(np.array(KPO.CVIS))
        temp[1] = np.imag(np.array(KPO.CVIS))
        output_models.cvis_data = temp

        self.log.info("--> Extract kerphase step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(ExtractKerphaseStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
