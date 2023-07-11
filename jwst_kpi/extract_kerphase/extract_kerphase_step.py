import os

import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
from xara import core, kpo

from .. import pupil_data
from .. import utils as ut
from ..constants import (gain, pscale, wave_miri, wave_nircam, wave_niriss,
                         weff_miri, weff_nircam, weff_niriss, DIAM)
from ..datamodels import KPFitsModel
from .extract_kerphase_plots import plot_kerphase

PUPIL_DIR = pupil_data.__path__[0]


class ExtractKerphaseStep(Step):
    """
    Extract the kernel phase while re-centering in complex visibility space.

    The KPFITS file structure has been agreed upon by the participants of
    Steph Sallum's masking & kernel phase hackathon in 2021 and is defined
    here:
        https://ui.adsabs.harvard.edu/abs/2022arXiv221017528K/abstract

    Parameters
    -----------
    input_data :  ~jwst_kpi.datamodels.KPFitsModel
        Single filename for extracted kernel phase data
    plot : bool
        Generate plots
    show_plots : bool
        Show plots
    previous_suffix : Optional[str]
        Suffix of previous file. DEPRECATED: use ~input_data directly instead
    good_frames : List[int]
        List of good frames, bad frames will be skipped.
    instrume_allowed : List[str]
        Instruments allowed by the pipeline
        (default ['NIRCAM', 'NIRISS', 'MIRI'] should not be changed by users)
    bmax : Optional[float]
        Maximum baseline to consider in extraction (defaults to None)
    recenter_method : Optional[str]
        Recentering method to be used (defaults to None, for no recentering)
    recenter_bmax : float
        Maximum baseline to consider when recentering (in m, defaults to 6.0)
    recenter_method_allowed : Optional[List[str]]
        Recentering methods allowed
        (defaults to None, should be inherited from recenter step)
    wrad : Optional[int]
        Windowing radius to use for super-gaussian
        (defaults to None, should be inherited from windowing step)
    pupil_path : Optional[str]
        Optional path to custom pupil model
    verbose : bool
        Enable verbose mode
    """

    class_alias = "extract_kerphase"

    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        instrume_allowed = string_list(default=list('NIRCAM', 'NIRISS', 'MIRI'))
        bmax = float(default=None)
        recenter_method = string(default=None)
        recenter_bmax = float(default=6.0)
        recenter_method_allowed = string_list(default=None)
        wrad = integer(default=None)
        pupil_path = string(default=None)
        verbose = boolean(default=False)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(self, input_data):

        self.log.info("--> Running extract kerphase step...")

        good_frames = self.good_frames

        # Open file.
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        data = input_models.data
        erro = input_models.err
        pxdq = input_models.dq
        try:
            data_org = input_models.data_org
            erro_org = input_models.err_org
            has_org = True
        except Exception:
            has_org = False
        try:
            pxdq_mod = input_models.dq_mod
            has_mod = True
        except Exception:
            has_mod = False
        if data.ndim not in [2, 3]:
            raise UserWarning("Only implemented for 2D image/3D data cube")
        if data.ndim == 2:
            is2d = True
            data = data[np.newaxis]
            erro = erro[np.newaxis]
            pxdq = pxdq[np.newaxis]
            if has_org:
                data_org = data_org[np.newaxis]
                erro_org = erro_org[np.newaxis]
            if has_mod:
                pxdq_mod = pxdq_mod[np.newaxis]
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
        pupil_name = input_models.meta.instrument.pupil

        # Get detector pixel scale and position angle.
        # TODO: Dup code with recentering and badpix step
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
        if pupil_name in filter_allowed:
            FILTER = pupil_name

        # TODO: Dup code lasts until this next block (inclusively)
        # Get pupil model path and filter properties.
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
            hexa=True,
            ID="",
        )
        m2pix = core.mas2rad(PSCALE) * sx / wave

        do_recenter = self.recenter_method is not None
        do_window = self.wrad is not None
        if do_recenter and self.recenter_method not in self.recenter_method_allowed:
            raise ValueError("Unknown recentering method")

        # Recenter frames, window frames, and extract kernel phase.
        if do_recenter or do_window:
            data_extract = data_org
        else:
            data_extract = data
        if do_recenter:
            dx = []  # pix
            dy = []  # pix
        for i in range(nf):
            if (
                good_frames is None
                or i in good_frames
                or (nf - i) * (-1) in good_frames
            ):
                temp = KPO.extract_KPD_single_frame(
                    data_extract[i],
                    PSCALE,
                    wave,
                    target=None,
                    recenter=do_recenter,
                    wrad=self.wrad,
                    method="LDFT1",
                    algo_cent=self.recenter_method,
                    bmax_cent=self.recenter_bmax,
                )
                if do_recenter:
                    dx += [temp[0]]
                    dy += [temp[1]]

        # Extract kernel phase covariance. See:
        # https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
        kpcov = []
        kpsig = []
        kpcor = []
        # TODO: Should this use "erro_org" for consistency re. recenter and window?
        # Or will baseline definition be "immune" to recenter?
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
                if has_org:
                    data_org_good += [data_org[i]]
                    erro_org_good += [erro_org[i]]
                if has_mod:
                    pxdq_mod_good += [pxdq_mod[i]]
        data_good = np.array(data_good)
        erro_good = np.array(erro_good)
        pxdq_good = np.array(pxdq_good)
        if has_org:
            data_org_good = np.array(data_org_good)
            erro_org_good = np.array(erro_org_good)
        if has_mod:
            pxdq_mod_good = np.array(pxdq_mod_good)
        output_models = KPFitsModel()
        output_models.update(input_models, extra_fits=True)
        output_models.data = data_good[:, np.newaxis, :, :]
        output_models.err = erro_good[:, np.newaxis, :, :]
        output_models.dq = pxdq_good[:, np.newaxis, :, :]
        if has_org:
            output_models.data_org = data_org_good[:, np.newaxis, :, :]
            output_models.err_org = erro_org_good[:, np.newaxis, :, :]
        if has_mod:
            output_models.dq_mod = pxdq_mod_good[:, np.newaxis, :, :]
        output_models.meta.kpi_extract.pscale = PSCALE
        # TODO: Handle gain_kwd earlier and then use this to get without if/else
        if INSTRUME == "NIRCAM":
            output_models.meta.kpi_extract.gain = gain[
                INSTRUME + "_" + CHANNEL
            ]  # e-/ADU
        else:
            output_models.meta.kpi_extract.gain = gain[INSTRUME]  # e-/ADU
        output_models.meta.kpi_extract.diam = DIAM  # m (flat-to-flat)
        output_models.meta.kpi_extract.exptime = (
            input_models.meta.exposure.integration_time
        )
        output_models.meta.kpi_extract.dateobs = (
            input_models.meta.observation.date
            + "T"
            + input_models.meta.observation.time
        )  # YYYY-MM-DDTHH:MM:SS.MMM
        output_models.meta.kpi_extract.procsoft = "CALWEBB_KPI3"
        if do_window:
            try:
                output_models.meta.kpi_preprocess.wrad = (
                    input_models.meta.kpi_preprocess.wrad
                )  # pix
            except AttributeError:
                try:
                    output_models.meta.kpi_preprocess.wrad = self.wrad
                except AttributeError:
                    pass
        output_models.meta.kpi_extract.calflag = False
        output_models.meta.kpi_extract.content = "KPFITS1"
        # Aperture coordinates
        output_models.aperture = np.recarray(
            KPO.kpi.VAC.shape[0], output_models.aperture.dtype
        )
        output_models.aperture["XXC"] = KPO.kpi.VAC[:, 0]  # m
        output_models.aperture["YYC"] = KPO.kpi.VAC[:, 1]  # m
        output_models.aperture["TRM"] = KPO.kpi.TRM  # (0 <= 1 <= 1)
        # hdu_ape.header["TTYPE1"] = ("XXC", "Virtual aperture x-coord (meter)")
        # hdu_ape.header["TTYPE2"] = ("YYC", "Virtual aperture y-coord (meter)")
        # hdu_ape.header["TTYPE3"] = ("TRM", "Virtual aperture transmission (0 < t <= 1)")
        # UV plane
        output_models.uv_plane = np.recarray(
            KPO.kpi.UVC.shape[0], output_models.uv_plane.dtype
        )
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
        output_models.detpa = np.array(
            [output_models.meta.wcsinfo.roll_ref + V3I_YANG] * data_good.shape[0]
        )  # deg
        temp = np.zeros((2, data_good.shape[0], 1, output_models.ker_mat.shape[1]))
        temp[0] = np.real(np.array(KPO.CVIS))
        temp[1] = np.imag(np.array(KPO.CVIS))
        output_models.cvis_data = temp
        output_models.meta.cal_step_kpi.extract = "COMPLETE"
        if do_recenter:
            output_models.meta.cal_step_kpi.recenter = "COMPLETE"
        if self.wrad is not None:
            output_models.meta.cal_step_kpi.window = "COMPLETE"

        self.log.info("--> Extract kerphase step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(ExtractKerphaseStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator

    def default_suffix(self):
        og_default = super(ExtractKerphaseStep, self).default_suffix()
        return f"{og_default}_kpfits"
