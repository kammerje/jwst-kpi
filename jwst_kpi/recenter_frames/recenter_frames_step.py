import os

import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
from xara import core, kpo

from jwst_kpi.datamodels import RecenterCubeModel

from .. import utils as ut
from ..constants import PUPIL_DIR, pscale
from .recenter_frames_plots import plot_recenter


class RecenterFramesStep(Step):
    """
    Recenter frames.

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
    method : str
        Recentering method to be used (defaults to FPNM)
    recenter_method_allowed : List[str]
        Recentering methods allowed.
        Default ['BCEN', 'COGI', 'FPNM'] should not be modified by users.
    instrume_allowed : List[str]
        Instruments allowed by the pipeline
        (default ['NIRCAM', 'NIRISS', 'MIRI'] should not be changed by users)
    bmax : float
        Maximum baseline to consider when recentering (in m, defaults to 6.0)
    pupil_path : Optional[str]
        Optional path to custom pupil model
    verbose : bool
        Enable verbose mode
    """

    class_alias = "recenter_frames"

    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        method = string(default='FPNM')
        method_allowed = string_list(default=list('BCEN', 'COGI', 'FPNM'))
        instrume_allowed = string_list(default=list('NIRCAM', 'NIRISS', 'MIRI'))
        bmax = float(default=6.0)
        pupil_path = string(default=None)
        verbose = boolean(default=False)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(self, input_data):

        self.log.info("--> Running recenter frames step...")

        good_frames = self.good_frames

        # Open file. Use default pipeline way unless "previous suffix is used"
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        # TODO: Make dq (and dq mod??) follow with erro and data
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

        # Simple background subtraction to avoid discontinuity when zero-padding the data in XARA.
        data -= np.median(data, axis=(1, 2), keepdims=True)

        # Get detector pixel scale and position angle.
        if INSTRUME == "NIRCAM":
            CHANNEL = input_models.meta.instrument.channel
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas
        V3I_YANG = (
            -input_models.meta.wcsinfo.v3yangle * input_models.meta.wcsinfo.vparity
        )  # deg, counter-clockwise

        if "FPNM" in self.method:

            # Check if the filter is known.
            wave_inst, weff_inst = ut.get_wavelengths(INSTRUME)
            filter_allowed = wave_inst.keys()

            if FILTER not in filter_allowed:
                raise ValueError(
                    f"Unknown filter name {FILTER}. Available filters for instrument "
                    f"{INSTRUME} are: {filter_allowed}"
                )

            wave = wave_inst[FILTER] * 1e-6  # m
            weff = weff_inst[FILTER] * 1e-6  # m

            # Get pupil model path and filter properties.
            pupil_name = input_models.meta.instrument.pupil
            # TODO: Handle pupil path/loading in util function
            if INSTRUME == "NIRCAM":
                if self.pupil_path is None:
                    if pupil_name == "MASKRND":
                        default_pupil_model = "nircam_rnd_pupil.fits"
                    elif pupil_name == "MASKBAR":
                        default_pupil_model = "nircam_bar_pupil.fits"
                    else:
                        default_pupil_model = "nircam_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
            elif INSTRUME == "NIRISS":
                if self.pupil_path is None:
                    if pupil_name == "NRM":
                        default_pupil_model = "niriss_nrm_pupil.fits"
                    else:
                        default_pupil_model = "niriss_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)
            elif INSTRUME == "MIRI":
                if self.pupil_path is None:
                    default_pupil_model = "miri_clear_pupil.fits"
                    self.pupil_path = os.path.join(PUPIL_DIR, default_pupil_model)

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
            raise ValueError("Unknown recentering method")
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

        # Get output file path.
        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_recenter(data, data_recentered, dx, dy, good_frames=good_frames)
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        output_models = RecenterCubeModel()
        output_models.update(input_models, extra_fits=True)
        if is2d:
            data = data[0]
            erro = erro[0]
            data_recentered = data_recentered[0]
            erro_recentered = erro_recentered[0]
        output_models.data = data_recentered
        output_models.err = erro_recentered
        output_models.data_org = data
        output_models.err_org = erro
        # TODO: DQ is not recentered. Should it be, at pixel level?
        output_models.dq = input_models.dq
        try:
            output_models.dq_mod = input_models.dq_mod
        except AttributeError:
            self.log.warning("Could not pass bad pixel mask from BP step")
        if is2d:
            dx_arr = np.array([dx])  # pix
            dy_arr = np.array([dy])  # pix
        else:
            dx_arr = np.array(dx)  # pix
            dy_arr = np.array(dy)  # pix
        output_models.imshift = np.recarray(dx_arr.shape, output_models.imshift.dtype)
        output_models.imshift["XSHIFT"] = dx_arr
        output_models.imshift["YSHIFT"] = dy_arr
        output_models.meta.cal_step_kpi.recenter = "COMPLETE"

        self.log.info("--> Recenter frames step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(RecenterFramesStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
