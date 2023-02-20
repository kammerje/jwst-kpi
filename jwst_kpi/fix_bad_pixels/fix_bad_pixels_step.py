import os

import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.datamodels.dqflags import pixel as pxdq_flags
from jwst.stpipe import Step

from jwst_kpi.fix_bad_pixels.bp_medfilt_method import fix_bp_medfilt

from .. import utils as ut
from ..datamodels import BadPixCubeModel
from .fix_bad_pixels_plots import plot_badpix


class FixBadPixelsStep(Step):
    """
    Fix bad pixels.

    ..Notes:: References for the Fourier method:
              https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
              https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract

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
        Method used to correct bad pixels
    bad_bits : List[str]
        Bad pixel codes to consider as bad when correcting image
    method_allowed : List[str]
        Bad pixel correction methods allowed.
        Default ['medfilt', 'fourier'] should not be changed by users.
    bad_bits_allowed : Optional[List[str]]
        List of values allowed for "bad_bits" attribute.
        Default list imported from JWST pipeline.
    """

    class_alias = "fix_bad_pixels"

    spec = """
        plot = boolean(default=True)
        method = string(default='medfilt')
        previous_suffix = string(default=None)
        bad_bits = string_list(default=list('DO_NOT_USE'))
        method_allowed = string_list(default=list('medfilt', 'fourier'))
        medfilt_size = integer(default=5)
        bad_bits_allowed = string_list(default=None)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(self, input_data):

        self.log.info("--> Running fix bad pixels step...")

        bad_bits_allowed = self.bad_bits_allowed or list(pxdq_flags.keys())
        good_frames = self.good_frames

        # Open file.
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        data = input_models.data
        erro = input_models.err
        pxdq = input_models.dq
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

        # Make bad pixel map.
        mask = pxdq < 0
        for i in range(len(self.bad_bits)):
            if self.bad_bits[i] not in bad_bits_allowed:
                raise UserWarning("Unknown data quality flag")
            else:
                pxdq_flag = pxdq_flags[self.bad_bits[i]]
                mask = mask | (pxdq & pxdq_flag == pxdq_flag)
                if i == 0:
                    bb = self.bad_bits[i]
                else:
                    bb += ", " + self.bad_bits[i]

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
            data_orig = data.copy()
            erro_orig = erro.copy()
            pxdq_orig = pxdq.copy()
            mask_orig = mask.copy()
            data = data[good_frames]
            erro = erro[good_frames]
            pxdq = pxdq[good_frames]
            mask = mask[good_frames]
        self.log.info(
            "--> Found %.0f bad pixels (%.2f%%)"
            % (
                np.sum(mask),
                np.sum(mask) / np.prod(mask.shape) * 100.0,
            )
        )

        # Fix bad pixels.
        if self.method == "medfilt":
            data_bpfixed, erro_bpfixed = fix_bp_medfilt(
                data, erro, mask, medfilt_size=self.medfilt_size
            )
            mask_mod = mask.copy()
        elif self.method not in self.method_allowed:
            raise ValueError(f"Unknown bad pixel cleaning method '{self.method}'")
        else:
            raise NotImplementedError(
                f"{self.method} bad pixel cleaning method not implemented yet"
            )

        # Get output file path.
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_badpix(data, data_bpfixed, bb, mask, method=self.method)
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        if good_frames is not None:
            data_final = data_orig.copy()
            erro_final = erro_orig.copy()
            pxdq_final = pxdq_orig.copy()
            mask_final = mask_orig.copy()
            data_final[good_frames] = data_bpfixed
            erro_final[good_frames] = erro_bpfixed
            pxdq_final[good_frames] = pxdq
            mask_final[good_frames] = mask_mod
            data_bpfixed = data_final
            erro_bpfixed = erro_final
            pxdq = pxdq_final
            mask_mod = mask_final

        # Save file.
        output_models = BadPixCubeModel()
        output_models.update(input_models, extra_fits=True)
        if is2d:
            data_bpfixed = data_bpfixed[0]
            erro_bpfixed = erro_bpfixed[0]
            pxdq = pxdq[0]
            mask_mod = mask_mod[0]
        output_models.data = data_bpfixed
        output_models.err = erro_bpfixed
        output_models.dq = pxdq
        output_models.meta.kpi_preprocess.fix_meth = self.method
        output_models.dq_mod = mask_mod.astype("uint32")
        output_models.meta.kpi_preprocess.bad_bits = bb
        output_models.meta.kpi_preprocess.msize = self.medfilt_size
        output_models.meta.cal_step_kpi.fix_badpix = "COMPLETE"

        self.log.info("--> Fix bad pixels step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(FixBadPixelsStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
