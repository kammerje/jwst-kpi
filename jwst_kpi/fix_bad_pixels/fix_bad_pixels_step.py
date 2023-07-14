import os

import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.datamodels.dqflags import pixel as pxdq_flags
from jwst.stpipe import Step
from scipy.ndimage import median_filter

from .. import utils as ut
from ..datamodels import BadPixCubeModel, BadPixImageModel
from .fix_bad_pixels_plots import plot_badpix
from . import fourier_bpclean
from ..constants import (gain, pscale, wave_miri, wave_nircam, wave_niriss,
                         weff_miri, weff_nircam, weff_niriss, DIAM)


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
    instrume_allowed : List[str]
        Instruments available with this step.
        Default ['NIRCAM', 'NIRISS', 'MIRI'] should not be changed by users.
    do_bg_sub : bool
        Do simple background subtraction to avoid FT discontinuities when zero-padding.
    temp_sig_thresh : Optional[float]
        Number of sigmas to use for temporal bad outlier identification.
        None by default, i.e. the temporal identification is skipped.
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
        instrume_allowed = string_list(default=list('NIRCAM', 'NIRISS', 'MIRI'))
        bad_bits_allowed = string_list(default=None)
        show_plots = boolean(default=False)
        do_bg_sub = boolean(default=True)
        temp_sig_thresh = float(default=None)
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


        # Get observing setup info from data model
        pupil_name = input_models.meta.instrument.pupil
        INSTRUME = input_models.meta.instrument.name
        if INSTRUME not in self.instrume_allowed:
            raise ValueError("Unsupported instrument")
        FILTER = input_models.meta.instrument.filter

        # Get detector pixel scale and PA
        # TODO: Dup code with extract and badpix step
        if INSTRUME == "NIRCAM":
            CHANNEL = input_models.meta.instrument.channel
            PSCALE = pscale[INSTRUME + "_" + CHANNEL]  # mas
        else:
            PSCALE = pscale[INSTRUME]  # mas


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

        # Get filter properties.
        if INSTRUME == "NIRCAM":
            wave = wave_nircam[FILTER] * 1e-6 # m
            weff = weff_nircam[FILTER] * 1e-6 # m
        elif INSTRUME == "NIRISS":
            wave = wave_niriss[FILTER] * 1e-6 # m
            weff = weff_niriss[FILTER] * 1e-6 # m
        elif INSTRUME == "MIRI":
            wave = wave_miri[FILTER] * 1e-6 # m
            weff = weff_miri[FILTER] * 1e-6 # m

        # Make bad pixel map.
        mask = np.isnan(data)
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

        self.log.info(
            "--> Found %.0f bad pixels (%.2f%%)"
            % (
                np.sum(mask[good_frames]),
                np.sum(mask[good_frames]) / np.prod(mask[good_frames].shape) * 100.0,
            )
        )

        # Simple background subtraction to avoid discontinuity when zero-padding the data in XARA.
        if self.do_bg_sub:
            data_temp = data.copy()
            data_temp[mask] = np.nan
            data -= np.nanmedian(data_temp, axis=(1, 2), keepdims=True)

        # Find new bad pixels based on temporal evolution.
        if self.temp_sig_thresh is not None:
            nf_good = nf if good_frames is None else len(good_frames)
            if nf_good >= 3:
                if good_frames is None:
                    data_med = np.nanmedian(data, axis=0)
                    data_std = np.nanstd(data, axis=0)
                else:
                    data_med = np.nanmedian(data[good_frames], axis=0)
                    data_std = np.nanstd(data[good_frames], axis=0)
                mask[data > data_med + self.temp_sig_thresh * data_std] = 1
                mask[data < data_med - self.temp_sig_thresh * data_std] = 1

        # Fix bad pixels.
        data_bpfixed = data.copy()
        data_bpfixed[mask] = 0.
        erro_bpfixed = erro.copy()
        erro_bpfixed[mask] = 0.
        if self.method not in self.method_allowed:
            raise UserWarning("Unknown bad pixel cleaning method")
        else:
            if self.method == "medfilt":
                for i in range(nf):
                    if (
                        good_frames is None
                        or i in good_frames
                        or (nf - i) * (-1) in good_frames
                    ):
                        data_bpfixed[i][mask[i]] = median_filter(
                            data_bpfixed[i], size=5
                        )[mask[i]]
                        erro_bpfixed[i][mask[i]] = median_filter(
                            erro_bpfixed[i], size=5
                        )[mask[i]]
            elif self.method == "fourier":
                for i in range(nf):
                    if (
                        good_frames is None
                        or i in good_frames
                        or (nf - i) * (-1) in good_frames
                    ):
                        data_bpfixed[i], mask[i] = fourier_bpclean.run(
                            data_bpfixed[i],
                            mask[i],
                            INSTRUME,
                            FILTER,
                            pupil_name,
                            PSCALE,
                            wave,
                            weff,
                            find_new=True,
                        )
                        # erro_bpfixed[i] = fourier_bpclean.run(erro_bpfixed[i], mask[i], INSTRUME, FILTER, PUPIL, PSCALE, wave, weff, find_new=False)
                        erro_bpfixed[i][mask[i]] = median_filter(erro_bpfixed[i], size=5)[mask[i]]

        # Get output file path.
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_badpix(data, data_bpfixed, bb, mask, good_frames, method=self.method)
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            output_models = BadPixImageModel()
            data_bpfixed = data_bpfixed[0]
            erro_bpfixed = erro_bpfixed[0]
            pxdq = pxdq[0]
            mask = mask[0]
        else:
            output_models = BadPixCubeModel()
        output_models.update(input_models, extra_fits=True)
        output_models.data = data_bpfixed
        output_models.err = erro_bpfixed
        output_models.dq = pxdq
        output_models.meta.kpi_preprocess.fix_meth = self.method
        output_models.dq_mod = mask.astype("uint32")
        output_models.bad_bits = bb
        output_models.meta.cal_step_kpi.fix_badpix = "COMPLETE"

        self.log.info("--> Fix bad pixels step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(FixBadPixelsStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
