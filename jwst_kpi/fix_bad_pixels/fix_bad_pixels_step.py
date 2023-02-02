import os

import matplotlib.pyplot as plt
import numpy as np
# from astropy.io import fits
from jwst.stpipe import Step
from jwst import datamodels
from scipy.ndimage import median_filter

# from .. import utils as ut
from .fix_bad_pixels_plots import plot_badpix

# Bad pixel flags.
# https://jwst-reffiles.stsci.edu/source/data_quality.html
# TODO: import from jwst pipeline?
pxdq_flags = {
    "DO_NOT_USE": 1,
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


class FixBadPixelsStep(Step):
    """
    Fix bad pixels.

    ..Notes:: References for the Fourier method:
              https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract
              https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract
    """

    class_alias = "fix_bad_pixels"

    spec = """
        plot = boolean(default=True)
        method = string(default='medfilt')
        previous_suffix = string(default=None)
        bad_bits = string_list(default=['DO_NOT_USE'])
        method_allowed = string_list(default=['medfilt', 'fourier'])
        bad_bits_allowed = string_list(default=None)
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
        output_dir: str
            Output directory, if None uses same directory as input file.
        show_plots: bool
            Show plots?
        good_frames: list of int
            List of good frames, bad frames will be skipped.
        """

        self.log.info("--> Running fix bad pixels step...")

        bad_bits_allowed = self.bad_bits_allowed or list(pxdq_flags.keys())
        good_frames = self.good_frames

        # Open file.
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

        # Suffix for the file path from the current step.
        suffix_out = f"_{self.suffix or self.default_suffix()}"

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

        self.log.info(
            "--> Found %.0f bad pixels (%.2f%%)"
            % (
                np.sum(mask[good_frames]),
                np.sum(mask[good_frames]) / np.prod(mask[good_frames].shape) * 100.0,
            )
        )

        # Fix bad pixels.
        data_bpfixed = data.copy()
        erro_bpfixed = erro.copy()
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
                raise NotImplementedError(
                    "Fourier bad pixel cleaning method not implemented yet"
                )

        # Get output file path.
        # path = ut.get_output_base(file, output_dir=output_dir)
        mk_path = self.make_output_path()
        path = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_badpix(data, data_bpfixed, bb, mask, good_frames, method=self.method)
            plt.savefig(path + suffix_out + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        # TODO: Mark step completed
        # TODO: How add keywords in pipeline?
        # TODO: Add mask ext
        # TODO: Might want to just save this direclty here instead of using default saving mech
        output_models = input_models.copy()
        if is2d:
            data_bpfixed = data_bpfixed[0]
            erro_bpfixed = erro_bpfixed[0]
            mask = mask[0]
        output_models.data = data_bpfixed
        output_models.err = erro_bpfixed
        # hdul["SCI"].data = data_bpfixed
        # hdul["SCI"].header["FIX_METH"] = self.method
        # hdul["ERR"].data = erro_bpfixed
        # hdul["ERR"].header["FIX_METH"] = self.method
        # hdu_dq_mod = fits.ImageHDU(mask.astype("uint32"))
        # hdu_dq_mod.header["EXTNAME"] = "DQ-MOD"
        # hdu_dq_mod.header["BAD_BITS"] = bb
        # hdul += [hdu_dq_mod]
        # hdul.writeto(path + suffix_out + ".fits", output_verify="fix", overwrite=True)
        # hdul.close()

        self.log.info("--> Fix bad pixels step done")

        return output_models
