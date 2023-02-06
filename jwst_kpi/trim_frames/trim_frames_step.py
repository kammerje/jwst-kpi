import os

import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
from scipy.ndimage import median_filter

from .. import utils as ut
from ..datamodels import TrimmedCubeModel
from .trim_frames_plots import plot_trim


class TrimFramesStep(Step):
    """
    Trim frames.

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
    trim_cent : List[int]
        Center around which to trim (X and Y coordinates, list of length 2)
    trim_halfsize: int
        Half-size of trimmed image (default 32).
    """

    class_alias = "trim_frames"

    # NOTE: output_dir and suffix now handled automatically by parent Step class
    spec = """
        plot = boolean(default=True)
        previous_suffix = string(default=None)
        trim_cent = int_list(min=2, max=2, default=None)
        trim_halfsize = integer(default=32)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    def process(self, input_data):

        self.log.info("--> Running trim frames step...")

        good_frames = self.good_frames

        # Open file. Use default pipeline way unless "previous suffix is used"
        if self.previous_suffix is None:
            input_models = datamodels.open(input_data)
        else:
            raise ValueError("Unexpected previous_suffix attribute")
        data = input_models.data
        erro = input_models.err
        pxdq = input_models.dq
        if data.ndim not in [2, 3]:
            raise ValueError("Only implemented for 2D image/3D data cube")
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

        # Trim frames. Need to trim all frames (also bad ones) to match their
        # shapes.
        if self.trim_cent is None:
            ww_max = []
            for i in range(nf):
                ww_max += [
                    np.unravel_index(
                        np.nanargmax(median_filter(data[i], size=3)), data[i].shape
                    )
                ]
            ww_max = np.array(ww_max)
        else:
            ww_max = np.array([[self.trim_cent[0], self.trim_cent[1]]] * nf)
        if good_frames is None:
            if np.max(np.abs(ww_max - np.mean(ww_max, axis=0))) > 3.0:
                raise UserWarning(
                    "Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position"
                )
            ww_max = [
                int(round(np.median(ww_max[:, 0]))),
                int(round(np.median(ww_max[:, 1]))),
            ]
        else:
            if (
                np.max(
                    np.abs(ww_max[good_frames] - np.mean(ww_max[good_frames], axis=0))
                )
                > 3.0
            ):
                raise UserWarning(
                    "Brightest source jitters by more than 3 pixels, use trim_frames.trim_cent to specify source position"
                )
            ww_max = [
                int(round(np.median(ww_max[good_frames, 0]))),
                int(round(np.median(ww_max[good_frames, 1]))),
            ]
        xh_max = min(sx - ww_max[1], ww_max[1])
        yh_max = min(sy - ww_max[0], ww_max[0])
        sh_max = min(xh_max, yh_max)
        self.trim_halfsize = min(sh_max, self.trim_halfsize)

        self.log.info(
            "--> Trimming frames around (x, y) = (%.0f, %.0f) to a size of %.0fx%.0f pixels"
            % (ww_max[1], ww_max[0], 2 * self.trim_halfsize, 2 * self.trim_halfsize)
        )

        # TODO: Unused loop?? (repeating vectorized code)
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

        # NOTE: This is duplicated in all methods
        mk_path = self.make_output_path()
        stem = os.path.splitext(mk_path)[0]

        # Plot.
        if self.plot:
            plot_trim(
                data,
                data_trimmed,
                pxdq,
                pxdq_trimmed,
                ww_max,
                self.trim_halfsize,
                good_frames=good_frames,
            )
            plt.savefig(stem + ".pdf")
            if self.show_plots:
                plt.show()
            plt.close()

        # Save file.
        if is2d:
            data_trimmed = data_trimmed[0]
            erro_trimmed = erro_trimmed[0]
            pxdq_trimmed = pxdq_trimmed[0]
        output_models = TrimmedCubeModel()
        output_models.update(input_models)
        output_models.data = data_trimmed
        output_models.err = erro_trimmed
        output_models.dq = pxdq_trimmed
        output_models.meta.kpi_preprocess.cent_x = ww_max[1]
        output_models.meta.kpi_preprocess.cent_y = ww_max[0]
        output_models.meta.cal_step_kpi.trim = "COMPLETE"

        self.log.info("--> Trim frames step done")

        return output_models

    def remove_suffix(self, name):
        new_name, separator = super(TrimFramesStep, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator
