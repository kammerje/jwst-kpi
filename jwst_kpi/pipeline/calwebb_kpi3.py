"""
JWST stage 3 pipeline for kernel phase imaging.

Authors: Jens Kammerer, Thomas Vandal, Katherine Thibault, Frantz Martinache
Supported instruments: NIRCam, NIRISS, MIRI
"""
import logging
import os

import matplotlib
from jwst import datamodels
from jwst.stpipe import Pipeline

from .. import pupil_data
from ..empirical_uncertainties import empirical_uncertainties_step
from ..extract_kerphase import extract_kerphase_step
from ..fix_bad_pixels import fix_bad_pixels_step
from ..recenter_frames import recenter_frames_step
from ..trim_frames import trim_frames_step
from ..window_frames import window_frames_step

matplotlib.rcParams.update({"font.size": 14})


__all__ = ["Kpi3Pipeline"]

# Define logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

PUPIL_DIR = pupil_data.__path__[0]


class Kpi3Pipeline(Pipeline):
    """
    JWST stage 3 pipeline for kernel phase imaging.

    ..Notes:: AMI skips the ipc, photom, and resample steps in the stage 1 & 2
              pipelines. It is recommended to also skip these steps for kernel
              phase imaging (i.e. when generating calints that will be passed
              to this pipeline).
    """

    class_alias = "calwebb_kpi3"

    spec = """
        output_dir = string(default=None)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    step_defs = {
        "trim_frames": trim_frames_step.TrimFramesStep,
        "fix_bad_pixels": fix_bad_pixels_step.FixBadPixelsStep,
        "recenter_frames": recenter_frames_step.RecenterFramesStep,
        "window_frames": window_frames_step.WindowFramesStep,
        "extract_kerphase": extract_kerphase_step.ExtractKerphaseStep,
        "empirical_uncertainties": empirical_uncertainties_step.EmpiricalUncertaintiesStep,
    }

    # NOTE: `run` will now call this directly because subclass Pipeline
    def process(
        self,
        input_data,
    ):
        """
        Run the pipeline.

        Parameters
        ----------
        file: str
            Path to stage 2-calibrated pipeline product.
        """

        log.info("Starting calwebb_kpi3")

        # Make the output directory if it does not exist.
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # Propagate show_plots, output_dir, good_frames
        for step_name in self.step_defs:
            step = getattr(self, step_name)
            step.show_plots = self.show_plots
            step.output_dir = self.output_dir
            if hasattr(step, "good_frames"):
                step.good_frames = self.good_frames

        with datamodels.open(input_data) as input:

            # NOTE: Skipped steps are skipped in their own run/process
            input = self.trim_frames(input)
            input = self.fix_bad_pixels(input)
            # TODO: Skip these two if extract is not skipped?
            input = self.recenter_frames(input)
            input = self.window_frames(input)

            if not self.recenter_frames.skip:
                self.extract_kerphase.recenter_method = self.recenter_frames.method
                self.extract_kerphase.recenter_bmax = self.recenter_frames.bmax
                self.extract_kerphase.recenter_method_allowed = (
                    self.recenter_frames.method_allowed
                )
            if not self.window_frames.skip:
                self.extract_kerphase.wrad = self.window_frames.wrad
            input = self.extract_kerphase(input)

            input = self.empirical_uncertainties(input)

        return input
