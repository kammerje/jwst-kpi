"""
JWST stage 3 pipeline for kernel phase imaging.

Authors: Jens Kammerer, Thomas Vandal, Katherine Thibault, Frantz Martinache
Supported instruments: NIRCam, NIRISS, MIRI
"""
import logging
import os

import matplotlib

from jwst.stpipe import Pipeline
from jwst import datamodels

from ..extract_kerphase import extract_kerphase_step
from ..empirical_uncertainties import empirical_uncertainties_step
from ..fix_bad_pixels import fix_bad_pixels_step
from ..recenter_frames import recenter_frames_step
from ..trim_frames import trim_frames_step
from ..window_frames import window_frames_step
from . import pupil_data

matplotlib.rcParams.update({"font.size": 14})


__all__ = ["Kpi3Pipeline"]

# Define logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

PUPIL_DIR = pupil_data.__path__[0]


# TODO: Add included steps
class Kpi3Pipeline(Pipeline):
    """
    JWST stage 3 pipeline for kernel phase imaging.

    ..Notes:: AMI skips the ipc, photom, and resample steps in the stage 1 & 2
              pipelines. It is recommended to also skip these steps for kernel
              phase imaging.
    """

    class_alias = "calwebb_kpi3"

    # TODO: Add parameters
    # TODO: Does None work for list?
    spec = """
        output_dir = string(default=None)
        show_plots = boolean(default=False)
        good_frames = int_list(default=None)
    """

    # TODO: Replace with proper step names (like real pipeline)
    step_defs = {
        "trim_frames": trim_frames_step.trim_frames,
        "fix_bad_pixels": fix_bad_pixels_step.fix_bad_pixels,
        "recenter_frames": recenter_frames_step.recenter_frames,
        "window_frames": window_frames_step.window_frames,
        "extract_kerphase": extract_kerphase_step.extract_kerphase,
        "empirical_uncertainties": empirical_uncertainties_step.empirical_uncertainties,
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
        # TODO: Handle output (at step level?)
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # TODO: Preserve "step" name for their process method somewhere?
        with datamodels.open(input_data) as input:

            # NOTE: Skipped steps are skipped in their own run/process
            # TODO: Handle recenter_frames/window_frames passed to extract_kerphase in old version
            # ("internally in complex visibility space")
            input = self.trim_frames(input)
            input = self.fix_bad_pixels(input)
            input = self.recenter_frames(input)
            input = self.window_frames(input)
            input = self.extract_kerphase(input)
            input = self.empirical_uncertainties(input)

        return input
