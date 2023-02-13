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
from .. import utils as ut
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

    Parameters
    -----------
    input_data :  ~jwst_kpi.datamodels.KPFitsModel
        Single filename for extracted kernel phase data
    plot : bool
        Generate plots
    show_plots : bool
        Show plots
    good_frames : List[int]
        List of good frames, bad frames will be skipped.
    """

    class_alias = "calwebb_kpi3"

    spec = """
        output_dir = string(default=None)
        show_plots = boolean(default=False)
        plot = boolean(default=True)
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
    def process(self, input_data):

        log.info("Starting calwebb_kpi3")

        # Make the output directory if it does not exist.
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # Propagate plot, show_plots, output_dir, good_frames
        for step_name in self.step_defs:
            step = getattr(self, step_name)
            step.show_plots = self.show_plots
            step.plot = self.plot
            step.output_dir = self.output_dir
            if hasattr(step, "good_frames"):
                step.good_frames = self.good_frames

        input = datamodels.open(input_data)

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
        has_org = hasattr(input, "data_org")
        if has_org and self.recenter_frames.skip and self.window_frames.skip:
            raise RuntimeError(
                "There is a SCI-ORG FITS file extension although both the recentering and the windowing steps were skipped."
            )
        input = self.extract_kerphase(input)

        input = self.empirical_uncertainties(input)

        if not self.empirical_uncertainties.skip:
            self.suffix = "emp_kpfits"
        elif not self.extract_kerphase.skip:
            self.suffix = "kpfits"
        else:
            self.suffix = "calints_kpi"

        return input

    def remove_suffix(self, name):
        new_name, separator = super(Kpi3Pipeline, self).remove_suffix(name)
        if new_name == name:
            new_name, separator = ut.remove_suffix_kpi(new_name)
        return new_name, separator

    def finalize_result(self, result, reference_files_used):
        # Override this function to avoid error about CRDS: we don't use it
        log.info("KPI Pipeline does not rely on CRDS for reduction")
