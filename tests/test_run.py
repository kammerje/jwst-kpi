# Test running the steps on their own and the full pipeline
# This is just to make sure no unexpected errors occur at runtime

from pathlib import Path
import pytest

from jwst_kpi import Kpi3Pipeline
from jwst_kpi.empirical_uncertainties.empirical_uncertainties_step import EmpiricalUncertaintiesStep
from jwst_kpi.extract_kerphase.extract_kerphase_step import ExtractKerphaseStep
from jwst_kpi.fix_bad_pixels.fix_bad_pixels_step import FixBadPixelsStep
from jwst_kpi.recenter_frames.recenter_frames_step import RecenterFramesStep
from jwst_kpi.trim_frames.trim_frames_step import TrimFramesStep
from jwst_kpi.window_frames.window_frames_step import WindowFramesStep

SHOW_PLOTS = False
PLOT = False
SAVE_RESULTS = False

@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data/NIRISS/CPD-67-607"


@pytest.fixture
def calints_file(data_dir: Path):
    return data_dir / "jw01093011001_03103_00001_nis_calints.fits"


# TODO: Add coparison with saved output for each test to make sure no unexpected changes
def test_trim_step(calints_file: Path):
    step = TrimFramesStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(calints_file)

def test_bad_pixels_step(data_dir: Path):
    input_file = data_dir / "jw01093011001_03103_00001_nis_trimframesstep.fits"
    step = FixBadPixelsStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(input_file)

def test_recenter_step(data_dir: Path):
    input_file = data_dir / "jw01093011001_03103_00001_nis_fixbadpixelsstep.fits"
    step = RecenterFramesStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(input_file)

def test_window_step(data_dir: Path):
    input_file = data_dir / "jw01093011001_03103_00001_nis_recenterframesstep.fits"
    step = WindowFramesStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(input_file)

def test_extract_step(data_dir: Path):
    input_file = data_dir / "jw01093011001_03103_00001_nis_windowframesstep.fits"
    step = ExtractKerphaseStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(input_file)

def test_empirical_uncertainties_step(data_dir: Path):
    input_file = data_dir / "jw01093011001_03103_00001_nis_extractkerphasestep_kpfits.fits"
    step = EmpiricalUncertaintiesStep()
    step.show_plots = SHOW_PLOTS
    step.plot = PLOT
    step.save_results = SAVE_RESULTS
    step.run(input_file)

def test_pipeline(calints_file: Path):
    kpi3_pipe = Kpi3Pipeline()
    kpi3_pipe.show_plots = SHOW_PLOTS
    kpi3_pipe.plot = PLOT
    kpi3_pipe.save_results = SAVE_RESULTS
    kpi3_pipe.run(calints_file)
