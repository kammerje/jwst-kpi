import os

from jwst_kpi.trim_frames.trim_frames_step import TrimFramesStep

input_file = "examples/kerphase_testdata/NIRISS/CPD-67-607/jw01093011001_03103_00001_nis_calints.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

trimstep = TrimFramesStep()
trimstep.show_plots = True
trimstep.save_results = True
trimstep.output_dir = output_dir
output = trimstep.run(input_file)
