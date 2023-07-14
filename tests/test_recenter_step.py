import os

from jwst_kpi.recenter_frames.recenter_frames_step import RecenterFramesStep

input_file = "tests/outputs/jw01093011001_03103_00001_nis_fixbadpixelsstep.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

step = RecenterFramesStep()
step.show_plots = True
step.save_results = True
step.output_dir = output_dir
output = step.run(input_file)
print(output)
