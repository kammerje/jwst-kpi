import os

from jwst_kpi.fix_bad_pixels.fix_bad_pixels_step import FixBadPixelsStep

input_file = "tests/outputs/jw01093011001_03103_00001_nis_trimframesstep.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

step = FixBadPixelsStep()
step.show_plots = True
step.save_results = True
step.output_dir = output_dir
output = step.run(input_file)
print(output)
