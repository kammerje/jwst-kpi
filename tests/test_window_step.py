import os

from jwst_kpi.window_frames import WindowFramesStep

input_file = "tests/outputs/jw01093011001_03103_00001_nis_recenterframesstep.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

step = WindowFramesStep()
step.show_plots = True
step.save_results = True
step.output_dir = output_dir
output = step.run(input_file)
print(output)
