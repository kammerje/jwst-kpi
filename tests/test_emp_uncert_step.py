import os

from jwst_kpi.empirical_uncertainties import EmpiricalUncertaintiesStep

input_file = "tests/outputs/jw01093011001_03103_00001_nis_extractkerphasestep_kpfits.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

step = EmpiricalUncertaintiesStep()
step.show_plots = True
step.save_results = True
step.output_dir = output_dir
output = step.run(input_file)
print(output)
