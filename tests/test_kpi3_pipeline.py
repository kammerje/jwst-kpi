import os

from jwst_kpi import Kpi3Pipeline

input_file = "examples/kerphase_testdata/NIRISS/CPD-67-607/jw01093011001_03103_00001_nis_calints.fits"
output_dir = "tests/outputs/"
os.makedirs(output_dir, exist_ok=True)

kpi3_pipe = Kpi3Pipeline()
kpi3_pipe.output_dir = output_dir
kpi3_pipe.save_results = True
output = kpi3_pipe.run(input_file)
