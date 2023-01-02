from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from jwst_kpi import Kpi3Pipeline


# =============================================================================
# MAIN
# =============================================================================

# Initialize stage 3 pipeline.
pipe3 = Kpi3Pipeline()
pipe3.output_dir = None # output directory; if None, uses same directory as input data
pipe3.show_plots = False # show plots?
pipe3.good_frames = None # list of good frames, bad frames will be skipped

# Trimming step.
pipe3.trim_frames.skip = False # skip step?
pipe3.trim_frames.plot = True # make and save plots?
pipe3.trim_frames.trim_cent = None # y (axis 0), x (axis 1); center of trimmed frames; if None uses brightest source
pipe3.trim_frames.trim_halfsize = 32 # pix; half size of trimmed frames

# Bad pixel fixing step.
pipe3.fix_bad_pixels.skip = False # skip step?
pipe3.fix_bad_pixels.plot = True # make and save plots?
pipe3.fix_bad_pixels.bad_bits = ['DO_NOT_USE'] # DQ flags to be considered as bad pixels (see https://jwst-reffiles.stsci.edu/source/data_quality.html)
pipe3.fix_bad_pixels.method = 'medfilt' # method to fix bad pixels; 'medfilt' or 'KI'

# Recentering step.
pipe3.recenter_frames.skip = False # skip step?
pipe3.recenter_frames.plot = True # make and save plots?
pipe3.recenter_frames.method = 'FPNM' # XARA recentering method; 'BCEN', 'COGI', or 'FPNM'
pipe3.recenter_frames.bmax = 6. # m; maximum baseline length for FPNM recentering method
pipe3.recenter_frames.pupil_path = None # path of custom XARA pupil model to be used; if None, uses default model
pipe3.recenter_frames.verbose = False # print telemetry data?

# Windowing step.
pipe3.window_frames.skip = False # skip step?
pipe3.window_frames.plot = True # make and save plots?
pipe3.window_frames.wrad = 24 # pix; radius of super-Gaussian window mask

# Kernel phase extraction step.
pipe3.extract_kerphase.skip = False # skip step?
pipe3.extract_kerphase.plot = True # make and save plots?
pipe3.extract_kerphase.bmax = None # m; maximum baseline length for kernel phase extraction; if None, uses entire pupil model
pipe3.extract_kerphase.pupil_path = None # path of custom XARA pupil model to be used; if None, uses default model

# Empirical uncertainties step.
pipe3.empirical_uncertainties.skip = False # skip step?
pipe3.empirical_uncertainties.plot = True # make and save plots?
pipe3.empirical_uncertainties.get_emp_err = True # estimate uncertainties empirically from standard deviation over individual frames?
pipe3.empirical_uncertainties.get_emp_cor = False # estimate correlations empirically from individual frames?

# Run stage 3 pipeline.
pipe3.run('testdata/NIRISS/jw01093010001_03103_00001_nis_calints.fits')

pdb.set_trace()
