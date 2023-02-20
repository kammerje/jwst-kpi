"""
Constants used throughout KPI pipeline
"""
from . import pupil_data

# Detector pixel scales.
# TODO: assumes that NIRISS pixels are square but they are slightly
#       rectangular.
# TODO: Make uppercase to avoid conflicts with variable names
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview
pscale = {
    "NIRCAM_SHORT": 31.0,  # mas
    "NIRCAM_LONG": 63.0,  # mas
    "NIRISS": 65.55,  # mas
    "MIRI": 110.0,  # mas
}

PUPIL_DIR = pupil_data.__path__[0]

# All from JDocs "Detector Performance" pages
READ_NOISE = {
    "NIRCAM_SHORT": 15.77,  # e-
    "NIRCAM_LONG": 13.25,  # e-
    "NIRISS": 16.8,  # e-
    "MIRI": 6.0,  # e-
}

# Detector gains.
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview/nircam-detector-performance
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview/niriss-detector-performance
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview/miri-detector-performance
gain = {
    "NIRCAM_SHORT": 2.05,  # e-/ADU
    "NIRCAM_LONG": 1.82,  # e-/ADU
    "NIRISS": 1.61,  # e-/ADU
    "MIRI": 4.0,  # e-/ADU
}

WRAD_DEFAULT = 24

DIAM = 6.559348  # / Flat-to-flat distance across pupil in V3 axis
PUPLDIAM = 6.603464  # / Full pupil file size, incl padding.
