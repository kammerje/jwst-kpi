"""
Constants used throughout KPI pipeline
"""

import logging
from jwst_kpi.utils import has_network_access, get_wave_local, get_wave_svo

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Detector pixel scales.
# TODO: assumes that NIRISS pixels are square but they are slightly rectangular.
# https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview
# https://jwst-docs.stsci.edu/jwst-near-infrared-imager-and-slitless-spectrograph/niriss-instrumentation/niriss-detector-overview
# https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-detector-overview
pscale = {
    "NIRCAM_SHORT": 31.0,  # mas
    "NIRCAM_LONG": 63.0,  # mas
    "NIRISS": 65.55,  # mas
    "MIRI": 110.0,  # mas
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


if has_network_access():
    wave_nircam, weff_nircam = get_wave_svo("NIRCAM")
    wave_niriss, weff_niriss = get_wave_svo("NIRISS")
    wave_miri, weff_miri = get_wave_svo("MIRI")
else:
    log.warning("No network access. Using local SVO files.")
    wave_nircam, weff_nircam = get_wave_local("NIRCAM")
    wave_niriss, weff_niriss = get_wave_local("NIRISS")
    wave_miri, weff_miri = get_wave_local("MIRI")


WRAD_DEFAULT = 24

DIAM = 6.559348  # / Flat-to-flat distance across pupil in V3 axis
