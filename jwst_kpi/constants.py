from astroquery.svo_fps import SvoFps

# Detector pixel scales.
# TODO: assumes that NIRISS pixels are square but they are slightly
#       rectangular.
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

# TODO: Convert to function?
# Load the NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile
# Service.
# http://svo2.cab.inta-csic.es/theory/fps/
wave_nircam = {}
weff_nircam = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRCAM")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_nircam[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_nircam[name] = filter_list["WidthEff"][i] / 1e4  # micron
wave_niriss = {}
weff_niriss = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="NIRISS")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_niriss[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_niriss[name] = filter_list["WidthEff"][i] / 1e4  # micron
wave_miri = {}
weff_miri = {}
filter_list = SvoFps.get_filter_list(facility="JWST", instrument="MIRI")
for i in range(len(filter_list)):
    name = filter_list["filterID"][i]
    name = name[name.rfind(".") + 1 :]
    wave_miri[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
    weff_miri[name] = filter_list["WidthEff"][i] / 1e4  # micron
del filter_list
