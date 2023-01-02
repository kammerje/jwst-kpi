"""
Example script showing how to assess pupil model quality using:
    - Baseline redundancy
    - Fourier amplitude

For a point source, a good model will make the baseline redundancy match the
Fourier amplitude and will make the Fourier phase small.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import xara
from astropy.io import fits
from scipy.ndimage import median_filter

# %%
# Use pupil model for NIRISS generated with generate_pupil_model.py
KPO = xara.kpo.KPO(fname="pupil_results/niriss_clear_pupil.fits")

# %%
# Load a point source commissioning observation
file = "kerphase_testdata/NIRISS/TYC-8906-1660-1/jw01093022001_03103_00001_nis_calints.fits"

with fits.open(file) as hdul:
    data = hdul["SCI"].data
    pxdq = hdul["DQ"].data

PSCALE = 65.6
wave = 4.813019 * 1e-6
wrad = 15


# %%
# Extract kernel phase for each frame, median filter for bad pixels before
for i in range(data.shape[0]):
    ww = pxdq[i] & 1 == 1
    data[i][ww] = median_filter(data[i], size=5)[ww]
    KPO.extract_KPD_single_frame(
        data[i],
        PSCALE,
        wave,
        target=None,
        recenter=True,
        wrad=wrad,
        method="LDFT1",
        algo_cent="FPNM",
        bmax_cent=4.0,
    )

# %%
# Put CVIS in an array (list by default) and get mean values
cvis = np.concatenate(KPO.CVIS)
fa = np.mean(np.abs(cvis), axis=0)
fp = np.mean(np.angle(cvis), axis=0)
bred = KPO.kpi.RED

# %%
f = plt.figure()
ax = plt.gca()
ax.plot(
    fa[np.argsort(KPO.kpi.BLEN)],
    label="Source, RMS = %.2f" % np.sqrt(np.mean((fa - bred) ** 2)),
)
ax.plot(bred[np.argsort(KPO.kpi.BLEN)], label="Baseline redundancy")
ax.set_xlabel("Index sorted by baseline length")
ax.set_ylabel("Fourier amplitude")
ax.grid(axis="y")
ax.legend(loc="upper right")
plt.show(block=True)
# plt.close()

# %%
f = plt.figure()
ax = plt.gca()
ax.plot(fp[np.argsort(KPO.kpi.BLEN)])
ax.set_xlabel("Index sorted by baseline length")
ax.set_ylabel("Fourier phase")
ax.grid(axis="y")
plt.show(block=True)
# plt.close()
