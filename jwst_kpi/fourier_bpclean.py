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

from copy import deepcopy
from poppy import matrixDFT

sys.path.append("/Users/jkammerer/Documents/Code/webbpsf_ext")
from webbpsf_ext import robust

from . import pupil_data
PUPIL_DIR = pupil_data.__path__[0]


# =============================================================================
# MAIN
# =============================================================================

WL_OVERSIZEFACTOR = 0.1 # increase filter wl support by this amount to oversize in wl space
PUPLDIAM = 6.603464 # m; full pupil file size, including padding

pupilfile_niriss_nrm = os.path.join(PUPIL_DIR, "MASK_NRM.fits")
pupilfile_niriss_clearp = os.path.join(PUPIL_DIR, "MASK_CLEARP.fits")
pupilfile_nircam_clear = os.path.join(PUPIL_DIR, "MASK_CLEAR.fits")
pupil_niriss_nrm = pyfits.getdata(pupilfile_niriss_nrm)
pupil_niriss_clearp = pyfits.getdata(pupilfile_niriss_clearp)
pupil_nircam_clear = pyfits.getdata(pupilfile_nircam_clear)
pupil_masks = {
    "NIRISS_NRM": pupil_niriss_nrm,
    "NIRISS_CLEARP": pupil_niriss_clearp,
    "NIRCAM_CLEAR": pupil_nircam_clear,
}

def calc_psf(wave, # m
             sz, # pix
             INSTRUME,
             PUPIL,
             PSCALE): # mas

    # Calculate resolution element.
    resel = wave / PUPLDIAM # rad
    nresel = sz * PSCALE / 1000. / 3600. / 180. * np.pi / resel

    # Initialize MFT object.
    ft = matrixDFT.MatrixFourierTransform()

    # Calculate PSF.
    temp = INSTRUME + "_" + PUPIL
    if INSTRUME == "NIRCAM":
        temp = "NIRCAM_CLEAR" # default to NIRCam clear pupil since PUPIL can be a filter for NIRCam imaging
    pupil_mask = pupil_masks[temp]
    image_field = ft.perform(pupil_mask, nresel, sz)
    image_intensity = (image_field * image_field.conj()).real

    return image_intensity

def calc_support(INSTRUME,
                 PUPIL,
                 PSCALE, # mas
                 wave, # m
                 weff, # m
                 sz): # pix

    # Calculate wavelength support.
    weffh = weff / 2. * (1. + WL_OVERSIZEFACTOR)
    waves = (wave - weffh, wave, wave + weffh)

    # Calculate detector image.
    detimage = np.zeros((sz, sz))
    for wl in waves:
        psf = calc_psf(wl, sz, INSTRUME, PUPIL, PSCALE)
        detimage += psf

    # Initialize MFT object.
    ft = matrixDFT.MatrixFourierTransform()

    # Calculate complex visibility.
    fimage = ft.perform(detimage, detimage.shape[0], detimage.shape[1])

    return np.abs(fimage)

def find_bad_pixels_fourier(data):

    pass

def find_bad_pixels_sigclip(data,
                            sigclip=5,
                            pix_shift=1,
                            rows=True,
                            cols=True):

    # Get median background and standard deviation.
    bg_med = np.nanmedian(data)
    bg_std = robust.medabsdev(data)
    bg_ind = data < (bg_med + 10. * bg_std) # clip bright PSFs for final calculation
    bg_med = np.nanmedian(data[bg_ind])
    bg_std = robust.medabsdev(data[bg_ind])

    # Create initial mask of large negative values.
    mask = data < bg_med - sigclip * bg_std

    # Pad data.
    pix_shift = int(pix_shift)
    pad_x = pad_y = pix_shift
    pad_vals = ([pad_y] * 2, [pad_x] * 2)
    pad_data = np.pad(data, pad_vals, mode='edge')

    # Shift data.
    sy, sx = data.shape
    shift_vals = np.arange(pix_shift * 2 + 1) - pix_shift
    shift_x = shift_vals if rows else [0]
    shift_y = shift_vals if cols else [0]
    data_arr = []
    for i in shift_x:
        for j in shift_y:
            if i != 0 or j != 0:
                data_arr.append(np.roll(pad_data, (j, i), axis=(0, 1)))
    data_arr = np.array(data_arr)
    data_arr = data_arr[:, pad_y:pad_y + sy, pad_x:pad_x + sx]

    # Find bad pixels.
    data_med = np.nanmedian(data_arr, axis=0)
    diff = data - data_med
    data_std = np.nanstd(data_arr, axis=0)
    # data_std = robust.medabsdev(data_arr, axis=0)
    mask = mask | (diff > sigclip * data_std)

    return mask

def fourier_corr(data,
                 mask,
                 fmask):

    # Get coordinates.
    ww = np.where(mask)
    fww = np.where(fmask)

    # Calculate B_Z matrix from Section 2.5 of Ireland 2013. This matrix maps
    # the bad pixels onto their Fourier power in the domain Z, which is the
    # complement of the pupil support.
    B_Z = np.zeros((len(ww[0]), len(fww[0]) * 2))
    xh = data.shape[1] // 2
    yh = data.shape[0] // 2
    xx, yy = np.meshgrid(2. * np.pi * np.arange(xh + 1) / data.shape[1],
                         2. * np.pi * (((np.arange(data.shape[0]) + yh) % data.shape[0]) - yh) / data.shape[0])
    for i in range(len(ww[0])):
        cdft = np.exp(-1j * (ww[0][i] * yy + ww[1][i] * xx))
        B_Z[i, :] = np.append(cdft[fww].real, cdft[fww].imag)

    # Calculate corrections for the bad pixels using the Moore-Penrose pseudo
    # inverse of B_Z (Equation 19 of Ireland 2013).
    B_Z_ct = np.transpose(np.conj(B_Z))
    B_Z_mppinv = np.dot(B_Z_ct, np.linalg.inv(np.dot(B_Z, B_Z_ct)))

    # Apply corrections for the bad pixels.
    data_out = deepcopy(data)
    data_out[ww] = 0.
    fdata = np.fft.rfft2(data_out)[fww]
    corr = -np.real(np.dot(np.append(fdata.real, fdata.imag), B_Z_mppinv))
    data_out[ww] += corr

    return data_out

def run(data,
        mask,
        INSTRUME,
        FILTER,
        PUPIL,
        PSCALE, # mas
        wave, # m
        weff, # m
        find_new=True):

    # Check input.
    sy, sx = data.shape
    if sx == sy:
        sz = sx
    else:
        raise UserWarning("Requires square array")

    # Calculate Fourier support.
    fsupp = calc_support(INSTRUME, PUPIL, PSCALE, wave, weff, sz)
    fsupp /= np.max(fsupp)
    fmask = fsupp < 1e-3 # seems to be a reasonable threshold
    fmask = np.fft.fftshift(fmask)[:, :sz // 2 + 1]

    # Loop through max 10 iterations.
    data_orig = deepcopy(data)
    for it in range(10):

        # Correct bad pixels.
        data = fourier_corr(data, mask, fmask)

        # Find remaining bad pixels.
        if find_new:
            mask_new = find_bad_pixels_sigclip(data)
            # if it == 0:
            #     ww = np.unravel_index(np.nanargmax(data), data.shape)
            #     mask_new[ww] = True
            #     mask_new[(ww[0] + 1, ww[1])] = True
            #     mask_new[(ww[0] - 1, ww[1])] = True
            #     mask_new[(ww[0], ww[1] + 1)] = True
            #     mask_new[(ww[0], ww[1] - 1)] = True
            Nmask_new = np.sum(mask_new & np.logical_not(mask))
        else:
            break
        print("Iteration %.0f: %.0f bad pixels identified, %.0f are new" % (it + 1, np.sum(mask_new), Nmask_new))
        if Nmask_new == 0:
            break
        mask = mask | mask_new

    # Correct all bad pixels at once.
    data = fourier_corr(data_orig, mask, fmask)

    if find_new:
        return data, mask
    else:
        return data
