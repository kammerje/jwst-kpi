import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from jwst.datamodels.dqflags import pixel as pxdq_flags
from poppy import matrixDFT
from scipy.ndimage import median_filter

from jwst_kpi import utils as ut
from jwst_kpi.constants import PUPIL_DIR, PUPLDIAM, READ_NOISE, gain, pscale

WL_OVERSIZE = 0.1


# Define logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def find_psf_centers(
    data: np.ndarray, filt_size: Optional[int] = 3, nref: Optional[int] = 5
) -> np.ndarray:
    nframes, sx, sy = data.shape[0]
    centers = []
    for i in range(nframes):
        centers += [
            np.unravel_index(
                np.argmax(median_filter(data[i], size=filt_size)), data[i].shape
            )
        ]
    centers = np.array(centers)
    x_max_hsize = min(
        sx - np.max(centers[:, 0]), np.min(centers[:, 0]) - nref
    )  # the bottom 4/5 rows are reference pixels
    y_max_hsize = min(sy - np.max(centers[:, 1]), np.min(centers[:, 1]) - 0)
    max_half_size = min(x_max_hsize, y_max_hsize)

    return centers, max_half_size


def crop_to_psf(data, centers, max_half_size):
    nframes = data.shape[0]
    cropped_data = np.zeros(
        (nframes, 2 * max_half_size, 2 * max_half_size), dtype=data.dtype
    )
    for i in range(nframes):
        cropped_data[i] = data[
            i,
            centers[i, 0] - max_half_size : centers[i, 0] + max_half_size,
            centers[i, 1] - max_half_size : centers[i, 1] + max_half_size,
        ]

    return cropped_data


def get_wavelength_arr(filt: str, instrument: str) -> np.ndarray:
    wave_inst, weff_inst = ut.get_wavelegth_and_weff(instrument)
    wavel = wave_inst[filt]
    hwhm = weff_inst[filt] / 2
    dwavel = hwhm * (1 + WL_OVERSIZE)
    wavel_arr = np.array([wavel, wavel - dwavel, wavel + dwavel])

    return wavel_arr * 1e-6


def find_new_badpix(
    frame: np.ndarray,
    mask: np.ndarray,
    support_comp: np.ndarray,
    instrument: str,
    med_threshold: float = 50.0,
    med_size: float = 3.0,
) -> np.ndarray:
    support_comp_data = np.real(np.fft.irfft2(np.fft.rfft2(frame) * support_comp))
    mfil_data = median_filter(frame, size=med_size)
    rn = READ_NOISE[instrument]  # e-
    # Calculate poisson + read noise
    noise = np.sqrt(mfil_data / gain[instrument] + rn**2)
    support_comp_data /= noise
    mfil_compdata = median_filter(support_comp_data, size=med_size)
    absdiff_compdata = np.abs(support_comp_data - mfil_compdata)
    newdq = absdiff_compdata > med_threshold * np.median(absdiff_compdata)
    # Bad pixels where mask is True are not new
    newdq = newdq & ~mask

    return newdq


def get_pupil_mask(pupil: str):
    try:
        pupil_path = Path(PUPIL_DIR) / f"MASK_{pupil}.fits"
        pupil_mask = fits.getdata(pupil_path)
    except FileNotFoundError:
        raise ValueError(f"Pupil mask for {pupil} not found.")
    return pupil_mask


def get_psf(wavel: float, pupil: str, fov_px: float, instrument: str):
    # Copied from nis019 (as most of this module)
    reselt = wavel / PUPLDIAM
    pscale_rad = np.deg2rad(pscale[instrument] / 1000 / 3600)
    nlamD = fov_px * pscale_rad / reselt  # Soummer nlamD FOV in reselts
    # instantiate an mft object:
    ft = matrixDFT.MatrixFourierTransform()

    pupil_mask = get_pupil_mask(pupil)
    image_field = ft.perform(pupil_mask, nlamD, fov_px)
    image_intensity = (image_field * image_field.conj()).real

    return image_intensity


def transform_image(image):
    ft = matrixDFT.MatrixFourierTransform()
    ftimage = ft.perform(
        image, image.shape[0], image.shape[0]
    )  # fake the no-loss fft w/ dft

    return np.abs(ftimage)


def get_fourier_support(filt, pupil, fov_px, instrument):
    # Get central wavelength and +/- HWHM
    # TODO: Compare speed and results with webbpsf (calc PSF with nlambda=3)
    # currently not a dependency, so only add if significant imporvement
    wavel_arr = get_wavelength_arr(filt, instrument)

    psf = np.zeros((fov_px, fov_px), dtype=float)
    for wavel in wavel_arr:
        psf += get_psf(wavel, pupil, fov_px, instrument)

    psf_transform = transform_image(psf)

    return psf_transform


def fourier_correction(data, mask, support_comp):
    mask_inds = np.where(mask)
    fourier_inds = np.where(support_comp)

    nbadpix = len(mask_inds[0])
    support_size = len(fourier_inds[0]) * 2
    # Transfer matrix
    bz_mat = np.zeros((nbadpix, support_size))
    # x and y half sizes
    xs, ys = data.shape[0], data.shape[1]
    xh = xs // 2
    yh = ys // 2
    # Modulo to have full length, but going down in both dir(e.g. 51, 0, 50 for len 100)
    # subtract to go vrom 0 to value, then from -value to 0 (ish)
    # Then phase -> 0 to almost pi, - pi to 0
    # From what I understand this is to replicate fft shift
    # TODO: xh and yh, and shape[0] or shape[1] were flipped in generation
    # Don't matter for square arrays
    yphase = 2 * np.pi * np.arange(yh + 1) / ys  # 0 to pi
    xphase = 2 * np.pi * (((np.arange(xs) + xh) % xs) - xh) / xs
    xx, yy = np.meshgrid(yphase, xphase)
    # TODO: vectorize?
    # For each bad pixel, get the phase shift and add to the transfer matrix
    for i in range(nbadpix):
        cdft = np.exp(-1j * (mask_inds[0][i] * yy + mask_inds[1][i] * xx))
        bz_mat[i, :] = np.append(cdft[fourier_inds].real, cdft[fourier_inds].imag)

    # Compute correction with Moore-Penrose pseudoinverse
    bz_mp_pinv = np.linalg.pinv(bz_mat)

    data_out = data.copy()
    # Set bad pixels to 0 before doing FT and correction to avoid abrupt jumps and Gibbs phenomenon
    data_out[mask_inds] = 0.0
    data_ft = np.fft.rfft2(data_out)[fourier_inds]  # Vector of elements outside support
    data_ft_vect = np.append(data_ft.real, data_ft.imag)  # Stack real and imaginary
    corr = np.real(
        data_ft_vect @ bz_mp_pinv
    )  # Apply correction, real just for safety but should not affect
    data_out[mask_inds] -= corr

    return data_out


def fix_bp_fourier(
    data: np.ndarray,
    erro: np.ndarray,
    mask: np.ndarray,
    instrument: str,
    pupil: str,
    filt: str,
    crop_frames: bool = False,
) -> np.ndarray:

    # TODO: Could be removed, should usually be done by trim step
    if crop_frames:
        centers, max_half_size = find_psf_centers(data)
        data = crop_to_psf(data, centers, max_half_size)
        erro = crop_to_psf(erro, centers, max_half_size)
        mask = crop_to_psf(mask, centers, max_half_size)

    # Get fourier support of the pupil model, and complement of support
    # TODO: Could get wavelengths here and make correction more indep of filters
    fov_px = data.shape[1]
    # Get fourier support of the pupil
    support = get_fourier_support(filt, pupil, fov_px, instrument)
    support /= np.max(support)
    # Get where support is ~ 0 (where should there NOT be signal)
    support_comp = support < 1e-3
    support_comp = np.fft.fftshift(support_comp)[:, : fov_px // 2 + 1]

    ramp = np.arange(2 * fov_px) - 2 * fov_px // 2
    xx, yy = np.meshgrid(ramp, ramp)
    # Distance around center of array
    dist = np.sqrt(xx**2 + yy**2)
    NRM_FACTOR = 9.0
    CLEAR_FACTOR = 12.0
    factor = NRM_FACTOR if pupil == "NRM" else CLEAR_FACTOR
    fact_rad2deg = 180.0 / np.pi
    fact_deg2pix = 1000.0 * 3600.0 / pscale[instrument]
    wave_inst, _ = ut.get_wavelengths(instrument)
    max_dist = (
        factor * wave_inst[filt] * 1e-6 / PUPLDIAM * fact_rad2deg * fact_deg2pix
    )
    pupil_noise_mask = dist > max_dist
    flagged_per_frame = np.sum(mask, axis=(1, 2))
    if np.sum(pupil_noise_mask) < np.mean(flagged_per_frame):
        raise RuntimeError("Subarray too small to estimate noise.")

    median_size = 3  # pix
    median_tres = 50.0  # JK: changed from 28 to 20 in order to capture all bad pixels
    niter = 10
    for i in range(data.shape[0]):
        data_frame = data[i]
        erro_frame = erro[i]
        mask_frame = mask[i]
        for _j in range(niter):
            data_frame = fourier_correction(data_frame, mask_frame, support_comp)
            erro_frame = fourier_correction(erro_frame, mask_frame, support_comp)
            # FT times the comp support mask, invert, real for safety
            newdq = find_new_badpix(
                data_frame,
                mask_frame,
                support_comp,
                instrument,
                med_threshold=median_tres,
                med_size=median_size,
            )
            n_newdq = np.sum(newdq)

            if n_newdq == 0:
                break
            mask_frame = mask_frame | newdq
        data[i] = data_frame
        erro[i] = erro_frame
        mask[i] = mask_frame

    return data, erro, mask


if __name__ == "__main__":
    hdul = fits.open(
        "examples/kerphase_testdata/NIRISS/TYC-8906-1660-1/jw01093009001_03103_00001_nis_calints.fits"
    )
    __import__("ipdb").set_trace()
    data = hdul["SCI"].data
    erro = hdul["ERR"].data
    pxdq = hdul["DQ"].data
    bad_bits = ["DO_NOT_USE"]
    mask = pxdq < 0  # Should be all false
    for i in range(len(bad_bits)):
        pxdq_flag = pxdq_flags[bad_bits[i]]
        mask = mask | (pxdq & pxdq_flag == pxdq_flag)
        if i == 0:
            bb = bad_bits[i]
        else:
            bb += ", " + bad_bits[i]

    plt.ioff()
    fix_bp_fourier(data, erro, mask, "NIRISS", "CLEARP", "F480M")
