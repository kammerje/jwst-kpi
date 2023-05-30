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

import matplotlib.cm as cm
import warnings

from pathlib import Path
from scipy.ndimage import rotate, shift
from typing import Optional, Tuple, Union

from xaosim.pupil import hex_mirror_model, uniform_hex
from xara import create_discrete_model, kpi, symetrizes_model

from jwst_kpi import PUPIL_DIR


# =============================================================================
# MAIN
# =============================================================================


def create_hex_model(
    aper: np.ndarray,
    pscale: float,
    ns: int = 3,
    # ns: int = 2,
    sdiam: float = 1.32,
) -> Tuple[np.ndarray]:
    """
    Create discrete pupil model with hexagonal grid.

    The original code used for this function is from Frantz Martinache and
    Jens Kammerer. It was adapted by Thomas Vandal to fit in the curernt
    framework.

    Parameters
    ----------
    aper : np.ndarray
        Pupil mask at high resolution (like those from WebbPSF).
    pscale : float
        Pupil mask pixel scale in meter.
    ns : int
        Number of hexagonal rings within a mirror segment (one hexagon in the
        middle, ns rings around).
    sdiam : float
        Center-to-center distance of mirror segments in meter.

    Returns
    -------
    Tuple[np.ndarray]
        model : np.ndarray
            Discrete pupil model.
        tmp : np.ndarray
            Array describing the overlap (sum) of the original pupil mask and
            the discrete pupil model.
    """

    # x, y coordinates for the discrete model
    coords = hex_mirror_model(2, ns, sdiam, fill=False)
    coords = np.unique(np.round(coords, 3), axis=1)
    nc = coords.shape[1] # number of potential subaperture coordinates

    # appending a column for the transmission
    tcoords = np.ones((3, nc))
    tcoords[:2, :] = coords

    psz = aper.shape[0]
    seg_mask = uniform_hex(psz, psz, sdiam / (2 * ns + 1) / pscale * np.sqrt(3) / 2)
    seg_norm = seg_mask.sum()
    tmp = aper.copy()

    for ii in range(nc):

        # get coords of one subaperture in the grid
        sx, sy = np.round(tcoords[:2, ii] / pscale).astype(int)

        # move mask to this location
        mask = np.roll(seg_mask, (sy, sx), axis=(0, 1))

        # sum with pupil where both == 2¸ only one == 1, and nothing == 0
        tmp += mask

        # get overlap of subaperture with pupil mask
        tcoords[2, ii] = (mask * aper).sum() / seg_norm

    # eliminate edge cases, based on their transmission
    threshold = 0.7
    keep = tcoords[2] > threshold
    tcoords = tcoords[:, keep]

    return tcoords.T, tmp


def generate_pupil_model(
    input_mask: Union[Path, str],
    step: float,
    tmin: float,
    binary: bool = False,
    symmetrize: bool = False,
    pad: int = 50,
    cut: float = 0.1,
    rot_ang: float = 0.0,
    bmax: float = None,
    min_red: float = 10.0,
    hex_border: bool = True,
    hex_grid: bool = False,
    show: bool = True,
    out_plot: Optional[Union[Path, str]] = None,
    out_txt: Optional[Union[Path, str]] = None,
    out_fits: Optional[Union[Path, str]] = None,
):
    """
    Generate pupil model for a set of parameters with XARA.

    This function is used to generate a discrete XARA pupil model from a
    higher-resolution FITS mask of the pupil (such as those in WebbPSF).

    Parameters
    ----------
    input_mask : Union[Path, str]
        Input FITS mask.
    step : float
        Step size of the discrete pupil model in meter (used only for square
        grid, ignored for hex grid).
    tmin : float
        Minimum transmission to keep in the model.
    binary : bool
        Whether the model should be binary (True) or grey (False).
    symmetrize : bool
        Symmetrize the model along the horizontal direction.
    pad : int
        Pad the input FITS mask.
    cut : float
        Cutoff distance when symmetrizing model (must be < step size).
    rot_ang : float
        Rotation angle for the model.
    bmax : float
        Maximum baseline kept in the model.
    min_red : float
        Minium redundancy kept in the model.
    hex_border : bool
        Hexagongal border filtering for baselines.
    hex_grid : bool
        Use hexagonal grid for subapertures.
    show : bool
        Show pupil model and uv-coverage.
    out_plot : Optional[Union[Path, str]]
        Output path for pupil model plot.
    out_txt : Optional[Union[Path, str]]
        Output path for pupil model text file.
    out_fits : Optional[Union[Path, str]]
        Output path for pupil model FITS file.

    Returns
    -------
    KPI : xara.kpi.KPI
        XARA KPI object used to define the discrete pupil model.
    """

    pupil_dir = Path(PUPIL_DIR)
    available_masks = [f.stem.split("_")[1] for f in pupil_dir.iterdir() if f.stem.startswith("MASK")]
    if input_mask not in available_masks:
        raise ValueError(f"Input FITS mask must be one of: {available_masks}")

    input_mask_path = pupil_dir / f"MASK_{input_mask}.fits"
    with pyfits.open(input_mask_path) as hdul:
        aper = hdul[0].data
        pxsc = hdul[0].header["PUPLSCAL"] # m; pupil scale
    aper = aper[:-1, :-1]
    aper = shift(aper, (-0.5, -0.5))

    if pad > 0:
        aper = np.pad(aper, ((pad, pad), (pad, pad)))

    if hex_grid:
        model, tmp = create_hex_model(aper, pxsc)
    else:
        model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)

    if symmetrize:
        if step <= 0.1:
            warnings.warn(f"Symmetrize cut parameter ({cut}) should be smaller than step ({step})")
        model = symetrizes_model(model, cut=cut)

    if np.abs(rot_ang) > 0.:
        th0 = rot_ang * np.pi / 180. # rad; rotation angle
        rot_mat = np.array([[np.cos(th0), -np.sin(th0)],
                            [np.sin(th0),  np.cos(th0)]]) # rotation matrix
        model[:, :2] = model[:, :2].dot(rot_mat) # rotated model = model * rotation matrix
        if hex_grid:
            tmp = rotate(tmp, rot_ang, reshape=False, order=1)

    if out_txt is not None:
        np.savetxt(out_txt, model, fmt="%+.10e %+.10e %.2f")
        kpi_args = dict(fname=out_txt)
    else:
        kpi_args = dict(array=model)

    kpi_args = {**kpi_args, **dict(bmax=bmax, hexa=hex_border)}
    KPI = kpi.KPI(**kpi_args)

    if min_red > 0:
        KPI.filter_baselines(KPI.RED > min_red)
    KPI.package_as_fits(fname=out_fits)

    if show or out_plot is not None:
        if not hex_grid:
            KPI.plot_pupil_and_uv(cmap="inferno", marker=".")
        else:
            mmax = (aper.shape[0] * pxsc) / 2
            plt.figure(figsize=(6.4, 4.8))
            plt.clf()
            plt.imshow(tmp, extent=(-mmax, mmax, -mmax, mmax), cmap=cm.gray)
            plt.scatter(model[:, 0], model[:, 1], c=model[:, 2], s=50)
            cb = plt.colorbar()
            cb.set_label("Transmission", rotation=270, labelpad=20)
            plt.xlabel("Size [m]")
            plt.ylabel("Size [m]")
            plt.tight_layout()
        if out_plot is not None:
            plt.savefig(out_plot)
        if show:
            plt.show(block=True)
        plt.close()

    return KPI


if __name__ == "__main__":

    output_dir = Path("pupilmodel/")
    if not output_dir.is_dir():
        output_dir.mkdir()

    base_dict = dict(
        step=0.3,
        tmin=0.1,
        binary=False,
        pad=50,
        cut=0.1,
        bmax=None,
        hex_border=True,
        show=False,
    )

    nircam_clear_dict = {
        **dict(
            input_mask="CLEAR",
            symmetrize=True,
            rot_ang=0.47568395,
            min_red=10,
            hex_grid=True,
            out_plot=output_dir / "nircam_clear_pupil.pdf",
            out_fits=output_dir / "nircam_clear_pupil.fits",
            ),
        **base_dict,
    }

    nircam_rnd_dict = {
        **dict(
            input_mask="RND",
            symmetrize=True,
            rot_ang=0.47568395,
            min_red=1,
            hex_grid=False,
            out_plot=output_dir / "nircam_rnd_pupil.pdf",
            out_fits=output_dir / "nircam_rnd_pupil.fits",
            ),
        **base_dict,
    }

    nircam_bar_dict = {
        **dict(
            input_mask="BAR",
            symmetrize=True,
            rot_ang=0.47568395,
            min_red=1,
            hex_grid=False,
            out_plot=output_dir / "nircam_bar_pupil.pdf",
            out_fits=output_dir / "nircam_bar_pupil.fits",
            ),
        **base_dict,
    }

    niriss_clear_dict = {
        **dict(
            input_mask="CLEARP",
            symmetrize=True,
            rot_ang=0.56126717,
            min_red=10,
            hex_grid=True,
            out_plot=output_dir / "niriss_clear_pupil.pdf",
            out_fits=output_dir / "niriss_clear_pupil.fits",
            ),
        **base_dict,
    }

    niriss_nrm_dict = {
        **dict(
            input_mask="NRM",
            symmetrize=False,
            rot_ang=0.56126717,
            min_red=1,
            # min_red=0,
            hex_grid=True,
            out_plot=output_dir / "niriss_nrm_pupil.pdf",
            out_fits=output_dir / "niriss_nrm_pupil.fits",
            ),
        **base_dict,
    }

    miri_clear_dict = {
        **dict(
            input_mask="CLEAR",
            symmetrize=True,
            rot_ang=4.83544897,
            min_red=10,
            hex_grid=True,
            out_plot=output_dir / "miri_clear_pupil.pdf",
            out_fits=output_dir / "miri_clear_pupil.fits",
            ),
        **base_dict,
    }

    models = [nircam_clear_dict, nircam_rnd_dict, nircam_bar_dict, niriss_clear_dict, niriss_nrm_dict, miri_clear_dict]
    # models = [niriss_nrm_dict]

    for model in models:
        KPI = generate_pupil_model(**model)
