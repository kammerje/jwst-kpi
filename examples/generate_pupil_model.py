"""
Example script of how one can generate pupil models. This will be kept up to date to show
how the current default model has been implemented.
"""
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate, shift
from xaosim.pupil import hex_mirror_model, uniform_hex
from xara import create_discrete_model, kpi, symetrizes_model

from jwst_kpi import PUPIL_DIR


def create_hex_model(
    aper: np.ndarray,
    pscale: float,
    ns: int = 3,
    sdiam: float = 1.32,
) -> Tuple[np.ndarray]:
    """
    Create discrete pupil model for Xara with hexagonal grid

    The original code used for this function is from Frantz Martinache and Jens Kammerer.
    It was adapted by Thomas Vandal to fit in the curernt framework.

    Parameters
    ----------
    aper : np.ndarray
        Pupil mask at high resolution (like those from WebbPSF)
    pscale : float
        Pupil pixel scale in meter
    ns : int
        Number of hexagonal rings within a mirror segment
        (one hexagon in the middle, ns rings around)
    sdiam : float
        Center-to-center distance of mirror segments.

    Returns
    -------
    Tuple[np.ndarray]
        - model: Discrete pupil model
        - tmp: Array describing the overlap (sum) of the original array and of the discrete model.
    """
    psz = aper.shape[0]

    # x,y coordinates for the discrete model
    coords = hex_mirror_model(2, ns, sdiam, fill=False)
    coords = np.unique(np.round(coords, 3), axis=1)
    nc = coords.shape[1]  # number of potential coordinate holes

    # appending a column for transmission
    tcoords = np.ones((3, nc))
    tcoords[:2, :] = coords

    seg_mask = uniform_hex(psz, psz, sdiam / (2 * ns + 1) / pscale * np.sqrt(3) / 2)
    snorm = seg_mask.sum()

    tmp = aper.copy()

    for ii in range(nc):
        # Get coords of one sub-aperture in grid
        sx, sy = np.round(tcoords[:2, ii] / pscale).astype(int)
        # Move mask to this location
        mask = np.roll(seg_mask, (sy, sx), axis=(0, 1))

        # Sum with pupil: where both == 2¸ where only one == 1, where nothing == 0
        tmp += mask

        # Get overlap of sub-aperture with with pupil and get fraction of segment
        tcoords[2, ii] = (mask * aper).sum() / snorm

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
    show: bool = False,
    out_plot: Optional[Union[Path, str]] = None,
    out_txt: Optional[Union[Path, str]] = None,
    out_fits: Optional[Union[Path, str]] = None,
):
    """
    Generate pupil model for a set of parameters with XARA

    This function is used to gereate XARA pupil model from higher-resolution FITS
    mask of the pupil (such as those in WebbPSF). It shows how the default models
    have been generated and enables users to easily define their own.

    Parameters
    ----------
    input_mask : Union[Path, str]
        Input FITS mask to use.
    step : float
        Step size of the mask (used only for square grid, ignored for hex)
    tmin : float
        Minimum transmission to keep in the mask
    binary : bool
        Whether the model should be binary (True) or grey (False)
    symmetrize : bool
        Symmetrize the pupil model along horizontal direction
    cut : float
        Cutoff distance when symmetrizing model (must be < step size)
    rot_ang : float
        Rotation angle for pupil (applied to aperture from input FITS mask)
    bmax : float
        Max baseline kept in pupil model (to avoid baselines outside of actual pupil)
    min_red : float
        Minium redundancy kept in pupil model (to avoid edge baselines that are not really in the pupil)
    hex_border : bool
        Hexagongal border filtering for baselines
    hex_grid : bool
        Use hexagonal grid for sub-apertures
    show : bool
        Show pupil model and U-V coverage
    out_plot : Optional[Union[Path, str]]
        Output path for pupil model plot
    out_txt : Optional[Union[Path, str]]
        Output path for pupil model text file (this does not include all the information in the model as some is saved in KPI object)
    out_fits : Optional[Union[Path, str]]
        Output fits file for the KPI object defined for the pupil model

    Returns
    -------
    KPI : xara.kpi.KPI
        Xara KPI object used to define the pupil model.
    """

    pupil_dir = Path(PUPIL_DIR)
    available_masks = [
        f.stem.split("_")[1] for f in pupil_dir.iterdir() if f.stem.startswith("MASK")
    ]
    if input_mask not in available_masks:
        raise ValueError(f"input_mask must be one of: {available_masks}")

    input_mask_path = pupil_dir / f"MASK_{input_mask}.fits"
    with fits.open(input_mask_path) as hdul:
        aper = hdul[0].data
        pxsc = hdul[0].header["PUPLSCAL"]  # m, pupil scale
    aper = aper[:-1, :-1]
    aper = shift(aper, (-0.5, -0.5))
    if pad > 0:
        aper = np.pad(aper, ((pad, pad), (pad, pad)))
    if np.abs(rot_ang) > 0.0:
        aper = rotate(aper, rot_ang, order=1)

    if hex_grid:
        model, tmp = create_hex_model(aper, pxsc)
    else:
        model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)
    if symmetrize:
        if step <= 0.1:
            warnings.warn(
                f"Symmetrize cut parameter ({cut}) should be smaller than step ({step})"
            )
        model = symetrizes_model(model, cut=cut)
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
            _ = KPI.plot_pupil_and_uv(cmap="inferno", marker=".")
        else:
            mmax = (aper.shape[0] * pxsc) / 2
            plt.figure(1, figsize=(6.4, 4.8))
            plt.clf()
            plt.imshow(tmp, extent=(-mmax, mmax, -mmax, mmax), cmap=cm.gray)
            plt.scatter(
                model[:, 0], model[:, 1], c=model[:, 2], s=50
            )  # , cmap=cm.rainbow)
            cb = plt.colorbar()
            cb.set_label("Transmission", rotation=270, labelpad=20)
            plt.xlabel("Size [m]")
            plt.ylabel("Size [m]")
            plt.tight_layout()
        if show:
            plt.show(block=True)
        if out_plot is not None:
            plt.savefig(out_plot)
        plt.close()

    return KPI


if __name__ == "__main__":

    # Where should we save pupil model
    output_dir = Path("pupil_results/")
    if not output_dir.is_dir():
        output_dir.mkdir()

    base_dict = dict(
        step=0.3,
        tmin=0.1,
        binary=False,
        symmetrize=True,
        bmax=None,
        hex_border=True,
        show=True,
        hex_grid=True,  # For hex CLEARP and CLEAR
        # min_red=2,  # For hex CLEARP
    )

    niriss_clearp_dict = {
        **dict(input_mask="CLEARP", out_fits=output_dir / "niriss_clear_pupil.fits"),
        **base_dict,
    }

    nircam_clear_dict = {
        **dict(input_mask="CLEAR", out_fits=output_dir / "nircam_clear_pupil.fits"),
        **base_dict,
    }

    models = [niriss_clearp_dict, nircam_clear_dict]

    for model in models:
        KPI = generate_pupil_model(**model)
