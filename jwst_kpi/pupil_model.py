from typing import Optional, Union, Tuple
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate, shift
from xara import create_discrete_model, kpi, symetrizes_model
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import matplotlib.cm as cm
import warnings
from xaosim.pupil import hex_mirror_model, uniform_hex

from jwst_kpi import PUPIL_DIR



def create_hex_model(
    aper: np.ndarray,
    pscale: float,
    ns: int = 3,
    sdiam: float = 1.32,
    threshold: float = 0.7,
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
    threshold : float
        Transmission threshold below which subapertures will be rejected.

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
    keep = tcoords[2] > threshold
    tcoords = tcoords[:, keep]

    return tcoords.T, tmp


def generate_pupil_model(
    input_mask: Union[Path, str],
    step: float,
    tmin: float,
    nrings: int = 3,
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
    nrings : int
        Number of hexagonal rings per mirror segment. Defaults to 3.
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
    mask_found = input_mask in available_masks
    if mask_found:
        input_mask_path = pupil_dir / f"MASK_{input_mask}.fits"
    else:
        input_mask_path = Path(input_mask)
        mask_found = input_mask_path.is_file()
    if not mask_found:
        raise ValueError(f"Input FITS mask must be one of: {available_masks} or a valid full path.")

    with pyfits.open(input_mask_path) as hdul:
        aper = hdul[0].data
        pxsc = hdul[0].header["PUPLSCAL"] # m; pupil scale
    aper = aper[:-1, :-1]
    aper = shift(aper, (-0.5, -0.5))

    if pad > 0:
        # TODO: Check that default padding is sufficient even for ns=1
        aper = np.pad(aper, ((pad, pad), (pad, pad)))

    if hex_grid:
        model, tmp = create_hex_model(aper, pxsc, ns=nrings, threshold=tmin)
    else:
        model = create_discrete_model(aper, pxsc, step, binary=binary, tmin=tmin)

    if symmetrize:
        # TODO: Should this check against cut instead?
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
            # TODO: Add UV coverge for hex model as well
            mmax = (aper.shape[0] * pxsc) / 2
            plt.figure(figsize=(6.4, 4.8))
            plt.clf()
            plt.imshow(tmp, extent=(-mmax, mmax, -mmax, mmax), cmap=cm.gray)
            plt.scatter(model[:, 0], model[:, 1], c=model[:, 2], s=20)
            cb = plt.colorbar()
            cb.set_label("Transmission", rotation=270, labelpad=20)
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.tight_layout()
        if out_plot is not None:
            plt.savefig(out_plot)
        if show:
            plt.show(block=True)
        plt.close()

    return KPI
