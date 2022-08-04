"""
Example script of how one can generate pupil models. This will be kept up to date to show
how the current default model has been implemented.
"""
import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate, shift
from xara import create_discrete_model, hexagon, kpi, symetrizes_model

from jwst_kpi import PUPIL_DIR


# TODO: Add step and binary parameters
def create_hex_model(aper, pxsc):
    d_hex = 1.32  # m; short diagonal of an individual mirror segment
    D_hex = (
        d_hex * 2.0 / np.sqrt(3.0)
    )  # m; long diagonal of an individual mirror segment
    # TODO: Is there a way to make this a loop or simplify it?
    xy = np.array(
        [
            [0.0, 0.0],
            [0.0, -d_hex],
            [0.0, d_hex],
            [0.0, -2.0 * d_hex],
            [0.0, 2.0 * d_hex],
            [0.0, -3.0 * d_hex],
            [0.0, 3.0 * d_hex],
            [-0.75 * D_hex, -0.5 * d_hex],
            [-0.75 * D_hex, 0.5 * d_hex],
            [-0.75 * D_hex, -1.5 * d_hex],
            [-0.75 * D_hex, 1.5 * d_hex],
            [-0.75 * D_hex, -2.5 * d_hex],
            [-0.75 * D_hex, 2.5 * d_hex],
            [0.75 * D_hex, -0.5 * d_hex],
            [0.75 * D_hex, 0.5 * d_hex],
            [0.75 * D_hex, -1.5 * d_hex],
            [0.75 * D_hex, 1.5 * d_hex],
            [0.75 * D_hex, -2.5 * d_hex],
            [0.75 * D_hex, 2.5 * d_hex],
            [-1.5 * D_hex, 0.0],
            [-1.5 * D_hex, -d_hex],
            [-1.5 * D_hex, d_hex],
            [-1.5 * D_hex, -2.0 * d_hex],
            [-1.5 * D_hex, 2.0 * d_hex],
            [1.5 * D_hex, 0.0],
            [1.5 * D_hex, -d_hex],
            [1.5 * D_hex, d_hex],
            [1.5 * D_hex, -2.0 * d_hex],
            [1.5 * D_hex, 2.0 * d_hex],
            [-2.25 * D_hex, -0.5 * d_hex],
            [-2.25 * D_hex, 0.5 * d_hex],
            [-2.25 * D_hex, -1.5 * d_hex],
            [-2.25 * D_hex, 1.5 * d_hex],
            [2.25 * D_hex, -0.5 * d_hex],
            [2.25 * D_hex, 0.5 * d_hex],
            [2.25 * D_hex, -1.5 * d_hex],
            [2.25 * D_hex, 1.5 * d_hex],
        ]
    )
    xy_all = []
    for i in range(xy.shape[0]):
        for j in range(3):
            xx = d_hex / 3.0 * np.sin(j / 3.0 * 2.0 * np.pi)
            yy = d_hex / 3.0 * np.cos(j / 3.0 * 2.0 * np.pi)
            xy_all += [[xy[i, 0] + xx, xy[i, 1] + yy]]
    xy_all = np.array(xy_all)

    mask = np.zeros_like(aper)
    for i in range(xy_all.shape[0]):
        sapt = hexagon(aper.shape[0], 0.45 * D_hex / pxsc)
        sapt = shift(sapt, (xy_all[i, 1] / pxsc, xy_all[i, 0] / pxsc), order=0)
        mask = (mask > 0.5) | (sapt > 0.5)

    model = []
    for i in range(xy_all.shape[0]):
        sapt = hexagon(aper.shape[0], 0.50 * D_hex / pxsc)
        sapt = shift(sapt, (xy_all[i, 1] / pxsc, xy_all[i, 0] / pxsc), order=0)
        ww = sapt > 0.5
        if np.sum(ww) < 0.5:
            tt = 0.0
        else:
            tt = np.mean(aper[ww])
        if tt == 0.0:
            continue
        else:
            model += [[xy_all[i, 0], xy_all[i, 1], tt]]
    model = np.array(model)

    return model


# TODO: Add docstring
def generate_pupil_model(
    input_mask: Union[Path, str],
    step: float,
    tmin: float,
    binary: bool = False,
    symmetrize: bool = False,
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
    if np.abs(rot_ang) > 0.0:
        aper = rotate(aper, rot_ang, order=1)

    if hex_grid:
        model = create_hex_model(aper, pxsc)
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
    KPI.filter_baselines(KPI.RED > min_red)
    KPI.package_as_fits(fname=out_fits)

    if show or out_plot is not None:
        _ = KPI.plot_pupil_and_uv(cmap="inferno")
        if show:
            plt.show(block=True)
        if out_plot is not None:
            plt.savefig(out_plot)
        plt.close()

    return KPI


if __name__ == "__main__":
    output_dir = Path("extras/pupil_results/")

    base_dict = dict(
        step=0.3,
        tmin=0.1,
        binary=False,
        symmetrize=True,
        bmax=None,
        hex_border=True,
        hex_grid=True,  # For hex
        min_red=2,  # For hex
    )

    niriss_clearp_dict = {
        **dict(input_mask="CLEARP", out_fits=output_dir / "niriss_clear_pupil.fits"),
        **base_dict,
    }

    niriss_nrm_dict = {
        **dict(input_mask="NRM", out_fits=output_dir / "niriss_nrm_pupil.fits"),
        **base_dict,
    }

    nircam_clear_dict = {
        **dict(input_mask="CLEAR", out_fits=output_dir / "nircam_clear_pupil.fits"),
        **base_dict,
    }

    models = [niriss_clearp_dict, niriss_nrm_dict, nircam_clear_dict]

    for model in models:
        KPI = generate_pupil_model(**model)
