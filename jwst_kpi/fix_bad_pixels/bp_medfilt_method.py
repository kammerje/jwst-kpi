from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter


def fix_bp_medfilt(
    data: np.ndarray,
    erro: np.ndarray,
    mask: np.ndarray,
    medfilt_size: int = 5,
) -> Tuple[np.ndarray]:
    """
    Fix bad pixels in data and error cubes with a median filter.

    Parameters
    ----------
    data : np.ndarray
        3D array containing data
    erro : np.ndarray
        3D array containing errors
    mask
        3D array containing bad pixel mask (True where bad)
    medfilt_size
        Size of median filter used to correct bad pixels

    Returns
    -------
    Tuple[np.ndarray]
        Fixed data and error cubes
    """

    nf = data.shape[0]
    data_bpfixed = data.copy()
    erro_bpfixed = erro.copy()
    for i in range(nf):
        data_bpfixed[i][mask[i]] = median_filter(data_bpfixed[i], size=medfilt_size)[mask[i]]
        erro_bpfixed[i][mask[i]] = median_filter(erro_bpfixed[i], size=medfilt_size)[mask[i]]

    return data_bpfixed, erro_bpfixed
