from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter


def fix_bp_medfilt(
    data: np.ndarray,
    erro: np.ndarray,
    mask: np.ndarray,
    medfilt_size: int = 5,
) -> Tuple[np.ndarray]:

    nf = data.shape[0]
    data_bpfixed = data.copy()
    erro_bpfixed = erro.copy()
    for i in range(nf):
        data_bpfixed[i][mask[i]] = median_filter(data_bpfixed[i], size=medfilt_size)[mask[i]]
        erro_bpfixed[i][mask[i]] = median_filter(erro_bpfixed[i], size=medfilt_size)[mask[i]]

    return data_bpfixed, erro_bpfixed
