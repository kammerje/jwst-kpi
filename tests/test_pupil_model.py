from pathlib import Path

import numpy as np
import pytest
from xara.kpi import KPI

from jwst_kpi import pupil_data
from jwst_kpi.pupil_model import generate_pupil_model

PUPIL_DIR = Path(pupil_data.__path__[0])


@pytest.fixture
def base_args():
    base_dict = dict(
        step=0.3,
        tmin=0.7,
        binary=False,
        symmetrize=True,
        bmax=None,
        hex_border=True,
        show=False,
        hex_grid=True,  # For hex CLEARP and CLEAR
    )
    return base_dict

def test_nircam_clear(base_args):
    nircam_args = dict(
        input_mask="CLEAR",
        symmetrize=True,
        rot_ang=0.47568395,
    )

    args = base_args | nircam_args
    kpi_test = generate_pupil_model(**args)

    pupil_path_pkg = PUPIL_DIR / "nircam_clear_pupil.fits"
    kpi_pkg = KPI(pupil_path_pkg)

    assert kpi_test.UVC.shape == kpi_pkg.UVC.shape
    assert kpi_test.KPM.shape == kpi_pkg.KPM.shape
    np.testing.assert_allclose(kpi_test.UVC, kpi_pkg.UVC)
    np.testing.assert_allclose(kpi_test.BLM, kpi_pkg.BLM, atol=1e-15)
    # It looks like the initial pupil model was created before normalization!
    norm = np.linalg.norm(kpi_pkg.KPM, axis=1)
    kpm_pkg_norm = np.divide(kpi_pkg.KPM.T, norm).T
    np.testing.assert_allclose(kpi_test.KPM, kpm_pkg_norm)
    # The latest version has a pupil mask saved
    assert kpi_pkg.pupil_mask is not None
