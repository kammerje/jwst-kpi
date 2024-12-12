import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================
from pathlib import Path

from jwst_kpi.pupil_model import generate_pupil_model

# =============================================================================
# MAIN
# =============================================================================
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

    for model in models:
        KPI = generate_pupil_model(**model)
