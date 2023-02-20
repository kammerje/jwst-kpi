from __future__ import division

import re

import matplotlib
from astroquery.svo_fps import SvoFps

matplotlib.rcParams.update({"font.size": 14})


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from typing import Optional, Tuple, Union

import astropy.io.fits as pyfits

KPI_SUFFIXES = [
    "trimframesstep",
    "fixbadpixelsstep",
    "recenterframesstep",
    "windowframesstep",
    "extractkerphasestep",
    "empiricaluncertaintiesstep",
]

REMOVE_SUFFIX_REGEX_KPI = re.compile(
    "^(?P<root>.+?)((?P<separator>_|-)(" + "|".join(KPI_SUFFIXES) + "))?$"
)

# =============================================================================
# MAIN
# =============================================================================


def split_file_path(
    file: Union[Path, str],
):

    file_path = Path(file)
    suffixes = file_path.suffixes[-2:]
    n_suffixes = len(suffixes)
    fext = "".join(suffixes)
    parent_dir = str(file_path.parent)
    file_stem = file_path
    for _ in range(n_suffixes):
        file_stem = Path(file_stem.stem)
    file_stem = str(file_stem)

    return parent_dir, file_stem, fext


def open_fits(
    file: Union[Path, str],
    suffix: Optional[str] = None,
    file_dir: Optional[Union[str, Path]] = None,
):

    file_path = Path(file)
    suffix = suffix or ""
    parent_dir, file_stem, fext = split_file_path(file_path)
    basename = file_stem + suffix + fext  # handle compressed files, e.g., fits.gz
    if file_dir is None:
        file_path = Path(parent_dir) / basename
    else:
        file_path = Path(file_dir) / basename

    return pyfits.open(file_path)


def get_output_base(
    file: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
):

    file_path = Path(file)
    parent_dir, file_stem, _ = split_file_path(file_path)
    if output_dir is None:
        output_base = Path(parent_dir) / file_stem
    else:
        output_base = Path(output_dir) / file_stem

    return str(output_base)


def remove_suffix_kpi(name):
    """
    Remove suffix from any KPI step output

    Remove suffix from JWST KPI Pipeline file. This function was made to
    catch suffixes from KPI pipeline steps, which are not recognized by main
    JWST pipeline.

    Parameters
    ----------
    name : str
        File name from which suffix should be removed
    """
    separator = None
    match = REMOVE_SUFFIX_REGEX_KPI.match(name)
    try:
        name = match.group("root")
        separator = match.group("separator")
    except AttributeError:
        pass
    if separator is None:
        separator = "_"
    return name, separator


def get_wavelengths(instrument: str) -> Tuple[dict, dict]:
    """
    Get wavelengths for a given instruments

    Gives the mean wavelength and effective width for each filter in a dictionary

    Parameters
    ----------
    instrument
        Instrument name

    Returns
    -------
    Tuple[dict, dict]
        Dictionary of filter names and wavelengths, and dictionary of filter
        names and effective widths
    """
    filt_tbl = SvoFps.get_filter_list(facility="JWST", instrument=instrument)
    wave_dict = dict()
    weff_dict = dict()
    for filt_row in filt_tbl:
        name = filt_row["filterID"].split(".")[-1]
        wave_dict[name] = filt_row["WavelengthMean"] / 1e4  # Angstrom -> micron
        weff_dict[name] = filt_row["WidthEff"] / 1e4  # Angstrom -> micron
    return wave_dict, weff_dict
