from __future__ import division

import re

import matplotlib

matplotlib.rcParams.update({"font.size": 14})


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from typing import Optional, Union

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
    separator = None
    match = REMOVE_SUFFIX_REGEX_KPI.match(name)
    try:
        name = match.group('root')
        separator = match.group('separator')
    except AttributeError:
        pass
    if separator is None:
        separator = '_'
    return name, separator
