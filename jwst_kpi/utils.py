from __future__ import division

import json
import os
import re
import socket

import matplotlib

matplotlib.rcParams.update({"font.size": 14})


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from typing import Optional, Union

import astropy.io.fits as pyfits
from astroquery.svo_fps import SvoFps

from jwst_kpi import filter_data

FILTER_DIR = filter_data.__path__[0]

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


def has_network_access(host="8.8.8.8", port=53, timeout=3):
    """
    Check for network access by attempting a quick socket connection.
    Kept here to avoid any imports before settings environment variables

    Coded with copilot.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, socket.timeout):
        return False


# Load the NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile
# Service.
# http://svo2.cab.inta-csic.es/theory/fps/
def get_wave_svo(instrument: str, save_local: bool = False):
    wave = {}
    weff = {}
    filter_list = SvoFps.get_filter_list(facility="JWST", instrument=instrument.upper())
    for i in range(len(filter_list)):
        name = filter_list["filterID"][i]
        name = name[name.rfind(".") + 1 :]
        wave[name] = filter_list["WavelengthMean"][i] / 1e4  # micron
        weff[name] = filter_list["WidthEff"][i] / 1e4  # micron
    if save_local:
        data = {"wave": wave, "weff": weff}
        filename = f"{instrument.lower()}_filters.json"
        filepath = os.path.join(FILTER_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    return wave, weff


def get_wave_local(instrument: str):
    filename = f"{instrument.lower()}_filters.json"
    filepath = os.path.join(FILTER_DIR, filename)

    with open(filepath, "r") as f:
        data = json.load(f)

    return data["wave"], data["weff"]
