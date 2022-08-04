from pkg_resources import get_distribution, DistributionNotFound

from jwst_kpi import pupil_data

try:
    __version__ = get_distribution(__name__).version  # get version from setup
except DistributionNotFound:
    pass  # package is not installed

PUPIL_DIR = pupil_data.__path__[0]
