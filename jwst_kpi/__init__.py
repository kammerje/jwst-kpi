from pkg_resources import get_distribution, DistributionNotFound

from .calwebb_kpi3 import Kpi3Pipeline, PUPIL_DIR

try:
    __version__ = get_distribution(__name__).version  # get version from setup
except DistributionNotFound:
    pass  # package is not installed


__all__ = ["Kpi3Pipeline", "PUPIL_DIR"]
