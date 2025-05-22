from importlib.metadata import PackageNotFoundError, version

from .pipeline.calwebb_kpi3 import PUPIL_DIR, Kpi3Pipeline


try:
    __version__ = version("jwst-kpi")
except PackageNotFoundError:
    # package is not installed
    pass


__all__ = ["Kpi3Pipeline", "PUPIL_DIR"]
