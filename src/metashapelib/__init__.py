# Check if Metashape python module is installed
try:
    import Metashape
except ImportError:
    raise ImportError(
        "Metashape module not found. Please check that you have installed the Metashape python module."
    )


# Check if Metashape is activated
def check_license() -> bool:
    if Metashape.app.activated:
        return True
    else:
        return False


if not check_license():
    raise Exception(
        "No license found. Please check that you linked your license (floating or standalone) with the Metashape python module."
    )


def backward_compatibility() -> bool:
    if Metashape.app.version < "2.0":
        return True
    else:
        return False


# Import modukes
from . import montecarlo
from . import utils
from . import workflow

# Import submodules and functions
from .utils.log import setup_logger, getlogger
from .export import *
from .msutils import *

# Setup logger
log = setup_logger(name="metashapelib", log_level="DEBUG")
