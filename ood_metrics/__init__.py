import pkg_resources
__version__ = pkg_resources.get_distribution('ood_metrics').version

from .metrics import *
from .plots import *
