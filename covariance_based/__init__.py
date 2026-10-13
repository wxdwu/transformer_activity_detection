"""Traditional covariance-based device activity detection."""

import os
import sys

# Package imports must select the backend before detector imports NumPy.
if sys.platform == "darwin":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from .detector import CovarianceMLEResult, covariance_mle_activity

__all__ = ["CovarianceMLEResult", "covariance_mle_activity"]
