from __future__ import annotations

# Both entry points are optional at the sub-package level: the SITK path
# requires SimpleITK, the DeepALI path requires ``hf-deepali`` (and PyTorch).
# The top-level ``TPTBox.registration`` package installs helpful stubs so
# callers get a clear "install X" message instead of a ``NameError``. Here we
# just swallow the ImportError so that whichever backend *is* installed
# remains usable.

try:
    from .point_registration import Point_Registration, ridged_points_from_poi, ridged_points_from_subreg_vert
except ImportError:
    pass

try:
    from .deepali_point_registration import (
        Deepali_Point_Registration,
        ridged_points_from_poi_deepali,
        ridged_points_from_subreg_vert_deepali,
    )
except ImportError:
    pass
