from __future__ import annotations

try:
    from ._ridged_points.point_registration import Point_Registration, ridged_points_from_poi, ridged_points_from_subreg_vert

except Exception:
    pass
try:
    from TPTBox.registration._deformable.deformable_reg import Deformable_Registration
    from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

    from ._deepali.spine_rigid_elements_reg import Rigid_Elements_Registration
except Exception:
    pass

try:
    from ._deepali.deepali_model import General_Registration

except Exception:
    pass
__all__ = [
    "Deformable_Registration",
    "General_Registration",
    "Point_Registration",
    "Rigid_Elements_Registration",
    "Template_Registration",
    "ridged_points_from_poi",
    "ridged_points_from_subreg_vert",
]
