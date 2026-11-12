from __future__ import annotations

try:
    from .spine_rigid_elements_reg import Rigid_Elements_Registration

except ImportError:
    pass

try:
    from .polyrigid_reg import Polyrigid_Registration

except ImportError:
    pass

try:
    from .deepali_model import General_Registration

except ImportError:
    pass
