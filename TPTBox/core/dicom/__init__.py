"""DICOM import/export helpers.

Requires the optional ``dicom`` extra (``pydicom`` + ``dicom2nifti``). Importing
this package without them succeeds; calling ``extract_dicom_folder`` then raises
an ``ImportError`` naming the extra to install.
"""

from __future__ import annotations

from TPTBox.core.internal.optional_deps import missing_dependency_func

_DICOM_PACKAGES = "pydicom dicom2nifti"

try:
    from TPTBox.core.dicom.dicom_extract import extract_dicom_folder
except ImportError as _e_dicom:
    extract_dicom_folder = missing_dependency_func(  # type: ignore[assignment]
        "extract_dicom_folder", _e_dicom, "dicom", _DICOM_PACKAGES
    )

__all__ = ["extract_dicom_folder"]
