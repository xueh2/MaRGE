"""DEPRECATED: VD support is now built into ``fromMATtoMRD3D_RARE``.

This module remains as a thin redirect so that any out-of-tree caller
keeps working. New code should import from
``marge.marge_tyger.fromMATtoMRD3D_RARE`` directly; that converter
now auto-detects variable-density / embedded undersampling from the
``.mat`` contents (``vd_ordering``, ``accelerationFactor``,
``undersamplingType``, ``undersamplingAxis``, ``densityMode``,
``calibrationSize``) and emits the ISMRMRD ``parallel_imaging``
header block plus the ``IS_PARALLEL_CALIBRATION_AND_IMAGING`` flag
on calibration lines.
"""
import warnings

from marge.marge_tyger.fromMATtoMRD3D_RARE import matToMRD as _matToMRD


def matToMRD(*args, **kwargs):
    warnings.warn(
        "marge.marge_tyger.fromMATtoMRD3D_RARE_vd.matToMRD is deprecated; "
        "use marge.marge_tyger.fromMATtoMRD3D_RARE.matToMRD instead "
        "(it auto-handles VD/embedded undersampling).",
        DeprecationWarning,
        stacklevel=2,
    )
    return _matToMRD(*args, **kwargs)
