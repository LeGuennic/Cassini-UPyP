from __future__ import annotations

__all__ = ["UVIS_Observation", "UVIS_Bin"]

def __getattr__(name: str):
    if name in __all__:
        # Lazy import to keep top-level import light
        from . import uvis as _uvis
        return getattr(_uvis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
