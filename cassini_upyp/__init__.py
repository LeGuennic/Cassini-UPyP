from __future__ import annotations

__all__ = ["UVIS_Observation", "UVIS_Bin"]

def __getattr__(name: str):
    if name in __all__:
        from .lib import uvis
        return getattr(uvis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
