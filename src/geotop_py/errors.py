"""The single abort point for every condition the C++ model answers with
``t_error`` (immediate, fatal stop). Carrying the same message text makes a
Python abort greppable against the C++ failing report."""

__all__ = ["GeotopAbort"]


class GeotopAbort(Exception):
    """GEOtop called ``t_error``: the run cannot continue."""
