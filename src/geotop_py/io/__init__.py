"""Readers for the inputs and outputs of a 1-D ``PointSim`` GEOtop simulation.

Pure-Python parsers of GEOtop's own plain-text formats; each returns values,
never model state.

- :mod:`geotop_py.io.parfile`    -- ``geotop.inpts`` against the v3.0 keyword tables
- :mod:`geotop_py.io.table`      -- GEOtop's CSV tables (header + rows)
- :mod:`geotop_py.io.meteo`      -- native meteo station files
- :mod:`geotop_py.io.soil`       -- soil parameter files
- :mod:`geotop_py.io.points`     -- point geometry, inline or from raster maps
- :mod:`geotop_py.io.rastermap`  -- ESRI ASCII raster maps
- :mod:`geotop_py.io.horizon`    -- horizon files
- :mod:`geotop_py.io.vegfile`    -- land-cover (vegetation) parameters
- :mod:`geotop_py.io.gt_output`  -- GEOtop ``point*.txt`` tables and profiles
- :mod:`geotop_py.io.keywords`   -- the v3.0 keyword tables (generated)
- :mod:`geotop_py.io.geomorphology` -- DEM smoothing and curvature, for point geometry
"""
