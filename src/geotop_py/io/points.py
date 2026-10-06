"""The list of points a 1D run is made of, and their topographic properties.

Every point is a row of nineteen columns -- identity, coordinates, elevation,
land cover and soil type, slope, aspect, sky view factor, the four directional
curvatures, and a few per-point limits -- addressed by name through :data:`COL`.
The matrix is assembled in three passes:

1. :func:`points_from_keywords` -- the inline ``geotop.inpts`` keywords, one
   component per point, everything else left at ``NUMBER_NOVALUE``;
2. :func:`read_point_file` -- ``listpoints.txt`` over the top, cell by cell,
   falling back to pass 1's value for that row (or to its last row, when the
   file lists more points than the keywords did);
3. :func:`apply_defaults` -- what is *still* undefined gets a fixed default:
   flat, unshaded, unit land cover, an unlimited snow water equivalent, the
   simulation's own latitude and longitude.

Between 2 and 3 the gaps are filled from raster maps where there are any --
:func:`fill_from_maps`: elevation from the DEM, land cover, soil type, slope,
aspect, sky view factor and bedrock depth each from their own map, and the
four curvatures computed over the smoothed DEM. What no map covers is left for
pass 3. Slope, aspect and sky view factor can also be *derived* from the DEM
when their map is missing; that is not implemented, and is refused rather than
silently given the flat default. :func:`map_derived_columns` reports which
columns a run would take from a map, without reading anything.

Column units: coordinates [m], elevation [m a.s.l.], slope and aspect [deg],
sky view factor [-], curvatures [1/m], depth of the free surface at the
boundary and maximum snow water equivalent [mm], latitude and longitude [deg],
bedrock depth [mm].
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..constants import NUMBER_NOVALUE, STRING_NOVALUE, is_novalue, is_undefined
from . import geomorphology, rastermap
from .table import read_txt_matrix

#: Point columns, in storage order. The index of a name here, plus one, is the
#: column index the point matrix is addressed by.
COLUMNS = ("ptID", "ptX", "ptY", "ptZ", "ptLC", "ptSY", "ptS", "ptA", "ptSKY",
           "ptCNS", "ptCWE", "ptCNwSe", "ptCNeSw", "ptDrDEPTH", "ptHOR",
           "ptMAXSWE", "ptLAT", "ptLON", "ptBED")
COL = {name: i + 1 for i, name in enumerate(COLUMNS)}
PTTOT = len(COLUMNS)

#: The ``Header*`` keyword naming each column in the point file, in column order.
HEADER_KEYWORDS = (
    "HeaderPointID", "HeaderCoordinatePointX", "HeaderCoordinatePointY",
    "HeaderPointElevation", "HeaderPointLandCoverType", "HeaderPointSoilType",
    "HeaderPointSlope", "HeaderPointAspect", "HeaderPointSkyViewFactor",
    "HeaderPointCurvatureNorthSouthDirection",
    "HeaderPointCurvatureWestEastDirection",
    "HeaderPointCurvatureNorthwestSoutheastDirection",
    "HeaderPointCurvatureNortheastSouthwestDirection",
    "HeaderPointDepthFreeSurface", "HeaderPointHorizon", "HeaderPointMaxSWE",
    "HeaderPointLatitude", "HeaderPointLongitude", "HeaderPointBedrockDepth",
)

#: The numeric keyword giving each column inline in ``geotop.inpts``, in column
#: order. The last one really is called ``NONE``: bedrock depth has a ``Header*``
#: keyword and a file column, but no inline keyword, and the table keeps its
#: slot so the positional indexing stays aligned.
POINT_KEYWORDS = (
    "PointID", "CoordinatePointX", "CoordinatePointY", "PointElevation",
    "PointLandCoverType", "PointSoilType", "PointSlope", "PointAspect",
    "PointSkyViewFactor", "PointCurvatureNorthSouthDirection",
    "PointCurvatureWestEastDirection",
    "PointCurvatureNorthwestSoutheastDirection",
    "PointCurvatureNortheastSouthwestDirection", "PointDepthFreeSurface",
    "PointHorizon", "PointMaxSWE", "PointLatitude", "PointLongitude", "NONE",
)

#: Only the first sixteen keywords are counted when sizing the point matrix:
#: declaring latitude, longitude or bedrock depth for a point does not, on its
#: own, create that point.
NPOINTS_KEYWORDS = 16

#: Maximum snow water equivalent [mm] when the point does not set one -- large
#: enough to never bind.
DEFAULT_MAXSWE = 1.0e10
#: Bedrock depth [mm] when the point does not set one: below any column.
DEFAULT_BEDROCK_DEPTH = 1.0e99

#: The map-file keywords that can fill a point's missing topography, by the
#: short name :func:`map_derived_columns` refers to them by.
MAP_KEYWORDS = {
    "dem": "DemFile",
    "lu": "LandCoverMapFile",
    "soil": "SoilMapFile",
    "sky": "SkyViewFactorMapFile",
    "slp": "SlopeMapFile",
    "asp": "AspectMapFile",
    "bed": "BedrockDepthMapFile",
}

#: Extensions GEOtop recognises a raster map by, in the order it tries them.
MAP_EXTENSIONS = (".grass", ".asc")


class PointError(ValueError):
    """Raised where GEOtop would abort while reading the point list."""


@dataclass
class PointOptions:
    """The run-wide parameters the per-point defaults fall back to.

    Defaults are GEOtop's own for each keyword.
    """
    DepthFreeSurface: float = 0.0    # [mm]
    latitude: float = 45.0           # [deg]
    longitude: float = 0.0           # [deg]
    soil_type_land_default: int = 1

    @classmethod
    def from_parfile(cls, pf) -> "PointOptions":
        return cls(
            DepthFreeSurface=pf.number("DepthFreeSurfaceAtTheBoundary", 0, 0.0),
            latitude=pf.number("Latitude", 0, 45.0),
            longitude=pf.number("Longitude", 0, 0.0),
            soil_type_land_default=int(pf.number("DefaultSoilTypeLand", 0, 1.0)),
        )


@dataclass
class PointProperties:
    """One point's topography and per-point limits, all defaults resolved.

    These are exactly the inputs the meteorological distribution needs: the
    elevation it applies lapse rates over, and the slope, aspect and four
    curvatures the wind correction reads.
    """
    ID: int
    East: float             # [m]
    North: float            # [m]
    Z: float                # [m a.s.l.]
    LC: int                 # land cover type
    soil_type: int
    slope: float            # [deg]
    aspect: float           # [deg from N, clockwise]
    sky: float              # sky view factor [-]
    curvature1: float       # N-S  [1/m]
    curvature2: float       # W-E  [1/m]
    curvature3: float       # NW-SE [1/m]
    curvature4: float       # NE-SW [1/m]
    BC_DepthFreeSurface: float   # [mm]
    horizon_point: int
    maxSWE: float           # [mm]
    latitude: float         # [deg]
    longitude: float        # [deg]
    bed: float              # bedrock depth [mm]


# GEOtop: src/geotop/parameters.cc:1450-1474
def points_from_keywords(pf) -> List[List[float]]:
    """The point matrix as ``geotop.inpts`` alone defines it.

    Returns ``chkpt[n][j]``, both axes 1-based, with as many rows as the most
    componentful of the point keywords declares and ``NUMBER_NOVALUE`` wherever
    a keyword says nothing.
    """
    npoints = 0
    for name in POINT_KEYWORDS[:NPOINTS_KEYWORDS]:
        npoints = max(npoints, pf.components(name))

    chkpt = [[0.0] * (PTTOT + 1) for _ in range(npoints + 1)]
    for i in range(1, npoints + 1):
        for j in range(1, PTTOT + 1):
            chkpt[i][j] = pf.number(POINT_KEYWORDS[j - 1], i - 1, NUMBER_NOVALUE)
    return chkpt


# GEOtop: src/geotop/parameters.cc:2651-2723
def read_point_file(name: Optional[str], col_names: Sequence[str],
                    chkpt: List[List[float]]) -> List[List[float]]:
    """Read the point file over ``chkpt`` and return the resulting matrix.

    ``name`` is the file stem as ``geotop.inpts`` gives it; the file read is
    ``<name>.txt``. A missing file is not an error -- GEOtop simply keeps the
    keyword matrix, which is how the cases that declare their single point
    inline work.

    The file decides how many points there are. Each of its cells that is absent
    or novalue falls back to the same column of the keyword matrix's row of the
    same index, clamped to its last row.
    """
    if len(col_names) != PTTOT:
        raise ValueError(f"expected {PTTOT} column names, got {len(col_names)}")
    if name is None or name == STRING_NOVALUE:
        return [list(r) for r in chkpt]

    path = f"{name}.txt"
    if not os.path.exists(path):
        return [list(r) for r in chkpt]

    points = read_txt_matrix(path, col_names)
    if not points:
        raise PointError(f"{path}: no data lines")

    old = chkpt
    old_npoints = len(old) - 1
    out = [[0.0] * (PTTOT + 1) for _ in range(len(points) + 1)]
    for n in range(1, len(points) + 1):
        for j in range(1, PTTOT + 1):
            out[n][j] = points[n - 1][j - 1]
            if is_undefined(out[n][j]):
                out[n][j] = old[n][j] if n <= old_npoints else old[old_npoints][j]
    return out


# GEOtop: src/geotop/input.cc:2952-3406 (the read_dem/read_lu/read_sl/... decisions)
def map_derived_columns(chkpt: List[List[float]],
                        available: FrozenSet[str]) -> List[Tuple[int, str]]:
    """Which still-undefined columns GEOtop would fill from a raster map.

    ``available`` is the set of map keys (:data:`MAP_KEYWORDS`) whose file is on
    disk. Returns ``(column index, source)`` pairs; an empty list means the run
    is fully specified by keywords and the point file, and
    :func:`apply_defaults` alone finishes the job.

    Reading the maps themselves is not implemented: it needs the raster reader,
    the topographic low-pass filter and the slope/aspect/curvature/sky-view
    operators, none of which a point run otherwise touches.
    """
    npoints = len(chkpt) - 1

    def any_missing(*names: str) -> bool:
        return any(is_undefined(chkpt[n][COL[name]])
                   for n in range(1, npoints + 1) for name in names)

    coordinates = not any_missing("ptX", "ptY")
    if not coordinates:
        return []

    curvatures = ("ptCNS", "ptCWE", "ptCNwSe", "ptCNeSw")
    read_dem = any_missing("ptLC", "ptSY", "ptS", "ptA", *curvatures) \
        and "dem" in available

    out: List[Tuple[int, str]] = []
    if read_dem and any_missing("ptZ"):
        out.append((COL["ptZ"], "dem"))
    # Land cover falls back to a uniform 1.0 over the DEM's extent when there is
    # no land cover map, which is the same value apply_defaults would give it --
    # so only an actual map counts as map-derived here.
    if any_missing("ptLC") and "lu" in available:
        out.append((COL["ptLC"], "lu"))
    if any_missing("ptSY") and "soil" in available:
        out.append((COL["ptSY"], "soil"))
    for name, key in (("ptS", "slp"), ("ptA", "asp"), ("ptSKY", "sky")):
        if any_missing(name) and (key in available or read_dem):
            out.append((COL[name], key if key in available else "dem"))
    if any_missing("ptBED") and "bed" in available:
        out.append((COL["ptBED"], "bed"))
    if read_dem:
        for name in curvatures:
            if any_missing(name):
                out.append((COL[name], "dem"))
    return out


def _map_path(pf, key: str, base_dir: str) -> Optional[str]:
    """The file stem of map ``key``, or ``None`` if the run does not name one."""
    stem = pf.string(MAP_KEYWORDS[key])
    if stem == STRING_NOVALUE:
        return None
    return os.path.join(base_dir, stem)


def _lookup(M: "rastermap.RasterMap", chkpt: List[List[float]],
           n: int) -> Tuple[int, int]:
    """The grid cell point ``n`` falls in, as 1-based ``(row, column)``."""
    r = geomorphology.row(chkpt[n][COL["ptY"]], M.nrows, M.dy, M.Y0,
                          NUMBER_NOVALUE)
    c = geomorphology.col(chkpt[n][COL["ptX"]], M.ncols, M.dx, M.X0,
                          NUMBER_NOVALUE)
    if is_undefined(r) or is_undefined(c):
        raise PointError(
            f"point {n} at ({chkpt[n][COL['ptX']]}, {chkpt[n][COL['ptY']]}) "
            "falls outside the map it would be read from")
    return r, c


# GEOtop: src/geotop/input.cc:2952-3406
def fill_from_maps(chkpt: List[List[float]], pf, base_dir: str) -> List[List[float]]:
    """Fill the still-undefined topography columns from the run's raster maps.

    Each column is read from its own map where the run names one and the file
    is there; the four curvatures are computed over the smoothed elevation
    model instead, since no map holds them. A point outside the grid, or a
    column no map covers, is left undefined for :func:`apply_defaults`.

    Nothing is read when no point has coordinates: there would be no way to
    say which cell it is in.

    Raises :class:`NotImplementedError` for the one path not ported: deriving
    slope, aspect or sky view factor from the elevation model when their own
    map is absent.
    """
    out = [list(r) for r in chkpt]
    npoints = len(out) - 1
    novalue = NUMBER_NOVALUE

    def missing(name: str) -> bool:
        return any(is_undefined(out[n][COL[name]]) for n in range(1, npoints + 1))

    def fill(name: str, M: "rastermap.RasterMap") -> None:
        for n in range(1, npoints + 1):
            if is_undefined(out[n][COL[name]]):
                r, c = _lookup(M, out, n)
                out[n][COL[name]] = M.data[r][c]

    coordinates = not any(is_undefined(out[n][COL["ptX"]])
                          or is_undefined(out[n][COL["ptY"]])
                          for n in range(1, npoints + 1))

    # (a) elevation model. It is smoothed once here, and every elevation read
    # out of it afterwards -- the point's own, and the curvatures' -- is of the
    # smoothed grid, not the file's.
    curvatures = ("ptCNS", "ptCWE", "ptCNwSe", "ptCNeSw")
    read_dem = any(missing(name) for name in
                   ("ptLC", "ptSY", "ptS", "ptA") + curvatures)
    if read_dem and not coordinates:
        read_dem = False
    Z = None
    if read_dem:
        path = _map_path(pf, "dem", base_dir)
        if path is not None and rastermap.existing_file(path) > 0:
            Z = rastermap.read_map(path, novalue)
            Z.data = geomorphology.multipass_topofilter(
                int(pf.number("NumLowPassFilterOnDemForAll", 0, 0.0)),
                Z.data, novalue, 1)
        else:
            read_dem = False
    if read_dem and missing("ptZ"):
        fill("ptZ", Z)

    # (b) land cover. With no map but a DEM, the cover is uniformly 1 over the
    # DEM's own extent -- the same value apply_defaults would give it.
    read_lu = missing("ptLC") and coordinates
    if read_lu:
        path = _map_path(pf, "lu", base_dir)
        if path is not None and rastermap.existing_file(path) > 0:
            fill("ptLC", rastermap.read_map(path, novalue, Z))
        elif read_dem:
            fill("ptLC", rastermap.uniform_like(1.0, Z))

    # (c) soil type, (f2) bedrock depth: from their map or not at all.
    for name, key in (("ptSY", "soil"), ("ptBED", "bed")):
        if missing(name) and coordinates:
            path = _map_path(pf, key, base_dir)
            if path is not None and rastermap.existing_file(path) > 0:
                fill(name, rastermap.read_map(path, novalue, Z))

    # (d) slope, (e) aspect, (f) sky view factor: from their map, or derived
    # from the elevation model, which is the part not ported.
    for name, key, derivation in (("ptS", "slp", "slope"),
                                  ("ptA", "asp", "aspect"),
                                  ("ptSKY", "sky", "sky view factor")):
        if not (missing(name) and coordinates):
            continue
        path = _map_path(pf, key, base_dir)
        if path is not None and rastermap.existing_file(path) > 0:
            fill(name, rastermap.read_map(path, novalue, Z))
        elif read_dem:
            raise NotImplementedError(
                f"{MAP_KEYWORDS[key]} is absent, so the {derivation} would be "
                "derived from the elevation model, which is not implemented")

    # (g) curvature. Smoothed a second time, by its own pass count, on top of
    # the smoothing (a) already applied.
    read_curv = any(missing(name) for name in curvatures)
    if read_curv and coordinates and read_dem:
        Q = geomorphology.multipass_topofilter(
            int(pf.number("NumLowPassFilterOnDemForCurv", 0, 0.0)),
            Z.data, novalue, 1)
        # The two cell sizes reach curvature() the other way round: the
        # north-south spacing arrives as deltax. They are equal on every ESRI
        # grid, which has one cellsize, so this only matters as provenance.
        grids = geomorphology.curvature(Z.dy, Z.dx, Q, novalue)
        for name, grid in zip(curvatures, grids):
            for n in range(1, npoints + 1):
                if is_undefined(out[n][COL[name]]):
                    r, c = _lookup(Z, out, n)
                    out[n][COL[name]] = grid[r][c]

    return out


# GEOtop: src/geotop/input.cc:3408-3459
def apply_defaults(chkpt: List[List[float]],
                   options: PointOptions) -> List[List[float]]:
    """Give every still-undefined column its fixed default, and return the result.

    Coordinates are deliberately not defaulted: a point without them stays
    without them, since there is nothing sensible to put there.
    """
    npoints = len(chkpt) - 1
    out = [list(r) for r in chkpt]
    for i in range(1, npoints + 1):
        row = out[i]
        for name, value in (("ptZ", 0.0), ("ptLC", 1.0),
                            ("ptSY", float(options.soil_type_land_default)),
                            ("ptS", 0.0), ("ptA", 0.0), ("ptSKY", 1.0),
                            ("ptCNS", 0.0), ("ptCWE", 0.0),
                            ("ptCNwSe", 0.0), ("ptCNeSw", 0.0),
                            ("ptDrDEPTH", options.DepthFreeSurface),
                            ("ptMAXSWE", DEFAULT_MAXSWE),
                            ("ptLAT", options.latitude),
                            ("ptLON", options.longitude)):
            if is_novalue(row[COL[name]]):
                row[COL[name]] = value
        if is_novalue(row[COL["ptID"]]):
            row[COL["ptID"]] = float(i)
        if is_novalue(row[COL["ptHOR"]]):
            row[COL["ptHOR"]] = row[COL["ptID"]]
    return out


# GEOtop: src/geotop/input.cc:3512-3597
def properties(chkpt: List[List[float]], n: int) -> PointProperties:
    """Point ``n`` of a defaults-resolved matrix, as named topography.

    ``n`` is 1-based, as the matrix is. Land cover and soil type must be
    positive; a bedrock depth left undefined means "no bedrock", represented by
    a depth below any column.
    """
    row = chkpt[n]
    LC = int(row[COL["ptLC"]])
    if LC <= 0:
        raise PointError(f"point {n} has land cover type {LC} <= 0")
    soil_type = int(row[COL["ptSY"]])
    if soil_type <= 0:
        raise PointError(f"point {n} has soil type {soil_type} <= 0")

    bed = row[COL["ptBED"]]
    if is_novalue(bed):
        bed = DEFAULT_BEDROCK_DEPTH

    return PointProperties(
        ID=int(row[COL["ptID"]]),
        East=row[COL["ptX"]], North=row[COL["ptY"]], Z=row[COL["ptZ"]],
        LC=LC, soil_type=soil_type,
        slope=row[COL["ptS"]], aspect=row[COL["ptA"]], sky=row[COL["ptSKY"]],
        curvature1=row[COL["ptCNS"]], curvature2=row[COL["ptCWE"]],
        curvature3=row[COL["ptCNwSe"]], curvature4=row[COL["ptCNeSw"]],
        BC_DepthFreeSurface=row[COL["ptDrDEPTH"]],
        horizon_point=int(row[COL["ptHOR"]]),
        maxSWE=row[COL["ptMAXSWE"]],
        latitude=row[COL["ptLAT"]], longitude=row[COL["ptLON"]],
        bed=bed)


def available_maps(pf, base_dir: str) -> FrozenSet[str]:
    """Which of the topography maps named in ``geotop.inpts`` are on disk."""
    found = set()
    for key, keyword in MAP_KEYWORDS.items():
        stem = pf.string(keyword)
        if stem == STRING_NOVALUE:
            continue
        path = os.path.join(base_dir, stem)
        if any(os.path.exists(path + ext) for ext in MAP_EXTENSIONS):
            found.add(key)
    return frozenset(found)


def column_names(strings: Dict[str, str]) -> List[str]:
    """Resolve the point ``Header*`` keywords against a parsed ``geotop.inpts``."""
    return [strings[k] for k in HEADER_KEYWORDS]


def load(pf, base_dir: str) -> List[List[float]]:
    """The fully resolved point matrix for a run rooted at ``base_dir``."""
    chkpt = points_from_keywords(pf)
    stem = pf.string("PointFile")
    chkpt = read_point_file(None if stem == STRING_NOVALUE
                            else os.path.join(base_dir, stem),
                            column_names(pf.strings), chkpt)
    chkpt = fill_from_maps(chkpt, pf, base_dir)
    return apply_defaults(chkpt, PointOptions.from_parfile(pf))
