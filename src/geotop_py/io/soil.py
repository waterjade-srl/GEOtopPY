"""Soil column parameters: keyword defaults first, then the soil file over them.

The parameters of a soil column live in a matrix ``pa[row][layer]``, both axes
1-based, whose rows are the fifteen properties named in :data:`ROWS` (thickness,
initial pressure and temperature, the two conductivities, the van Genuchten
closure, the two thermal properties, specific storativity). The same shape is
kept for the bedrock, which GEOtop carries alongside as a second matrix.

The matrix is built in two stages, and the second only patches the first:

1. :func:`soil_parameters_from_keywords` fills every layer from ``geotop.inpts``,
   each layer defaulting to the one above it. Field capacity and wilting point,
   if not given, are computed here from the van Genuchten closure.
2. :func:`read_soil_parameters` reads the soil file, which *sets the number of
   layers*, and takes each cell from the file -- falling back, cell by cell, to
   stage 1's value for that layer, or to that matrix's last layer when the file
   is deeper than the keywords were.

The consequence worth stating: a property the file does not carry keeps the
value stage 1 computed, even when the file overrides the parameters that value
was computed *from*. Field capacity is the case that bites -- see
:func:`fill_field_capacity_and_wilting_point`.

Two sentinels, two meanings, kept distinct throughout: ``NUMBER_ABSENT`` is "the
file has no such column", ``NUMBER_NOVALUE`` is "no value here".

**Scope: one soil type.** All 13 reference cases leave ``SoilLayerTypes`` at 1,
so the multi-type cascade (each type falling back to the one before it, one file
per type) is not implemented and raises instead of guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .. import constants as C
from .. import laws
from .parfile import NUMBER_NOVALUE, STRING_NOVALUE
from .table import NUMBER_ABSENT, read_txt_matrix

#: Soil property rows, in storage order. The index of a name here, plus one, is
#: the row index the matrix is addressed by.
ROWS = ("jdz", "jpsi", "jT", "jKn", "jKl", "jres", "jwp", "jfc",
        "jsat", "ja", "jns", "jv", "jkt", "jct", "jss")
ROW = {name: i + 1 for i, name in enumerate(ROWS)}
NSOILPROP = len(ROWS)

#: The ``Header*`` keyword naming each row's column in the soil file, in row
#: order. These are the ``NSOILPROP`` string keywords that follow the meteo
#: block, which is how they are sliced out of the string keyword table.
HEADER_KEYWORDS = (
    "HeaderSoilDz", "HeaderSoilInitPres", "HeaderSoilInitTemp",
    "HeaderNormalHydrConductivity", "HeaderLateralHydrConductivity",
    "HeaderThetaRes", "HeaderWiltingPoint", "HeaderFieldCapacity",
    "HeaderThetaSat", "HeaderAlpha", "HeaderN", "HeaderV",
    "HeaderKthSoilSolids", "HeaderCthSoilSolids", "HeaderSpecificStorativity",
)

#: The numeric keyword giving each row's pre-file value. ``jdz`` is absent:
#: layer thickness comes from ``SoilLayerThicknesses``/``SoilLayerNumber``,
#: which also decide how many layers there are.
PARAMETER_KEYWORDS: Dict[str, str] = {
    "jpsi": "InitSoilPressure",
    "jT": "InitSoilTemp",
    "jKn": "NormalHydrConductivity",
    "jKl": "LateralHydrConductivity",
    "jres": "ThetaRes",
    "jwp": "WiltingPoint",
    "jfc": "FieldCapacity",
    "jsat": "ThetaSat",
    "ja": "AlphaVanGenuchten",
    "jns": "NVanGenuchten",
    "jv": "VMualem",
    "jkt": "ThermalConductivitySoilSolids",
    "jct": "ThermalCapacitySoilSolids",
    "jss": "SpecificStorativity",
}

#: The same rows for the bedrock. Every one of them defaults to novalue, which
#: is what marks "take the soil's value here" at the end of the cascade.
BEDROCK_KEYWORDS: Dict[str, str] = {
    "jpsi": "InitSoilPressureBedrock",
    "jT": "InitSoilTempBedrock",
    "jKn": "NormalHydrConductivityBedrock",
    "jKl": "LateralHydrConductivityBedrock",
    "jres": "ThetaResBedrock",
    "jwp": "WiltingPointBedrock",
    "jfc": "FieldCapacityBedrock",
    "jsat": "ThetaSatBedrock",
    "ja": "AlphaVanGenuchtenBedrock",
    "jns": "NVanGenuchtenBedrock",
    "jv": "VMualemBedrock",
    "jkt": "ThermalConductivitySoilSolidsBedrock",
    "jct": "ThermalCapacitySoilSolidsBedrock",
    "jss": "SpecificStorativityBedrock",
}

#: Value of each row's topmost layer when ``geotop.inpts`` says nothing.
#: ``jpsi``, ``jwp`` and ``jfc`` default to novalue on purpose: the first means
#: "use the water table depth instead", the other two "compute me".
DEFAULTS: Dict[str, float] = {
    "jpsi": NUMBER_NOVALUE,
    "jT": 5.0,
    "jKn": 1.0e-4,
    "jKl": 1.0e-4,
    "jres": 0.05,
    "jwp": NUMBER_NOVALUE,
    "jfc": NUMBER_NOVALUE,
    "jsat": 0.5,
    "ja": 0.004,
    "jns": 1.3,
    "jv": 0.5,
    "jkt": 2.5,
    "jct": 1.0e6,
    "jss": 1.0e-7,
}

#: Layer thickness [mm] when neither thicknesses nor a soil file are given.
DEFAULT_LAYER_THICKNESS = 100.0
#: Number of layers when neither ``SoilLayerThicknesses`` nor a soil file say.
DEFAULT_LAYER_NUMBER = 5
#: Initial water table depth [mm] below the surface.
DEFAULT_INIT_WATER_TABLE_DEPTH = 5000.0

#: Matric potentials [mm] at which field capacity and wilting point are read off
#: the retention curve: -1/3 bar and -15 bar, as heads of water.
PSI_FIELD_CAPACITY = (-1.0 / 3.0) * 1.0e5 / C.GRAVITY
PSI_WILTING_POINT = -15.0 * 1.0e5 / C.GRAVITY


class SoilError(ValueError):
    """Raised where GEOtop would abort while reading the soil parameters."""


def _absent(value: float) -> bool:
    return int(value) == int(NUMBER_ABSENT)


def _novalue(value: float) -> bool:
    return int(value) == int(NUMBER_NOVALUE)


def _undefined(value: float) -> bool:
    return _absent(value) or _novalue(value)


def _new_matrix(nlayers: int) -> List[List[float]]:
    """A ``[row][layer]`` matrix, both 1-based; index 0 of each axis is unused."""
    return [[0.0] * (nlayers + 1) for _ in range(NSOILPROP + 1)]


@dataclass
class SoilParameters:
    """One soil type's parameters, layer by layer.

    ``pa[ROW[name]][l]`` is property ``name`` of layer ``l``, ``l`` counted from
    the surface down starting at 1. ``pa_bed`` has the same shape and holds the
    bedrock's parameters, used where the bedrock depth cuts into the column.

    ``init_water_table_depth`` [mm below the surface] is the initial condition
    on pressure -- unless every layer's ``jpsi`` is defined, in which case it is
    ``NUMBER_NOVALUE`` to mark that the layer pressures win over it.

    Units follow GEOtop: ``jdz`` [mm], ``jpsi`` [mm], ``jT`` [degC], ``jKn`` and
    ``jKl`` [mm/s], ``ja`` [mm^-1], ``jkt`` [W m^-1 K^-1], ``jct`` [J m^-3 K^-1],
    ``jss`` [mm^-1]; the water contents are volumetric fractions.
    """
    pa: List[List[float]]
    pa_bed: List[List[float]]
    init_water_table_depth: float

    @property
    def nlayers(self) -> int:
        return len(self.pa[1]) - 1

    def row(self, name: str) -> List[float]:
        """Property ``name`` over all layers, as a plain 0-based list."""
        return self.pa[ROW[name]][1:]

    def total_depth(self) -> float:
        """Depth of the modelled soil column [mm]."""
        return sum(self.row("jdz"))


# GEOtop: src/geotop/parameters.cc:1675-1693 (and :1742-1769 for the bedrock)
def fill_field_capacity_and_wilting_point(pa: List[List[float]],
                                          nlayers: int,
                                          guarded: bool = False) -> None:
    """Fill any undefined ``jfc``/``jwp`` from the retention curve, in place.

    Field capacity and wilting point are the water contents the van Genuchten
    closure gives at -1/3 bar and -15 bar. Only layers whose value is still
    novalue are touched, so an explicit value always wins.

    ``guarded`` is the bedrock variant: it computes nothing unless that layer's
    saturation, residual content, alpha and n are all defined *and* the integer
    part of its specific storativity is non-zero -- which, for the storativities
    the model actually uses (order 1e-7), means the bedrock keeps its novalues
    unless a storativity of 1 or more was given.
    """
    for i in range(1, nlayers + 1):
        if guarded:
            defined = all(not _novalue(pa[ROW[name]][i])
                          for name in ("jsat", "jres", "ja", "jns"))
            if not defined or int(pa[ROW["jss"]][i]) == 0:
                continue
        n = pa[ROW["jns"]][i]
        args = (0.0, pa[ROW["jsat"]][i], pa[ROW["jres"]][i], pa[ROW["ja"]][i],
                n, 1.0 - 1.0 / n, C.PsiMin, pa[ROW["jss"]][i])
        if _novalue(pa[ROW["jfc"]][i]):
            pa[ROW["jfc"]][i] = laws.teta_psi(PSI_FIELD_CAPACITY, *args)
        if _novalue(pa[ROW["jwp"]][i]):
            pa[ROW["jwp"]][i] = laws.teta_psi(PSI_WILTING_POINT, *args)


# GEOtop: src/geotop/parameters.cc:1581-1769
def soil_parameters_from_keywords(pf) -> SoilParameters:
    """Build the soil and bedrock matrices from ``geotop.inpts`` alone.

    The number of layers comes from ``SoilLayerThicknesses`` when it lists more
    than one thickness, otherwise from ``SoilLayerNumber`` with every layer the
    same thickness. Every other property is read layer by layer, each layer
    falling back to the value of the layer above it.

    This is the matrix a soil file is then read *over*, not a fallback used only
    when there is no file -- see :func:`read_soil_parameters`.
    """
    nsoiltypes = int(pf.number("SoilLayerTypes", 0, 1.0))
    if nsoiltypes < 1:
        nsoiltypes = 1
    if nsoiltypes > 1:
        raise NotImplementedError(
            "more than one soil type is out of scope for PointSim=1 "
            f"(SoilLayerTypes = {nsoiltypes}); all 13 reference cases use one")

    init_water_table_depth = pf.number("InitWaterTableDepth", 0,
                                       DEFAULT_INIT_WATER_TABLE_DEPTH)

    a = pf.number("SoilLayerThicknesses", 0, NUMBER_NOVALUE)
    if not _novalue(a) and pf.components("SoilLayerThicknesses") > 1:
        nlayers = pf.components("SoilLayerThicknesses")
        pa = _new_matrix(nlayers)
        pa[ROW["jdz"]][1] = a
        for i in range(2, nlayers + 1):
            pa[ROW["jdz"]][i] = pf.number("SoilLayerThicknesses", i - 1,
                                          pa[ROW["jdz"]][i - 1])
    else:
        if _novalue(a):
            a = DEFAULT_LAYER_THICKNESS
        nlayers = int(pf.number("SoilLayerNumber", 0, float(DEFAULT_LAYER_NUMBER)))
        pa = _new_matrix(nlayers)
        for i in range(1, nlayers + 1):
            pa[ROW["jdz"]][i] = a

    for name in ROWS[1:]:
        pa[ROW[name]][1] = pf.number(PARAMETER_KEYWORDS[name], 0, DEFAULTS[name])
    for i in range(2, nlayers + 1):
        for name in ROWS[1:]:
            pa[ROW[name]][i] = pf.number(PARAMETER_KEYWORDS[name], i - 1,
                                         pa[ROW[name]][i - 1])

    fill_field_capacity_and_wilting_point(pa, nlayers)

    if all(not _novalue(pa[ROW["jpsi"]][i]) for i in range(1, nlayers + 1)):
        init_water_table_depth = NUMBER_NOVALUE

    pa_bed = _new_matrix(nlayers)
    for i in range(1, nlayers + 1):
        pa_bed[ROW["jdz"]][i] = pa[ROW["jdz"]][i]
    for name in ROWS[1:]:
        pa_bed[ROW[name]][1] = pf.number(BEDROCK_KEYWORDS[name], 0, NUMBER_NOVALUE)
    for i in range(2, nlayers + 1):
        for name in ROWS[1:]:
            pa_bed[ROW[name]][i] = pf.number(BEDROCK_KEYWORDS[name], i - 1,
                                             pa_bed[ROW[name]][i - 1])
    fill_field_capacity_and_wilting_point(pa_bed, nlayers, guarded=True)

    return SoilParameters(pa=pa, pa_bed=pa_bed,
                          init_water_table_depth=init_water_table_depth)


def _cascade(pa: List[List[float]], defaults: SoilParameters,
             nlayers: int, row: int) -> None:
    """Fill row ``row``'s undefined cells from the keyword matrix, in place.

    Layer ``l`` takes the keyword matrix's own layer ``l``; past its depth it
    repeats the layer above, which by then has already been resolved.
    """
    old = defaults.pa
    old_nlayers = defaults.nlayers
    for j in range(1, nlayers + 1):
        if _undefined(pa[row][j]):
            pa[row][j] = old[row][j] if j <= old_nlayers else pa[row][j - 1]


# GEOtop: src/geotop/parameters.cc:2358-2644
def read_soil_parameters(name: Optional[str], col_names: Sequence[str],
                         defaults: SoilParameters,
                         bed: int = 1) -> SoilParameters:
    """Read the soil file over ``defaults`` and return the resulting parameters.

    ``name`` is the file stem as ``geotop.inpts`` gives it: the file read is
    ``<name>0001.txt``. ``None`` or ``"none"`` means no file, and the keyword
    matrix is used unchanged -- a stem that is given but has no file is fatal,
    as it is in GEOtop.

    ``col_names`` gives the header name of each row, in row order, i.e. the
    values of :data:`HEADER_KEYWORDS` resolved against ``geotop.inpts``.

    The file decides the number of layers. ``bed`` is the soil type the bedrock
    borrows its undefined properties from; with a single soil type it can only
    be 1.
    """
    if len(col_names) != NSOILPROP:
        raise ValueError(f"expected {NSOILPROP} column names, got {len(col_names)}")
    if bed != 1:
        raise NotImplementedError(
            f"bedrock soil type {bed}: only one soil type is in scope")

    rows: List[List[float]] = []
    if name is not None and name != STRING_NOVALUE:
        path = f"{name}0001.txt"
        if not os.path.exists(path):
            raise SoilError(f"soil file {path} not existing")
        rows = read_txt_matrix(path, col_names)
        if not rows:
            raise SoilError(f"{path}: no data lines")

    if rows:
        nlayers = len(rows)
        pa = _new_matrix(nlayers)
        for n in range(1, NSOILPROP + 1):
            for j in range(1, nlayers + 1):
                pa[n][j] = rows[j - 1][n - 1]

        # Layer thickness first and on its own: it is the one row the other
        # soil types are checked against, and the rest of the cascade reads it.
        _cascade(pa, defaults, nlayers, ROW["jdz"])
        for n in range(1, NSOILPROP + 1):
            if n != ROW["jdz"]:
                _cascade(pa, defaults, nlayers, n)

        fill_field_capacity_and_wilting_point(pa, nlayers)

        init_water_table_depth = defaults.init_water_table_depth
        if all(not _novalue(pa[ROW["jpsi"]][j]) for j in range(1, nlayers + 1)):
            init_water_table_depth = NUMBER_NOVALUE
    else:
        nlayers = defaults.nlayers
        pa = [list(r) for r in defaults.pa]
        init_water_table_depth = defaults.init_water_table_depth

    # The bedrock matrix is rebuilt whether or not a soil file was read: it is
    # here, not in the keyword pass, that its remaining novalues are replaced by
    # the soil's own values.
    old_bed = defaults.pa_bed
    old_nlayers = defaults.nlayers
    pa_bed = _new_matrix(nlayers)
    for n in range(1, NSOILPROP + 1):
        if n == ROW["jdz"]:
            for j in range(1, nlayers + 1):
                pa_bed[n][j] = pa[n][j]
            continue
        for j in range(1, nlayers + 1):
            pa_bed[n][j] = old_bed[n][j] if j <= old_nlayers else pa_bed[n][j - 1]
        for j in range(1, nlayers + 1):
            if _novalue(pa_bed[n][j]):
                pa_bed[n][j] = pa[n][j]

    return SoilParameters(pa=pa, pa_bed=pa_bed,
                          init_water_table_depth=init_water_table_depth)


def column_names(strings: Dict[str, str]) -> List[str]:
    """Resolve the soil ``Header*`` keywords against a parsed ``geotop.inpts``."""
    return [strings[k] for k in HEADER_KEYWORDS]


def load(pf, base_dir: str) -> SoilParameters:
    """Keyword defaults plus the soil file, for a run rooted at ``base_dir``."""
    defaults = soil_parameters_from_keywords(pf)
    stem = pf.string("SoilParFile")
    if stem == STRING_NOVALUE:
        return read_soil_parameters(None, column_names(pf.strings), defaults)
    return read_soil_parameters(os.path.join(base_dir, stem),
                                column_names(pf.strings), defaults)
