"""The soil parameter reader, checked against GEOtop's own.

Two independent references, because they cover different halves:

* the oracle runs ``read_soil_parameters`` itself, which pins the *file cascade*
  at bit equality -- but only given the same pre-file matrix, so it says nothing
  about whether that matrix was built from the right keywords;
* the reference runs' ``geotop.log-SE27XX`` prints the finished matrix, which
  pins the whole chain -- keyword pass included -- at the six decimals the log
  carries.

The second is the one that catches a keyword read from the wrong name, which is
exactly the failure a differential test cannot see (both sides then agree on the
same wrong number).
"""

import os
import re

import pytest

from geotop_py.io import parfile, soil
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _cases():
    """(case, case directory) for every reference case."""
    if not os.path.isdir(REFERENCE_1D):
        return []
    return [(c, os.path.join(REFERENCE_1D, c))
            for c in sorted(os.listdir(REFERENCE_1D))
            if os.path.exists(os.path.join(REFERENCE_1D, c, "geotop.inpts"))]


CASES = _cases()
needs_reference = pytest.mark.skipif(
    not CASES, reason=f"reference cases not found under {REFERENCE_1D}")


def _stem(pf, directory, keyword):
    value = pf.string(keyword)
    if value == parfile.STRING_NOVALUE:
        return None
    return os.path.join(directory, value)


def _log_matrix(directory, marker):
    """The soil matrix GEOtop logged, as ``{header keyword: [layer values]}``."""
    path = os.path.join(directory, "geotop.log-SE27XX")
    lines = open(path, errors="replace").read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(marker))
    nlayers = int(lines[start].split(":")[1])
    out = {}
    for line in lines[start + 2:start + 2 + soil.NSOILPROP]:
        name, rest = line.split(":", 1)
        out[name] = [float(m.group(1)) for m in re.finditer(r"([-\d.e+]+)\(", rest)]
    return nlayers, out


# --------------------------------------------------------------- slot mapping

def test_row_indices_match_the_cxx():
    for name, index in soil.ROW.items():
        assert _cxx.SOIL[name] == index, name
    assert soil.NSOILPROP == _cxx.SOIL["nsoilprop"]


def test_header_keywords_are_the_soil_slice_of_the_string_table():
    """assign_string_parameter slices them positionally, right after the meteo
    block. If the two ever diverge, every soil column shifts by one row."""
    start = _cxx.NMET
    assert list(soil.HEADER_KEYWORDS) == \
        _cxx.KEYWORDS_CHAR[start:start + soil.NSOILPROP]


def test_parameter_keywords_sit_where_the_cxx_indexes_them():
    """The pre-file matrix is filled by ``cod + row - 2``, one contiguous run of
    keywords starting two past ``SoilLayerThicknesses``. Checking the names
    against the compiled table is the only thing that catches an invented one:
    a differential test would take the same default on both sides."""
    base = _cxx.KEYWORDS_NUM.index("SoilLayerThicknesses")
    assert _cxx.KEYWORDS_NUM[base + 1] == "SoilLayerNumber"
    for name in soil.ROWS[1:]:
        assert _cxx.KEYWORDS_NUM[base + 2 + soil.ROW[name] - 2] == \
            soil.PARAMETER_KEYWORDS[name], name

    bedrock = _cxx.KEYWORDS_NUM.index("InitSoilPressureBedrock")
    assert bedrock == base + 2 + soil.NSOILPROP - 1
    for name in soil.ROWS[1:]:
        assert _cxx.KEYWORDS_NUM[bedrock + soil.ROW[name] - 2] == \
            soil.BEDROCK_KEYWORDS[name], name


def test_the_layer_count_keywords_are_real():
    for name in ("SoilLayerTypes", "InitWaterTableDepth", "SoilLayerThicknesses",
                 "SoilLayerNumber", "DefaultSoilTypeBedrock"):
        assert name in _cxx.KEYWORDS_NUM, name


# ------------------------------------------------------------- the 13 cases

@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_file_cascade_matches_the_cxx(case, directory):
    """Same pre-file matrix in, same resolved matrix out, at bit equality."""
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    names = soil.column_names(pf.strings)
    defaults = soil.soil_parameters_from_keywords(pf)
    stem = _stem(pf, directory, "SoilParFile")

    mine = soil.read_soil_parameters(stem, names, defaults)
    pa, pa_bed, iwtd = _cxx.read_soil_parameters(
        stem if stem is not None else _cxx.STRING_NOVALUE, names,
        defaults.pa, defaults.pa_bed, defaults.init_water_table_depth)

    assert mine.pa == pa
    assert mine.pa_bed == pa_bed
    assert mine.init_water_table_depth == iwtd


@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
@pytest.mark.parametrize("marker,attr",
                         [("Soil Layers:", "pa"),
                          ("Soil Bedrock Layers:", "pa_bed")])
def test_whole_chain_matches_the_reference_log(case, directory, marker, attr):
    """Keyword pass included -- this is what pins the keyword names themselves."""
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    mine = soil.load(pf, directory)
    nlayers, reference = _log_matrix(directory, marker)

    assert mine.nlayers == nlayers
    matrix = getattr(mine, attr)
    for row, header in enumerate(soil.HEADER_KEYWORDS, start=1):
        got = matrix[row][1:]
        want = reference[header]
        assert len(got) == len(want), header
        for a, b in zip(got, want):
            assert a == pytest.approx(b, rel=1e-6, abs=1e-6), header


@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_resolved_parameters_are_physically_plausible(case, directory):
    """An invariant check that does not depend on the reference at all."""
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    sp = soil.load(pf, directory)

    assert sp.nlayers >= 1
    assert all(dz > 0.0 for dz in sp.row("jdz"))
    assert 100.0 <= sp.total_depth() <= 1.0e6      # 10 cm to 1 km of soil
    for layer in range(sp.nlayers):
        res = sp.row("jres")[layer]
        sat = sp.row("jsat")[layer]
        assert 0.0 <= res < sat <= 1.0
        # jwp <= jfc always holds: both come from the same teta_psi call (same
        # sat/res/alpha/n) evaluated at PSI_WILTING_POINT <= PSI_FIELD_CAPACITY,
        # and teta_psi is monotonic increasing in psi. But neither is bounded
        # by *this layer's own* jres/jsat: per
        # test_field_capacity_comes_from_the_keyword_defaults_not_from_the_file,
        # a file that overrides jsat/jres/ja/jns without also giving jfc/jwp
        # leaves jfc/jwp computed from the pre-file *keyword* defaults, not the
        # file's values -- so they need not fall inside [res, sat] here.
        assert 0.0 <= sp.row("jwp")[layer] <= sp.row("jfc")[layer] <= 1.0
        assert sp.row("jns")[layer] > 1.0          # m = 1 - 1/n must be positive
        assert sp.row("ja")[layer] > 0.0
        assert sp.row("jKn")[layer] > 0.0 and sp.row("jKl")[layer] > 0.0


# ------------------------------------------------- what a "sensible" port breaks

@needs_reference
def test_field_capacity_comes_from_the_keyword_defaults_not_from_the_file():
    """Field capacity comes from the keyword defaults not from the file."""
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    sp = soil.load(pf, directory)

    # The file overrides all four retention parameters. Compared against
    # find_number's own parse, not a Python float literal: "0.6" and "0.61"
    # are two literals that land 1 ULP away from float(). The soil file uses
    # the same tokenizer as geotop.inpts, so the same divergence applies here.
    assert sp.row("jsat")[0] == _cxx.find_number("0.6")
    assert sp.row("jres")[0] == _cxx.find_number("0.1")
    assert sp.row("ja")[0] == _cxx.find_number("0.001")
    assert sp.row("jns")[0] == _cxx.find_number("1.8")

    from_defaults = _cxx.teta_psi(soil.PSI_FIELD_CAPACITY, 0.0,
                                  soil.DEFAULTS["jsat"], soil.DEFAULTS["jres"],
                                  soil.DEFAULTS["ja"], soil.DEFAULTS["jns"],
                                  1.0 - 1.0 / soil.DEFAULTS["jns"],
                                  -1.0e10, soil.DEFAULTS["jss"])
    from_the_file = _cxx.teta_psi(soil.PSI_FIELD_CAPACITY, 0.0,
                                  0.6, 0.1, 0.001, 1.8, 1.0 - 1.0 / 1.8,
                                  -1.0e10, 1.0e-6)
    assert from_defaults != pytest.approx(from_the_file, rel=1e-3)
    assert sp.row("jfc")[0] == from_defaults


def test_field_capacity_and_wilting_point_are_pinned_to_teta_psi():
    """The two heads, and the closure they are read off, at bit equality."""
    pa = [[0.0] * 2 for _ in range(soil.NSOILPROP + 1)]
    pa[soil.ROW["jdz"]][1] = 100.0
    pa[soil.ROW["jsat"]][1] = 0.45
    pa[soil.ROW["jres"]][1] = 0.06
    pa[soil.ROW["ja"]][1] = 0.0021
    pa[soil.ROW["jns"]][1] = 1.42
    pa[soil.ROW["jss"]][1] = 1.0e-7
    pa[soil.ROW["jfc"]][1] = parfile.NUMBER_NOVALUE
    pa[soil.ROW["jwp"]][1] = parfile.NUMBER_NOVALUE

    soil.fill_field_capacity_and_wilting_point(pa, 1)

    args = (0.0, 0.45, 0.06, 0.0021, 1.42, 1.0 - 1.0 / 1.42, -1.0e10, 1.0e-7)
    assert pa[soil.ROW["jfc"]][1] == _cxx.teta_psi(
        (-1.0 / 3.0) * 1.0e5 / 9.81, *args)
    assert pa[soil.ROW["jwp"]][1] == _cxx.teta_psi(
        -15.0 * 1.0e5 / 9.81, *args)


def test_an_explicit_field_capacity_is_never_overwritten():
    pa = [[0.0] * 2 for _ in range(soil.NSOILPROP + 1)]
    pa[soil.ROW["jsat"]][1] = 0.45
    pa[soil.ROW["jres"]][1] = 0.06
    pa[soil.ROW["ja"]][1] = 0.0021
    pa[soil.ROW["jns"]][1] = 1.42
    pa[soil.ROW["jss"]][1] = 1.0e-7
    pa[soil.ROW["jfc"]][1] = 0.31
    pa[soil.ROW["jwp"]][1] = parfile.NUMBER_NOVALUE

    soil.fill_field_capacity_and_wilting_point(pa, 1)
    assert pa[soil.ROW["jfc"]][1] == 0.31


def test_the_bedrock_guard_reads_the_integer_part_of_the_storativity():
    """The bedrock's field capacity is computed only when ``(long)Ss != 0``.

    That is an integer cast of a specific storativity whose realistic values are
    around 1e-7, so the guard is false for every physically sensible input and
    the bedrock keeps its novalues -- which is what makes the *soil's* value be
    substituted for them later. Writing the guard as ``Ss != 0.0``, which is
    what it looks like it means, would silently start computing them.
    """
    def bedrock_row(ss):
        pa = [[0.0] * 2 for _ in range(soil.NSOILPROP + 1)]
        pa[soil.ROW["jsat"]][1] = 0.45
        pa[soil.ROW["jres"]][1] = 0.06
        pa[soil.ROW["ja"]][1] = 0.0021
        pa[soil.ROW["jns"]][1] = 1.42
        pa[soil.ROW["jss"]][1] = ss
        pa[soil.ROW["jfc"]][1] = parfile.NUMBER_NOVALUE
        pa[soil.ROW["jwp"]][1] = parfile.NUMBER_NOVALUE
        soil.fill_field_capacity_and_wilting_point(pa, 1, guarded=True)
        return pa[soil.ROW["jfc"]][1]

    assert bedrock_row(1.0e-7) == parfile.NUMBER_NOVALUE
    assert bedrock_row(2.0) != parfile.NUMBER_NOVALUE


def test_the_bedrock_guard_needs_every_retention_parameter():
    pa = [[0.0] * 2 for _ in range(soil.NSOILPROP + 1)]
    pa[soil.ROW["jsat"]][1] = 0.45
    pa[soil.ROW["jres"]][1] = 0.06
    pa[soil.ROW["ja"]][1] = parfile.NUMBER_NOVALUE
    pa[soil.ROW["jns"]][1] = 1.42
    pa[soil.ROW["jss"]][1] = 2.0
    pa[soil.ROW["jfc"]][1] = parfile.NUMBER_NOVALUE
    pa[soil.ROW["jwp"]][1] = parfile.NUMBER_NOVALUE

    soil.fill_field_capacity_and_wilting_point(pa, 1, guarded=True)
    assert pa[soil.ROW["jfc"]][1] == parfile.NUMBER_NOVALUE


# ----------------------------------------------------------- the cascade edges

def _write_soil_file(tmp_path, header, rows):
    stem = tmp_path / "soil"
    path = tmp_path / "soil0001.txt"
    path.write_text(",".join(header) + "\n"
                    + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    return str(stem)


@needs_reference
def test_a_file_deeper_than_the_keywords_repeats_the_last_keyword_layer(tmp_path):
    """Layers past the keyword matrix's depth take the layer above them, not the
    keyword default -- and the layer above may itself have come from the file."""
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    names = soil.column_names(pf.strings)
    defaults = soil.soil_parameters_from_keywords(pf)
    assert defaults.nlayers == 5           # SoilLayerNumber absent -> 5 x 100 mm

    # Seven layers, and saturation only given on the first two.
    rows = [[100.0, 0.55], [100.0, 0.61]] + [[100.0, -9999.0]] * 5
    stem = _write_soil_file(tmp_path, ["Dz", "sat"], rows)

    mine = soil.read_soil_parameters(stem, names, defaults)
    pa, pa_bed, iwtd = _cxx.read_soil_parameters(
        stem, names, defaults.pa, defaults.pa_bed,
        defaults.init_water_table_depth)

    assert mine.nlayers == 7
    assert mine.pa == pa
    assert mine.pa_bed == pa_bed
    assert mine.init_water_table_depth == iwtd
    # layer 3..5 fall back to the keyword matrix, 6..7 repeat layer 5. Compared
    # against find_number's own parse of "0.55"/"0.61", not the Python float
    # literals -- "0.61" is 1 ULP away from float("0.61") (see the note in
    # test_field_capacity_comes_from_the_keyword_defaults_not_from_the_file).
    v55, v61 = _cxx.find_number("0.55"), _cxx.find_number("0.61")
    assert mine.row("jsat") == [v55, v61, 0.5, 0.5, 0.5, 0.5, 0.5]


@needs_reference
def test_layer_pressures_given_everywhere_disable_the_water_table(tmp_path):
    """``InitWaterTableDepth`` is dropped for a soil type whose every layer has
    an initial pressure: the two are alternative initial conditions, and the
    per-layer one wins. The flag is the water table depth itself going novalue.
    """
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    names = soil.column_names(pf.strings)
    defaults = soil.soil_parameters_from_keywords(pf)
    assert defaults.init_water_table_depth == 5000.0

    # PureDrainage does not set HeaderSoilInitPres, so that slot -- like every
    # other Header* keyword the case leaves unset -- resolves to STRING_NOVALUE
    # ("none"). Several slots share that same "none" name; picking a distinct
    # header string for the file column keeps this test's match unambiguous,
    # rather than writing a file column literally named "none".
    names = list(names)
    names[soil.ROW["jpsi"] - 1] = "InitPres"
    header = ["Dz", "InitPres"]
    rows = [[100.0, -300.0], [100.0, -200.0], [100.0, -100.0]]
    stem = _write_soil_file(tmp_path, header, rows)

    mine = soil.read_soil_parameters(stem, names, defaults)
    assert mine.init_water_table_depth == parfile.NUMBER_NOVALUE

    pa, pa_bed, iwtd = _cxx.read_soil_parameters(
        stem, names, defaults.pa, defaults.pa_bed,
        defaults.init_water_table_depth)
    assert mine.init_water_table_depth == iwtd
    assert mine.pa == pa


@needs_reference
def test_one_missing_layer_pressure_keeps_the_water_table(tmp_path):
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    names = soil.column_names(pf.strings)
    defaults = soil.soil_parameters_from_keywords(pf)

    header = ["Dz", pf.string("HeaderSoilInitPres")]
    rows = [[100.0, -300.0], [100.0, -9999.0], [100.0, -100.0]]
    stem = _write_soil_file(tmp_path, header, rows)

    # The novalue cascades to the keyword matrix's own jpsi, which is novalue.
    mine = soil.read_soil_parameters(stem, names, defaults)
    assert mine.init_water_table_depth == 5000.0

    _, _, iwtd = _cxx.read_soil_parameters(
        stem, names, defaults.pa, defaults.pa_bed,
        defaults.init_water_table_depth)
    assert mine.init_water_table_depth == iwtd


# ------------------------------------------------------------------ the edges

@needs_reference
def test_a_declared_soil_file_that_does_not_exist_is_fatal(tmp_path):
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    defaults = soil.soil_parameters_from_keywords(pf)
    with pytest.raises(soil.SoilError):
        soil.read_soil_parameters(str(tmp_path / "nowhere"),
                                  soil.column_names(pf.strings), defaults)


@needs_reference
def test_no_soil_file_leaves_the_keyword_matrix_alone():
    """ColdelaPorte has no SoilParFile: every property is a keyword default."""
    directory = os.path.join(REFERENCE_1D, "ColdelaPorte")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    assert pf.string("SoilParFile") == parfile.STRING_NOVALUE

    defaults = soil.soil_parameters_from_keywords(pf)
    mine = soil.load(pf, directory)
    assert mine.pa == defaults.pa
    assert mine.nlayers == defaults.nlayers


@needs_reference
def test_more_than_one_soil_type_is_out_of_scope():
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    pf.numeric["SoilLayerTypes"] = [3.0]
    with pytest.raises(NotImplementedError):
        soil.soil_parameters_from_keywords(pf)


@needs_reference
def test_every_reference_case_uses_a_single_soil_type():
    """The premise the scope reduction rests on, checked rather than assumed."""
    for case, directory in CASES:
        pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
        assert int(pf.number("SoilLayerTypes", 0, 1.0)) == 1, case
