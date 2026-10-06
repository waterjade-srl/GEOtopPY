"""GEOtop's comma-separated table reader.

Verbatim translation of ``readline`` / ``readline_of_strings`` /
``readline_of_numbers`` / ``ReadHeader`` / ``ColumnCoder`` / ``read_datamatrix``
/ ``read_txt_matrix`` (``tabs.cc``), which read every tabular input the model
takes: meteo, soil, horizon, point lists.

Two things are not what a CSV reader would normally do, and both change values:

* **The tokenizer discards characters rather than failing on them.** Anything in
  ``c <= 42``, ``58 <= c <= 64``, ``c == 96``, ``c >= 123`` is skipped inside a
  field. So ``17/06/2014 12:00`` -- slash, space and colon all being junk --
  becomes the single number ``170620141200``, which is exactly how GEOtop
  encodes ``Date12``. Note the class differs by one from the one
  ``readline_par`` uses on ``geotop.inpts``: here ``:`` (58) is junk.
* **Numbers go through** :func:`geotop_py.io.parfile.find_number`, GEOtop's own
  decimal parser, not ``float()``.

Missing columns are filled with ``number_absent``; short rows are padded with
``number_novalue``. The two sentinels mean different things downstream --
"the file has no such column" versus "this line stopped early" -- so they are
kept distinct.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..constants import NUMBER_ABSENT, NUMBER_NOVALUE
from .parfile import _Reader, find_number

#: read_txt_matrix is called as read_txt_matrix(name, 33, 44, ...) throughout.
COMMENT_CHAR = 33     # '!'
SEP_CHAR = 44         # ','
NEWLINE = 10

# GEOtop: src/libraries/ascii/tabs.cc:451-452
def _is_junk(c: int) -> bool:
    """The class readline skips inside a field.

    Wider than readline_par's by one codepoint: 58 (``:``) is junk here, which
    is what lets a ``hh:mm`` timestamp collapse into a single number.
    """
    return c <= 42 or (58 <= c <= 64) or c == 96 or c >= 123


def _readline(r: _Reader) -> Tuple[int, Optional[List[List[int]]]]:
    """One line of ``readline``. Returns ``(success, fields)``.

    ``success`` is GEOtop's: 1 for a data line, -1 for a comment or EOF. Fields
    come back as lists of character codes, undecoded, because the caller decides
    whether they are text (``find_string``) or numbers (``find_number``).
    """
    endoffile = False

    while True:
        c = r.getc()
        if c == -1:
            endoffile = True
        if not (not endoffile and _is_junk(c) and c != COMMENT_CHAR):
            break

    if endoffile:
        return -1, None

    if c == COMMENT_CHAR:
        while c != NEWLINE and c != -1:
            c = r.getc()
        return -1, None

    fields: List[List[int]] = []
    first = True
    while True:
        current: List[int] = []
        if first:
            current.append(c)
            first = False
        while True:
            while True:
                c = r.getc()
                if c == -1:
                    endoffile = True
                if not (not endoffile and _is_junk(c)
                        and c != NEWLINE and c != SEP_CHAR):
                    break
            if not endoffile and c != NEWLINE and c != SEP_CHAR:
                current.append(c)
            if endoffile or c == NEWLINE or c == SEP_CHAR:
                break
        fields.append(current)
        if endoffile or c == NEWLINE:
            break

    return 1, fields


def _read_header(r: _Reader) -> List[str]:
    """First data line of the file, as lower-cased column names."""
    while True:
        success, fields = _readline(r)
        if success == 1:
            return [bytes(f).decode("latin-1").lower() for f in fields]
        if r.at_end:
            raise ValueError("file contains only comments")


# GEOtop: src/libraries/ascii/tabs.cc:665-707 (ColumnCoder)
def column_coder(col_descr: Sequence[str], header: Sequence[str]) -> List[int]:
    """Map each wanted column name onto its index in the header, or -1.

    Matching is case-insensitive and the first hit wins; a name appearing twice
    in the header is an error in GEOtop's ``ColumnCoder``, and
    a header column literally named ``none`` never matches.
    """
    coder = [-1] * len(col_descr)
    for i, name in enumerate(col_descr):
        wanted = name.lower()
        for j, got in enumerate(header):
            if wanted == got and coder[i] == -1 and got != "none":
                coder[i] = j
            elif wanted == got and coder[i] != -1:
                raise ValueError(f"column {name!r} appears twice in the header")
    return coder


def read_txt_matrix(path: str, col_descr: Sequence[str]) -> List[List[float]]:
    """Read ``path`` into one row per line, one slot per name in ``col_descr``.

    Slots whose name is not in the file's header get ``NUMBER_ABSENT``; a line
    with fewer fields than the header gets ``NUMBER_NOVALUE`` for the rest.
    """
    with open(path, "rb") as fh:
        reader = _Reader(fh.read())

    header = _read_header(reader)
    coder = column_coder(col_descr, header)
    ncols = len(header)

    rows: List[List[float]] = []
    while True:
        success, fields = _readline(reader)
        if success == 1:
            raw = [find_number(f) for f in fields]
            if len(raw) <= ncols:
                raw = raw + [NUMBER_NOVALUE] * (ncols - len(raw))
            out = [raw[k] if k != -1 else NUMBER_ABSENT for k in coder]
            rows.append(out)
        if reader.at_end:
            break
    return rows


# GEOtop: src/libraries/ascii/tabs.cc:890-916 (read_txt_matrix_2)
def read_txt_matrix_2(path: str, ncols: int) -> List[List[float]]:
    """Read ``path`` into one row per line, ``ncols`` slots per row, by position.

    The header line is consumed and thrown away: unlike :func:`read_txt_matrix`
    the column *names* carry no meaning here, only their order in the file.
    Rows shorter than ``ncols`` are padded with ``NUMBER_NOVALUE``; rows longer
    are truncated.
    """
    with open(path, "rb") as fh:
        reader = _Reader(fh.read())

    _read_header(reader)

    rows: List[List[float]] = []
    while True:
        success, fields = _readline(reader)
        if success == 1:
            raw = [find_number(f) for f in fields]
            if len(raw) <= ncols:
                raw = raw + [NUMBER_NOVALUE] * (ncols - len(raw))
            rows.append(raw[:ncols])
        if reader.at_end:
            break
    return rows

