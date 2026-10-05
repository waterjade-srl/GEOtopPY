"""Read a GEOtop v3.0 ``geotop.inpts``.

Verbatim translation of the parsing half of ``read_inpts_par`` and of the
tokenizer it drives, ``readline_par``. The result is the raw table
``assign_numeric_parameters`` works from: every keyword of ``keywords.h`` mapped
to the components the file declared, or to GEOtop's "absent" sentinel.

Three details are load-bearing, and all three are silently wrong in the obvious
implementation:

* **Numbers do not go through ``float()``.** GEOtop parses decimals by hand
  (``find_number``), accumulating digit by digit with ``pow(10, cnt)``. On the
  13 reference cases, 23 of 208 distinct literals -- including ``0.15``,
  ``0.6``, ``0.65`` -- land one ULP away from what ``float()`` returns.
  These are albedos, porosities and van Genuchten parameters feeding a
  nonlinear solve, so the port reproduces the arithmetic rather than the intent.
* **The opening quote of a string is part of the token.** ``readline_par`` has
  to keep it, because ``string[0] != 34`` is how it tells a string from a
  number; ``find_string_int`` then drops it with
  ``string[i] = vector[i+1]``, which is why the reported length is ``i - 1``.
* **Keyword matching is case-insensitive** (``convert_string_in_lower_case``).

Verified against the oracle -- which runs GEOtop's own tokenizer -- on all 13
reference cases, and against GEOtop's logged values, in ``tests/test_parfile.py``.
"""

# GEOtop: src/geotop/parameters.cc:104-231 (the parsing half of read_inpts_par)
# GEOtop: src/libraries/ascii/tabs.cc:47-270 (readline_par)
# GEOtop: src/libraries/ascii/tabs.cc:401-414 (find_string_int)

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# readline_par is called as readline_par(f, 33, 61, 44, ...) throughout GEOtop.
COMMENT_CHAR = 33     # '!'
SEPFIELD_CHAR = 61    # '='
SEPVECT_CHAR = 44     # ','
QUOTE = 34            # '"'
NEWLINE = 10

# GEOtop: src/geotop/geotop.cc:157-159
#: GEOtop's sentinels for a keyword the file does not mention.
NUMBER_NOVALUE = -9999.0
NUMBER_ABSENT = -9998.0
STRING_NOVALUE = "none"


# GEOtop: src/libraries/ascii/tabs.cc:277-353 (find_number)
def find_number(chars: Sequence[int]) -> float:
    """GEOtop's decimal parser, ``find_number``, on a sequence of char codes.

    Digits are accumulated against ``pow(10, cnt)`` rather than converted by the
    C library, and non-digits inside the number are simply skipped -- so this
    also defines what GEOtop makes of a malformed token, which ``float()`` would
    reject outright.
    """
    n = len(chars)
    if n == 0:
        return 0.0

    N = 0.0
    Nexp = 0.0

    # position of the exponent marker
    ie = -1
    i = 0
    while True:
        if chars[i] == 69 or chars[i] == 101:      # 'E' or 'e'
            ie = i
        i += 1
        if not (ie == -1 and i < n):
            break
    if ie == -1:
        ie = n

    # position of the decimal separator
    ids = -1
    i = 0
    while True:
        if chars[i] == 46:                          # '.'
            ids = i
        i += 1
        if not (ids == -1 and i < n):
            break
    if ids == -1:
        ids = ie

    # integer part
    cnt = 0
    for i in range(ids - 1, -1, -1):
        if 48 <= chars[i] <= 57:
            N += float(chars[i] - 48) * math.pow(10.0, float(cnt))
            cnt += 1

    # fractional part
    cnt = -1
    for i in range(ids + 1, ie):
        if 48 <= chars[i] <= 57:
            N += float(chars[i] - 48) * math.pow(10.0, float(cnt))
            cnt -= 1

    # exponential part
    cnt = 0
    for i in range(n - 1, ie, -1):
        if 48 <= chars[i] <= 57:
            Nexp += float(chars[i] - 48) * math.pow(10.0, float(cnt))
            cnt += 1

    if ie < n - 1 and chars[ie + 1] == 45:          # '-'
        Nexp *= -1.0

    N = N * math.pow(10.0, Nexp)

    if chars[0] == 45:
        N *= -1.0

    return N


def _is_junk_key(c: int) -> bool:
    """The character class readline_par skips while looking for a keyword."""
    return c <= 46 or (59 <= c <= 64) or c == 96 or c >= 123


def _is_junk_arg(c: int) -> bool:
    """The class it skips inside an argument -- note `.` and `-` are kept here."""
    return c <= 42 or c == 44 or (59 <= c <= 64) or c == 96 or c >= 123


class _Reader:
    """A byte cursor with fgetc semantics: -1 at end of file."""

    def __init__(self, data: bytes):
        self._data = data
        self._i = 0

    def getc(self) -> int:
        if self._i >= len(self._data):
            return -1
        c = self._data[self._i]
        self._i += 1
        return c

    @property
    def at_end(self) -> bool:
        return self._i >= len(self._data)


def _readline_par(r: _Reader) -> Tuple[int, Optional[str], Optional[List[float]],
                                       Optional[str], bool]:
    """One line of readline_par.

    Returns ``(res, keyword, numbers, string, endoffile)`` where ``res`` is
    GEOtop's own code: -1 comment or EOF, 0 keyword with no argument,
    1 numeric, 2 string.
    """
    endoffile = False

    # first character of the line
    while True:
        c = r.getc()
        if c == -1:
            endoffile = True
        if not (not endoffile and _is_junk_key(c) and c != COMMENT_CHAR):
            break

    if endoffile:
        return -1, None, None, None, True

    if c == COMMENT_CHAR:
        while c != NEWLINE and c != -1:
            c = r.getc()
        return -1, None, None, None, c == -1

    # keyword: the character that ended the skip loop is its first
    key: List[int] = [c]
    while True:
        while True:
            c = r.getc()
            if c == -1:
                endoffile = True
            if not (not endoffile and _is_junk_key(c)
                    and c != SEPFIELD_CHAR and c != NEWLINE):
                break
        if not endoffile and c != SEPFIELD_CHAR and c != NEWLINE:
            key.append(c)
        if endoffile or c == SEPFIELD_CHAR or c == NEWLINE:
            break

    keyword = bytes(key).decode("latin-1")

    if endoffile or c == NEWLINE:
        return 0, keyword, None, None, endoffile

    # argument
    arg: List[int] = []
    while True:
        while True:
            c = r.getc()
            if c == -1:
                endoffile = True
            i = len(arg)
            if not (not endoffile and _is_junk_arg(c) and c != NEWLINE
                    and c != SEPVECT_CHAR and (i > 0 or c != QUOTE)):
                break
        if not endoffile and c != NEWLINE:
            arg.append(c)
        if endoffile or c == NEWLINE:
            break

    if not arg:
        return 0, keyword, None, None, endoffile

    if arg[0] != QUOTE:
        # numeric, possibly a vector separated by SEPVECT_CHAR
        numbers: List[float] = []
        piece: List[int] = []
        for ch in arg:
            if ch == SEPVECT_CHAR:
                numbers.append(find_number(piece))
                piece = []
            else:
                piece.append(ch)
        numbers.append(find_number(piece))
        return 1, keyword, numbers, None, endoffile

    # string: drop the leading quote, and the last character with it
    # (stringlength = i - 1, then find_string_int shifts by one)
    text = bytes(arg[1:len(arg)]).decode("latin-1")
    return 2, keyword, None, text, endoffile


@dataclass
class ParFile:
    """The parsed ``geotop.inpts``, keyed by keyword name.

    ``numeric[name]`` holds the components the file declared, or
    ``[NUMBER_NOVALUE]`` when it did not mention the keyword;
    ``strings[name]`` holds the value or ``STRING_NOVALUE``.

    Nothing here applies a default: this is deliberately the table *before*
    ``assign_numeric_parameters`` interprets it.
    """
    numeric: Dict[str, List[float]]
    strings: Dict[str, str]
    path: str = ""

    def components(self, name: str) -> int:
        """How many components the file declared (1 when absent, holding novalue)."""
        self._check(name, self.numeric, "numeric")
        return len(self.numeric[name])

    def _check(self, name: str, table: Dict[str, object], kind: str) -> None:
        """An unknown keyword is a bug, not a missing value.

        Without this a typo silently takes the default, which looks like the
        parameter was simply not declared -- the one failure mode a differential
        test cannot catch, because both sides then agree on the wrong number.
        """
        if name not in table:
            raise KeyError(f"{name!r} is not a GEOtop {kind} keyword")

    def has_number(self, name: str, j: int = 0) -> bool:
        self._check(name, self.numeric, "numeric")
        v = self.numeric[name]
        if j >= len(v):
            return False
        return v[j] != NUMBER_NOVALUE

    def number(self, name: str, j: int = 0,
               default: Optional[float] = None) -> float:
        """``assignation_number`` semantics: the declared component, else default.

        GEOtop treats a component equal to ``number_novalue`` as absent, so an
        explicit ``-9999`` in the file falls back to the default just like a
        missing keyword does.
        """
        if self.has_number(name, j):
            return self.numeric[name][j]
        if default is None:
            return NUMBER_NOVALUE
        return default

    def has_string(self, name: str) -> bool:
        self._check(name, self.strings, "string")
        return self.strings[name] != STRING_NOVALUE

    def string(self, name: str, default: Optional[str] = None) -> str:
        if self.has_string(name):
            return self.strings[name]
        return STRING_NOVALUE if default is None else default


def parse(path: str, keywords_num: Optional[Sequence[str]] = None,
          keywords_char: Optional[Sequence[str]] = None) -> ParFile:
    """Parse ``path`` against GEOtop's two keyword tables.

    The tables default to the generated ones in :mod:`geotop_py.io.keywords`, so a
    normal run needs no oracle. They can be passed in instead, which is how the
    tests feed the compiled model's own tables and check the two agree.
    """
    if keywords_num is None or keywords_char is None:
        from . import keywords as _kw
        keywords_num = _kw.KEYWORDS_NUM if keywords_num is None else keywords_num
        keywords_char = _kw.KEYWORDS_CHAR if keywords_char is None else keywords_char

    with open(path, "rb") as fh:
        reader = _Reader(fh.read())

    read_num: List[Tuple[str, List[float]]] = []
    read_str: List[Tuple[str, str]] = []
    while True:
        res, keyword, numbers, text, endoffile = _readline_par(reader)
        if res == 1:
            read_num.append((keyword.lower(), numbers))
        elif res == 2:
            read_str.append((keyword.lower(), text))
        if endoffile:
            break

    # Later occurrences win, as in the C++ (the match loop does not break).
    numeric = {name: [NUMBER_NOVALUE] for name in keywords_num}
    for name in keywords_num:
        want = name.lower()
        for got, values in read_num:
            if got == want:
                numeric[name] = list(values)

    strings = {name: STRING_NOVALUE for name in keywords_char}
    for name in keywords_char:
        want = name.lower()
        for got, value in read_str:
            if got == want:
                strings[name] = value

    return ParFile(numeric=numeric, strings=strings, path=os.fspath(path))
