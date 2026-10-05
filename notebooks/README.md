# Get started notebooks

For users of GEOtop 3.0 (C++) moving to GEOtopPY. They do not explain GEOtop;
they show what stays the same, what changes in practice, and what Python adds.

| Notebook | Content | Run time |
|---|---|---|
| `00_get_started.ipynb` | Clone, install, and a one-day reference case (ColdelaPorte) run in place from the command line | ~5 s |
| `01_from_geotop.ipynb` | The same case run from the shell and from Python, the output files, the comparison with the C++ outputs, the in-memory results, an out-of-scope error | ~40 s |
| `02_matsch_snow.ipynb` | Results on Matsch P2 and B2 (October 2009) and on the ARF_1D arctic snowpack, against the C++ outputs | ~50 s |
| `03_python_workflows.ipynb` | Parameters as the model reads them, parallel parametric runs, internal-step diagnostics, the laws as functions | ~45 s |

Each notebook is independent. `00` runs a case in place and removes its
outputs at the end; the others copy the cases from `tests/1D/` to
`notebooks/runs/` (ignored by git), since they modify them.

## Setup

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[notebooks]'
jupyter lab notebooks/
```

`00_get_started.ipynb` walks through these steps.

`_helpers.py` holds the few helpers the notebooks share (copying a case,
editing keywords, reading GEOtop tables into pandas, comparing with the C++
outputs). It is not part of the `geotop_py` API.

## Notebooks are committed without outputs

Before committing, strip the outputs:

```bash
nbstripout notebooks/*.ipynb
```

or let git do it on every commit with `nbstripout --install`. CI fails on a
notebook that contains outputs, and executes all three to keep them in step
with the code.
