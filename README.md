# GEOtopPY

A Python port of the official GEOtop v3.0 hydrological model in **1D point
configuration**, with coupled energy and water balances. The implementation
preserves GEOtop's numerical algorithms while making state and component
boundaries explicit. The model itself uses only the Python standard library.

> **Disclaimer.** Porting GEOtop is a large undertaking, and this work would
> not have been possible without extensive help from AI coding agents.
> Although all 13 official 1D reference cases pass, discrepancies with the
> original GEOtop may remain, especially in configurations or code paths the
> reference cases do not exercise. Reports of any difference, however small,
> are very welcome.

## Purpose and scope

The guiding principle is reproducibility: given the same inputs, reproduce
what the reference GEOtop implementation computes, within declared numerical
tolerances. This is a translation of the original model, not a recalibration
or a replacement of its equations with similar formulations.

The reference is fixed in `UPSTREAM`: official branch `v3.0`, commit
`1f1fc5afc7a4b0b9daacef181d8be21a8dac47cb`. Source citations beside translated
functions point into that revision. The port supports `PointSim=1`,
`EnergyBalance=1`, and `WaterBalance=0` or `1`.

The implementation covers meteorological preprocessing, surface radiation and
turbulence, vegetation, snow and glacier layers, heat conduction with phase
change, Richards flow in the soil, and the tabular outputs exercised by the
13 official 1D reference cases. Distributed 2D/3D flow, channel routing,
wind-driven snow transport, restart/recovery, and raster/netCDF outputs are
outside this distribution's scope. Passing these cases does not establish
support for every possible GEOtop configuration.

## Implementation choices

- **Translate numerical laws faithfully.** Preserve expressions, argument
  order, physical units, layer ordering and summation order. Original GEOtop
  function names, parameter keywords and scientific symbols are retained for
  traceability and file compatibility; documentation and messages are English.
- **Retain the original solvers.** The energy column uses Backward Euler and
  Newton iterations with a line search and tridiagonal linear solves. Richards
  flow retains GEOtop's Newton and BiCGSTAB algorithms. No alternative energy
  solver is included.
- **Make state explicit.** Dataclasses and function arguments replace access
  to the original global model container. Constitutive laws, solver kernels,
  input/output and simulation orchestration are separated.
- **Expose testable boundaries.** The energy solver receives the surface flux
  and its derivative. Energy and hydrology exchange soil pressure, liquid and
  ice contents, precipitation and evapotranspiration terms.
- **Preserve numerical decisions.** Tolerances, default parameters, reporting
  intervals and timestep retries follow GEOtop. Kernel arithmetic is not
  vectorized: even last-bit changes can alter convergence or snow-layer merges.
  Failed trials are discarded; accepted substeps update persistent state
  before the next substep starts.

`src/geotop_py/` is organized into `energy`, `water`, `snow`, `meteo`, `point`,
`io` and `output`, with shared laws and numerical primitives at package level.

## Install and run

Python 3.9 or newer is required; CI checks Python 3.9, 3.10 and 3.12.
From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
GEOtopPY /path/to/case
```

The case directory must contain `geotop.inpts` and its referenced input files.
Outputs use the paths requested by that configuration. Existing outputs at
those paths are overwritten. Use `GEOtopPY --help` for output suffix and
progress options. Meteorological input uses GEOtop's native CSV format.

## Verify the port

Install the test dependencies, then run from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q -n auto
python -m pytest -q -n auto -m slow
python -m tools.citations --check
python -m tools.dashboard --run
```

A plain `pytest` leaves out the end-to-end tests that simulate the longer
reference cases; `-m slow` runs only those, and CI runs both on every push.
`-n auto` spreads the tests over the available cores (`pytest-xdist`, part of
the `test` extra).

The dashboard **runs new simulations** in `runs/` and compares their outputs
with the bundled official reference tables. A case also fails when the model
writes an output file the reference lacks, since GEOtop writes one only when
its keyword is set. It returns zero only when every selected case passes. Running it without `--run` compares existing results.
A full run includes the long `Bro` case and can take several minutes.
For a shorter acceptance check:

```bash
python -m tools.dashboard --run --exclude Bro
```

Validation has three complementary levels:

1. **Direct numerical comparisons:** individual laws, matrix assembly and
   numerical routines are compared with a C++ oracle, generally around
   `1e-12`, with tolerances chosen explicitly in each test.
2. **Physical and structural invariants:** conservation, residuals, state
   transfer, indexing and timestep retries test behavior beyond the reference
   trajectories. The complete Richards iteration trajectory is not separately
   pinned to a struct-level oracle call.
3. **End-to-end acceptance:** all 13 official 1D cases are compared column by
   column, including output structure and timestamps. A numeric value passes
   when its absolute error is at most `1e-5` **or** its relative error is at
   most `1e-5`; the comparator uses `max(abs(actual), abs(reference))` as the
   relative denominator. The complete acceptance run passed **13/13 cases**.

The bundled inputs and expected tables under `tests/1D/` are unchanged upstream
files. `tests/data/oracle_replay.json.gz` contains recorded oracle responses,
so the oracle tests can run without a compiler or C++ checkout. Answers too
large to record in full (whole meteorological tables and raster maps) are kept
as SHA-256 digests, and those tests compare the digest of the Python result.
Unrecorded calls are explicitly skipped. The recording is tied to `UPSTREAM`; it
is a snapshot of C++ results, not a substitute for rebuilding the oracle.

## Verify against the original C++

For an independent check, install Git and a C++11-capable `g++`, then obtain the
pinned checkout. These commands assume `upstream/` does not already exist:

```bash
git clone https://github.com/geotopmodel/geotop.git upstream
git -C upstream checkout 1f1fc5afc7a4b0b9daacef181d8be21a8dac47cb
bash oracle/build.sh
python -m pytest -q
python -m tools.oracle_replay --verify
python -m tools.citations --check
python -m tools.gen_keywords --check
```

The oracle compiles the unmodified GEOtop sources except its main program,
adds C-compatible wrappers, and exposes the functions through `ctypes`.
`MATH_OPTIM` remains disabled to match the reference arithmetic. Tests also
verify the wrapper boundary, including GEOtop's 1-based array conventions.
The original fatal-error paths can terminate the calling process, so oracle
tests use inputs within the C++ functions' accepted domains.

`GEOTOP_SRC` can select an existing pinned checkout instead of `upstream/`.
Other optional overrides are `GEOTOP_TESTS_1D` (reference cases),
`GEOTOP_ORACLE_SO` (compiled library), and `GEOTOP_REPLAY_TABLE` (recording).
The CI workflow checks the offline tests and shorter acceptance suite on each
push, and all cases plus the rebuilt oracle on scheduled or manual runs.

## Release verification

The distribution was verified on 2026-09-24:

- Offline suite: 2439 passed, 1 skipped (the keyword-table check, which needs
  the GEOtop sources), on Python 3.9 and 3.10, in under two minutes with
  `-n auto`.
- Suite with the compiled oracle and the pinned sources: 2440 passed, no skips.
- End-to-end tests (`-m slow`): 13 passed.
- All 2425 recorded oracle calls reproduced the original results.
- All 13 official acceptance cases passed, with no output file beyond the
  reference ones; source citations, keyword tables, lint and the wheel build
  also passed.

## Validation limits

Reference-case agreement is numerical, not a general bit-for-bit guarantee.
None of the 13 cases fails an energy solve, so that path is covered by
forced-failure tests instead, which follow GEOtop's control flow:

- The first point whose energy solve fails ends the attempt: nothing after its
  solve runs for it, later points are not solved, and no water balance runs
  for any point. The whole domain retries with half the timestep.
- Likewise, the first point whose Richards solve fails ends the water balance:
  later points are not solved, and the net precipitation and mass error keep
  the values of the last completed water balance.
- If the attempt still fails at the minimum timestep it is committed as
  GEOtop commits it, while the clock advances by the halved timestep. Points
  after the failing one keep their state for that interval.
- The output of a point whose step did not complete repeats the values of its
  last completed step, weighted by that step's timestep, so the weights of an
  output interval can exceed one; snow and glacier columns are read from the
  current state. This reproduces GEOtop's output accumulation.

One GEOtop behaviour is reproduced although it does not conserve mass: when the
last glacier layer melts out with no snow above it, the water it still holds
leaves the balance instead of draining to the soil. No reference case has a
glacier.

Some preprocessing branches, including missing-map terrain derivations and
lapse-rate table-file input, are not covered by the 13 cases. Check support
before using configurations outside that validation set.

## License and provenance

GPL-3.0-only. This is a derivative work of GEOtop. `LICENSE` contains the full
license; `NOTICE` records copyright and attribution for the translated code,
reference cases and oracle results. `UPSTREAM` identifies the exact source
revision used for verification.
