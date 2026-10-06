"""The package's layering, read from the imports themselves.

Every ``geotop_py`` module belongs to one layer; a module may import only from
its own layer or from the ones below it. ``snow``, ``water`` and ``energy``
share a rank and must not import each other. Imports inside functions count
exactly like module-level ones.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "geotop_py"

# Modules sitting directly in the package root, and their layer.  A new root
# module fails the test until it is placed here: the layer is a decision, not a
# default.
ROOT_MODULES = {
    "__init__": "app", "__main__": "app", "cli": "app", "pipeline": "app",
    "initialize": "app", "results": "app",
    "constants": "base", "errors": "base", "laws": "base", "numerics": "base",
    "dates": "base", "psychro": "base", "_cxx": "base",
}

RANK = {"base": 0, "io": 1, "snow": 2, "water": 2, "energy": 2,
        "meteo": 3, "point": 4, "output": 5, "app": 6}


def _modules():
    """Every module of the package as (dotted name relative to the package, path)."""
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__" and len(parts) > 1:
            parts = parts[:-1]
        yield ".".join(parts), path


def _layer(mod):
    head = mod.split(".")[0]
    if head in RANK:
        return head
    return ROOT_MODULES.get(head)


def _is_package(mod):
    return (SRC.joinpath(*mod.split(".")) / "__init__.py").exists()


def _is_module(mod):
    return _is_package(mod) or SRC.joinpath(*mod.split(".")).with_suffix(".py").exists()


def _imported(mod, path):
    """(line, target module) for every geotop_py import made by ``mod``."""
    tree = ast.parse(path.read_text())
    here = mod.split(".") if _is_package(mod) else mod.split(".")[:-1]
    if mod == "__init__":
        here = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("geotop_py."):
                    yield node.lineno, a.name[len("geotop_py."):]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module != "geotop_py" and not (node.module or "").startswith("geotop_py."):
                    continue
                base = node.module.split(".")[1:]
            else:
                up = node.level - 1
                base = (here[:len(here) - up] if up else here) + \
                    (node.module.split(".") if node.module else [])
            for a in node.names:
                sub = ".".join(base + [a.name])
                target = sub if _is_module(sub) else ".".join(base)
                if target:
                    yield node.lineno, target


def _violations():
    out = []
    for mod, path in _modules():
        src = _layer(mod)
        for line, target in _imported(mod, path):
            dst = _layer(target)
            where = f"{path.relative_to(SRC.parent.parent)}:{line}: {mod} -> {target}"
            if RANK[dst] > RANK[src]:
                out.append(f"{where} ({src} imports the higher layer {dst})")
            elif RANK[dst] == RANK[src] and dst != src:
                out.append(f"{where} ({src} and {dst} must stay independent)")
    return out


def test_every_root_module_is_assigned_a_layer():
    unassigned = [m for m, _ in _modules() if _layer(m) is None]
    assert unassigned == [], "place these in ROOT_MODULES: " + ", ".join(unassigned)


@pytest.mark.parametrize("mod, target", [
    ("point.step", "energy.surface"),    # from .. import, parenthesised names
    ("point.step", "water.coupling"),    # import inside a function body
    ("pipeline", "io.parfile"),          # from .io import parfile
    ("io.points", "io.geomorphology"),   # sibling in the same package
])
def test_the_resolver_sees_real_imports(mod, target):
    """An empty result would pass the layering test vacuously."""
    path = dict(_modules())[mod]
    assert target in {t for _, t in _imported(mod, path)}


def test_no_module_imports_above_its_layer():
    assert _violations() == []


@pytest.mark.parametrize("mod, target", [
    ("io.meteo", "energy.rad"),          # a reader reaching into the model
    ("energy.surface", "water.coupling"),  # two peer physics packages
    ("snow.mass_balance", "point.state"),          # physics reaching up to the column state
])
def test_the_checker_catches_each_kind_of_violation(mod, target, monkeypatch):
    """Guard the guard: inject one bad import and require it to be reported."""
    original = _imported

    def with_extra(m, p):
        yield from original(m, p)
        if m == mod:
            yield 0, target

    monkeypatch.setitem(globals(), "_imported", with_extra)
    found = _violations()
    assert len(found) == 1 and f"{mod} -> {target}" in found[0]
