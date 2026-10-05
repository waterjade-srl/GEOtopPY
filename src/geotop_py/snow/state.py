"""State of a single snow column.

GEOtop stores the snowpack as a ``Statevar3D`` -- tensors ``Dzl``, ``w_ice``,
``w_liq``, ``T`` indexed ``[layer][row][col]`` (1-based in the layer axis), plus
``lnum[r][c]`` active layers and ``type[r][c]``. Here we fix one (r, c) and keep
just that column, preserving GEOtop's **1-based** layer indexing: arrays have
length ``max + 1`` and index 0 is unused. Layers 1..lnum are active; layers
above ``lnum`` up to ``max`` are allocated but empty.

Units follow GEOtop: Dzl [mm], w_ice/w_liq [kg/m2] (== mm of water equivalent),
T [degC].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .. import laws


class SnowError(RuntimeError):
    """Raised where the C++ would call t_error() and abort the run."""


@dataclass
class SnowColumn:
    max: int                       # allocated number of layers
    lnum: int = 0                  # active layers (1..lnum)
    type: int = 0                  # 0 none, 1 simplified, 2 ordinary
    Dzl: List[float] = field(default_factory=list)     # [mm]
    w_ice: List[float] = field(default_factory=list)   # [kg/m2]
    w_liq: List[float] = field(default_factory=list)   # [kg/m2]
    T: List[float] = field(default_factory=list)       # [degC]

    def __post_init__(self):
        # length max+1, index 0 unused (mirrors the C 1-based tensors)
        for name in ("Dzl", "w_ice", "w_liq", "T"):
            v = getattr(self, name)
            if not v:
                setattr(self, name, [0.0] * (self.max + 1))
            elif len(v) != self.max + 1:
                raise ValueError(
                    f"{name} must have length max+1 = {self.max + 1}, got {len(v)}")

    # --- constructors ----------------------------------------------------

    @classmethod
    def from_layers(cls, layers, max=None):
        """Build from a top-down or bottom-up list of ``(Dzl, w_ice, w_liq, T)``.

        ``layers[0]`` becomes active layer 1, and so on -- same ordering as the
        C tensors, where layer 1 is the bottom of the pack.
        """
        n = len(layers)
        max = max if max is not None else n
        col = cls(max=max, lnum=n, type=2 if n else 0)
        for i, (dz, wi, wl, t) in enumerate(layers, start=1):
            col.Dzl[i] = dz
            col.w_ice[i] = wi
            col.w_liq[i] = wl
            col.T[i] = t
        return col

    # --- diagnostics / invariants ---------------------------------------

    def depth(self) -> float:
        """Total snow depth over all allocated layers [mm]."""
        return sum(self.Dzl[1:self.max + 1])

    def swe(self) -> float:
        """Total snow water equivalent over all allocated layers [kg/m2]."""
        return sum(self.w_ice[l] + self.w_liq[l] for l in range(1, self.max + 1))

    # GEOtop: src/geotop/snow.cc:447-459 (DEPTH)
    # GEOtop: src/geotop/output.cc:305-326 (point snow depth and SWE)
    # GEOtop: src/geotop/snow.cc:1163-1169 (new_snow, type == 0)
    # GEOtop's ``DEPTH()`` sums Dzl over the *active* layers 1..lnum only, and
    # the point output does the same.
    # The distinction is not cosmetic: ``new_snow`` on an empty pack
    # (``type == 0``) parks the fresh snow in Dzl[1]/w_ice[1]
    # while leaving ``lnum == 0``, so between that deposit and the next
    # ``snow_layer_combination`` the pack carries mass that GEOtop does not
    # see -- neither in the reported depth nor in ``snowD``, which drives the
    # canopy burying fraction and the effective vegetation height.
    def active_depth(self) -> float:
        """Snow depth over the active layers 1..lnum [mm] (GEOtop ``DEPTH``)."""
        return sum(self.Dzl[1:self.lnum + 1])

    def active_swe(self) -> float:
        """Snow water equivalent over the active layers 1..lnum [kg/m2]."""
        return sum(self.w_ice[l] + self.w_liq[l] for l in range(1, self.lnum + 1))

    def total_internal_energy(self) -> float:
        """Total internal energy over active layers [J/m2].

        This is the quantity the merge/remap operations must conserve.
        """
        return sum(
            laws.internal_energy(self.w_ice[l], self.w_liq[l], self.T[l])
            for l in range(1, self.lnum + 1)
        )

    def copy(self) -> "SnowColumn":
        return SnowColumn(
            max=self.max, lnum=self.lnum, type=self.type,
            Dzl=list(self.Dzl), w_ice=list(self.w_ice),
            w_liq=list(self.w_liq), T=list(self.T),
        )
