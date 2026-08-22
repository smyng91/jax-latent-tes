"""Seed-scatter architecture classifier for the (Λ, Ra_d) chart.

A class is the argmin of seed-mean sequential residual J_cycle. Dual is
admitted only as a two-network graph: both metals occupy volume and each
network is 4-connected to its tube. Ties are reported when the gap is
inside seed scatter.
"""

from __future__ import annotations

from collections import deque

import numpy as np

TIE_MARGIN = 0.005
DUAL_MIN_PHI = 0.02
METAL_THRESH = 0.5
CONNECTED_FRAC = 0.5


def pick_winner(
    js: dict[str, float], margin: float = TIE_MARGIN, scatter: dict[str, float] | None = None
) -> str:
    """Lowest J; tie if the gap is inside seed scatter or a fixed margin."""
    names = [k for k in js if js[k] == js[k]]
    names.sort(key=lambda k: js[k])
    if not names:
        return "none"
    best, second = names[0], names[1] if len(names) > 1 else names[0]
    gap = js[second] - js[best]
    noise = margin
    if scatter is not None:
        noise = max(margin, 0.5 * (scatter.get(best, 0.0) + scatter.get(second, 0.0)))
    if len(names) > 1 and gap <= noise:
        return "tie"
    return best


def dual_two_networks(vol_c: float, vol_d: float, min_phi: float = DUAL_MIN_PHI) -> bool:
    return min(vol_c, vol_d) >= min_phi


def _neighbors(i: int, j: int, nx: int, ny: int):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = i + di, j + dj
        if 0 <= ii < nx and 0 <= jj < ny:
            yield ii, jj


def connected_component(metal: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """4-connected metal cells that touch a seed mask (typically a tube)."""
    metal = np.asarray(metal, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    nx, ny = metal.shape
    vis = np.zeros_like(metal, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for i in range(nx):
        for j in range(ny):
            if not seed[i, j]:
                continue
            for ii, jj in _neighbors(i, j, nx, ny):
                if metal[ii, jj] and not vis[ii, jj]:
                    vis[ii, jj] = True
                    q.append((ii, jj))
    while q:
        i, j = q.popleft()
        for ii, jj in _neighbors(i, j, nx, ny):
            if metal[ii, jj] and not vis[ii, jj]:
                vis[ii, jj] = True
                q.append((ii, jj))
    return vis


def dual_tube_graphs(
    gamma_c,
    gamma_d,
    geom,
    thresh: float = METAL_THRESH,
    min_frac: float = CONNECTED_FRAC,
) -> tuple[bool, dict[str, float]]:
    """Charge metal must attach to the charge tube; discharge metal to the discharge tube."""
    design = np.asarray(geom.design)
    gc = (np.asarray(gamma_c) >= thresh) & design
    gd = (np.asarray(gamma_d) >= thresh) & design
    reach_c = connected_component(gc, np.asarray(geom.charge))
    reach_d = connected_component(gd, np.asarray(geom.discharge))
    n_c = int(gc.sum())
    n_d = int(gd.sum())
    frac_c = float(reach_c.sum() / max(n_c, 1))
    frac_d = float(reach_d.sum() / max(n_d, 1))
    ok = bool(reach_c.any() and reach_d.any() and frac_c >= min_frac and frac_d >= min_frac)
    return ok, {"frac_c": frac_c, "frac_d": frac_d, "n_c": n_c, "n_d": n_d}


def dual_admitted(vol_c: float, vol_d: float, gamma_c=None, gamma_d=None, geom=None) -> bool:
    if not dual_two_networks(vol_c, vol_d):
        return False
    if gamma_c is None or gamma_d is None or geom is None:
        return True
    ok, _ = dual_tube_graphs(gamma_c, gamma_d, geom)
    return ok


def reclassify_cell(cell: dict, geom) -> dict:
    """Drop dual if the saved networks fail the tube-graph test; otherwise keep winner."""
    dual = cell.get("winner_dual") or cell.get("dual_fields") or {}
    gc, gd = dual.get("gamma_c"), dual.get("gamma_d")
    vol_c = dual.get("vol_c", 0.0)
    vol_d = dual.get("vol_d", 0.0)
    graphs_ok = None
    if gc is not None and gd is not None:
        graphs_ok, stats = dual_tube_graphs(gc, gd, geom)
        cell = {**cell, "dual_graph_stats": stats, "dual_tube_graphs": graphs_ok}
        admitted = dual_two_networks(vol_c, vol_d) and graphs_ok
    else:
        admitted = bool(cell.get("dual_two_networks"))
    cell = {**cell, "dual_admitted": admitted}
    if cell.get("winner") == "dual" and not admitted:
        mean_j = {k: v for k, v in cell["mean_J"].items() if k != "dual"}
        cell = {**cell, "winner": pick_winner(mean_j, scatter=cell.get("std_J")), "dual_dropped": "graph"}
    return cell
