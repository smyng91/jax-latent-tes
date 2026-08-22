"""Equal-volume annular-fin baseline versus topology-optimized metal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pcm.design import volume_fraction
from pcm.optimize import BETAS, cross_eval, optimize_architecture
from pcm.physics import simulate_liquid_trace, tube_bcs, two_tube_geom
from pcm.sweep import cycle_j, make_sims

jax.config.update("jax_enable_x64", True)


def annular_gamma(geom, phi: float, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0, r_tube: float = 0.08):
    """Metal sheath around each tube; outer radius chosen so design-cell volume is phi."""
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    d = np.minimum(np.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.22 * Ly) ** 2), np.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.78 * Ly) ** 2))
    design = np.asarray(geom.design)
    lo, hi = r_tube, 0.55
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        vol = float(np.sum((d < mid) & design) / (np.sum(design) + 1e-12))
        if vol < phi:
            lo = mid
        else:
            hi = mid
    g = ((d < hi) & design).astype(np.float64)
    g = np.where(design, g, 1.0)
    return jnp.asarray(g)


def spoke_gamma(geom, phi: float, nx: int, ny: int, n_lines: int = 3, Lx: float = 1.0, Ly: float = 1.0):
    """Equal-volume radial spokes: n_lines diameters through each tube."""
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    design = np.asarray(geom.design)
    centers = ((0.5 * Lx, 0.22 * Ly), (0.5 * Lx, 0.78 * Ly))

    def mask(w):
        m = np.zeros((nx, ny), dtype=bool)
        for cx, cy in centers:
            dx, dy = X - cx, Y - cy
            for k in range(n_lines):
                ang = np.pi * k / n_lines
                dist = np.abs(dx * np.sin(ang) - dy * np.cos(ang))
                m |= dist < w
        return m & design

    lo, hi = 0.004, 0.25
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        vol = float(np.sum(mask(mid)) / (np.sum(design) + 1e-12))
        if vol < phi:
            lo = mid
        else:
            hi = mid
    g = mask(hi).astype(np.float64)
    g = np.where(design, g, 1.0)
    return jnp.asarray(g)


def fo_to_level(hist, dt: float, level: float, rising: bool) -> dict:
    f = np.asarray(hist)
    hit = np.where(f >= level)[0] if rising else np.where(f <= level)[0]
    if hit.size == 0:
        return {"Fo": None, "reached": False, "f_end": float(f[-1]), "Fo_end": float(len(f) * dt)}
    i = int(hit[0])
    return {"Fo": float((i + 1) * dt), "reached": True, "f_end": float(f[i]), "Fo_end": float((i + 1) * dt)}


def time_metrics(gamma, geom, sim_c, sim_d) -> dict:
    Tf_hist = simulate_liquid_trace(
        jnp.full((sim_c.nx, sim_c.ny), 1.0), gamma, geom, sim_c, tube_bcs(charge_on=True, discharge_on=False)
    )
    Tm_hist = simulate_liquid_trace(
        jnp.full((sim_d.nx, sim_d.ny), -1.0), gamma, geom, sim_d, tube_bcs(charge_on=False, discharge_on=True)
    )
    freeze = fo_to_level(Tf_hist, sim_c.dt, 0.05, rising=False)
    melt = fo_to_level(Tm_hist, sim_d.dt, 0.95, rising=True)
    freeze90 = fo_to_level(Tf_hist, sim_c.dt, 0.10, rising=False)
    melt90 = fo_to_level(Tm_hist, sim_d.dt, 0.90, rising=True)
    return {
        "Fo_freeze_95": freeze["Fo"],
        "freeze_reached": freeze["reached"],
        "liquid_freeze_end": freeze["f_end"],
        "Fo_melt_95": melt["Fo"],
        "melt_reached": melt["reached"],
        "liquid_melt_end": melt["f_end"],
        "Fo_freeze_90": freeze90["Fo"],
        "freeze90_reached": freeze90["reached"],
        "Fo_melt_90": melt90["Fo"],
        "melt90_reached": melt90["reached"],
    }


def _eval_design(gamma, geom, sim_c, sim_d, sim_c_long, sim_d_long) -> dict:
    m = cross_eval(gamma, geom, sim_c, sim_d)
    m["J_cycle"] = cycle_j(m)
    m["volume"] = float(volume_fraction(gamma, geom))
    t = time_metrics(gamma, geom, sim_c_long, sim_d_long)
    m.update(t)
    m["gamma"] = np.asarray(gamma).tolist()
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=36)
    p.add_argument("--phi", type=float, default=0.1)
    p.add_argument("--fo-d", type=float, default=1.2)
    p.add_argument("--fo-horizon", type=float, default=6.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.nx = min(args.nx, 24)
        args.ny = min(args.ny, 24)
        args.n_iter = min(args.n_iter, 12)
        args.fo_d = 0.6
        args.fo_horizon = 1.2

    corners = [
        {"Lambda": 1.0 / 3.0, "Ra_d": 0.0},
        {"Lambda": 3.0, "Ra_d": 3.0e5},
    ]
    geom = two_tube_geom(args.nx, args.ny)
    cells = []
    for c in corners:
        fo_d = args.fo_d
        fo_c = c["Lambda"] * fo_d
        sim_c, sim_d, fo_c_act, fo_d_act = make_sims(args.nx, args.ny, fo_c, fo_d, c["Ra_d"])
        sim_cl, sim_dl, _, _ = make_sims(args.nx, args.ny, args.fo_horizon, args.fo_horizon, c["Ra_d"])
        print(f"\n=== baseline Λ={c['Lambda']:.3g} Ra_d={c['Ra_d']:g} ===", flush=True)
        g_ann = annular_gamma(geom, args.phi, args.nx, args.ny)
        ann = _eval_design(g_ann, geom, sim_c, sim_d, sim_cl, sim_dl)
        g_spk = spoke_gamma(geom, args.phi, args.nx, args.ny)
        spk = _eval_design(g_spk, geom, sim_c, sim_d, sim_cl, sim_dl)
        gamma_to, hist, extra = optimize_architecture(
            "cycle", geom, sim_c, sim_d, phi=args.phi, n_iter=args.n_iter, seed=0, betas=BETAS
        )
        opt = _eval_design(gamma_to, geom, sim_c, sim_d, sim_cl, sim_dl)
        opt["loss_hist"] = hist
        row = {
            "Lambda": fo_c_act / fo_d_act,
            "Lambda_target": c["Lambda"],
            "Ra_d": c["Ra_d"],
            "Fo_c": fo_c_act,
            "Fo_d": fo_d_act,
            "Fo_horizon": args.fo_horizon,
            "annular": {k: v for k, v in ann.items() if k != "gamma"},
            "spoke": {k: v for k, v in spk.items() if k != "gamma"},
            "optimized": {k: v for k, v in opt.items() if k != "gamma"},
            "annular_gamma": ann["gamma"],
            "spoke_gamma": spk["gamma"],
            "optimized_gamma": opt["gamma"],
            "dJ_opt_minus_annular": opt["J_cycle"] - ann["J_cycle"],
            "dJ_opt_minus_spoke": opt["J_cycle"] - spk["J_cycle"],
            "vol_annular": ann["volume"],
            "vol_spoke": spk["volume"],
            "vol_optimized": opt["volume"],
        }
        if "gamma_c" in extra:
            row["dual"] = True
        cells.append(row)
        print(
            f"  annular J={ann['J_cycle']:.4f} vol={ann['volume']:.3f} "
            f"Fo95,fr={ann['Fo_freeze_95']} Fo95,m={ann['Fo_melt_95']} "
            f"Fo90,fr={ann['Fo_freeze_90']}"
        )
        print(
            f"  spoke   J={spk['J_cycle']:.4f} vol={spk['volume']:.3f} "
            f"Fo95,fr={spk['Fo_freeze_95']} Fo95,m={spk['Fo_melt_95']} "
            f"Fo90,fr={spk['Fo_freeze_90']}"
        )
        print(
            f"  cycle   J={opt['J_cycle']:.4f} vol={opt['volume']:.3f} "
            f"Fo95,fr={opt['Fo_freeze_95']} Fo95,m={opt['Fo_melt_95']} "
            f"Fo90,fr={opt['Fo_freeze_90']}"
        )

    Path("results").mkdir(exist_ok=True)
    report = {
        "nx": args.nx,
        "ny": args.ny,
        "n_iter": args.n_iter,
        "phi": args.phi,
        "Fo_d": args.fo_d,
        "Fo_horizon": args.fo_horizon,
        "cells": cells,
        "note": "Equal-volume annular sheaths and radial spokes vs cycle TO at two (Λ, Ra_d) corners.",
    }
    Path("results/baseline.json").write_text(json.dumps(report, indent=2))
    print("wrote results/baseline.json")


if __name__ == "__main__":
    main()
