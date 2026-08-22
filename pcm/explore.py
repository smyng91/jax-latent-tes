"""Explore dual-cycle novelties. Writes results/explore.json and paper/figures/explore_*.png."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from pcm.design import volume_fraction
from pcm.optimize import cross_eval, optimize_architecture, pcm_liquid, run_cycle_switched
from pcm.physics import two_tube_geom
from pcm.report import plot_explore
from pcm.sweep import cycle_j, make_sims

jax.config.update("jax_enable_x64", True)


def _metrics(gamma, geom, sim_c, sim_d, extra=None):
    m = cross_eval(gamma, geom, sim_c, sim_d)
    m["J_cycle"] = cycle_j(m)
    if extra and extra.get("mode") == "switched":
        gc, gd = extra["gamma_c"], extra["gamma_d"]
        Tf, Tm = run_cycle_switched(gc, gd, geom, sim_c, sim_d)
        m["J_switched"] = float(
            pcm_liquid(Tf, gc, geom, sim_c.eps) + (1.0 - pcm_liquid(Tm, gd, geom, sim_d.eps))
        )
        m["vol_c"] = float(volume_fraction(gc, geom))
        m["vol_d"] = float(volume_fraction(gd, geom))
    return m


def _fit(name, geom, sim_c, sim_d, n_iter, phi=0.1, **kw):
    gamma, hist, extra = optimize_architecture(
        name, geom, sim_c, sim_d, phi=phi, n_iter=n_iter, **kw
    )
    m = _metrics(gamma, geom, sim_c, sim_d, extra)
    m["loss_final"] = hist[-1]
    m["name"] = name
    return np.asarray(gamma), m, extra


def main() -> None:
    nx = ny = 24
    n_iter = 20
    fo_d = 0.6
    dt = 0.005
    geom = two_tube_geom(nx, ny)
    sim_c, sim_d, fo_c, fo_d_act = make_sims(nx, ny, fo_d, fo_d, 1.0e5, dt=dt)
    out: dict = {
        "nx": nx,
        "n_iter": n_iter,
        "Fo_c": fo_c,
        "Fo_d": fo_d_act,
        "Ra_d": 1.0e5,
        "note": "Conflict cell Λ=1, Ra_d=1e5 unless a block says otherwise.",
    }

    print("\n=== dual: valved vs switched vs extra metal ===")
    dual = {}
    for name, phi in (("cycle", 0.1), ("dual", 0.1), ("dual", 0.2), ("dual_switched", 0.1), ("dual_switched", 0.2)):
        key = f"{name}_phi{phi}"
        _, m, _ = _fit(name, geom, sim_c, sim_d, n_iter, phi=phi)
        dual[key] = {k: m[k] for k in m if k != "name"}
        print(key, "J_cycle", m["J_cycle"], "J_sw", m.get("J_switched"), "vol", m["volume"])
    out["dual"] = dual

    print("\n=== Pareto freeze/melt weights ===")
    pareto = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, m, _ = _fit("pareto", geom, sim_c, sim_d, n_iter, w_fr=w)
        row = {
            "w_fr": w,
            "J_cycle": m["J_cycle"],
            "metal_top": m["metal_top"],
            "metal_bottom": m["metal_bottom"],
            "liquid_after_freeze": m["liquid_after_freeze"],
            "liquid_after_melt": m["liquid_after_melt"],
            "volume": m["volume"],
        }
        pareto.append(row)
        print("w", w, row)
    out["pareto"] = pareto

    print("\n=== humidity-weighted discharge ===")
    humid = []
    for h in (0.0, 0.5, 1.0):
        g, m, _ = _fit("humid", geom, sim_c, sim_d, n_iter, humidity=h)
        row = {
            "h": h,
            "J_cycle": m["J_cycle"],
            "metal_top": m["metal_top"],
            "metal_bottom": m["metal_bottom"],
            "volume": m["volume"],
        }
        humid.append(row)
        print("h", h, row)
        _ = g
    out["humidity"] = humid

    print("\n=== Ste_c / COP lift (Ste_d fixed) ===")
    cop = []
    for ste_c, cop_w in ((0.05, 2.0), (0.1, 2.0), (0.2, 2.0), (0.1, 0.0)):
        sc, sd, _, _ = make_sims(nx, ny, fo_d, fo_d, 1.0e5, dt=dt, ste_c=ste_c, ste_d=0.1)
        _, m, _ = _fit("cop", geom, sc, sd, n_iter, cop_w=cop_w)
        lam = (fo_d * ste_c) / (fo_d * 0.1)
        row = {
            "Ste_c": ste_c,
            "Ste_d": 0.1,
            "Lambda": lam,
            "cop_w": cop_w,
            "J_cycle": m["J_cycle"],
            "metal_top": m["metal_top"],
            "metal_bottom": m["metal_bottom"],
            "liquid_after_freeze": m["liquid_after_freeze"],
            "volume": m["volume"],
        }
        cop.append(row)
        print(row)
    out["cop"] = cop

    print("\n=== schedule co-design: Fo_c at fixed Fo_d ===")
    sched = []
    for fo_c in (0.2, 0.4, 0.8):
        sc, sd, fo_c_act, fo_d_act = make_sims(nx, ny, fo_c, fo_d, 1.0e5, dt=dt)
        _, m, _ = _fit("cycle", geom, sc, sd, n_iter)
        row = {
            "Fo_c": fo_c_act,
            "Fo_d": fo_d_act,
            "Lambda": fo_c_act / fo_d_act,
            "J_cycle": m["J_cycle"],
            "liquid_after_freeze": m["liquid_after_freeze"],
            "liquid_after_melt": m["liquid_after_melt"],
            "metal_top": m["metal_top"],
            "metal_bottom": m["metal_bottom"],
        }
        sched.append(row)
        print(row)
    out["schedule"] = sched

    print("\n=== transfer: melt vs freeze designs on sequential cycle ===")
    _, m_m, _ = _fit("melt", geom, sim_c, sim_d, n_iter)
    _, m_f, _ = _fit("freeze", geom, sim_c, sim_d, n_iter)
    transfer = {
        "melt_on_cycle": m_m["J_cycle"],
        "freeze_on_cycle": m_f["J_cycle"],
        "melt_metal_top": m_m["metal_top"],
        "melt_metal_bottom": m_m["metal_bottom"],
        "freeze_metal_top": m_f["metal_top"],
        "freeze_metal_bottom": m_f["metal_bottom"],
        "cross_gap": abs(m_m["J_cycle"] - m_f["J_cycle"]),
        "layout_gap_top": m_m["metal_top"] - m_f["metal_top"],
    }
    out["transfer"] = transfer
    print(transfer)

    figdir = Path("paper/figures")
    plot_explore(out, figdir)
    Path("results").mkdir(exist_ok=True)
    Path("results/explore.json").write_text(json.dumps(out, indent=2, default=float))
    print("wrote results/explore.json and paper/figures/explore_*.png")


if __name__ == "__main__":
    main()
