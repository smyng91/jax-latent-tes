"""Forward-solver checks. Writes results/validate.json with measured numbers only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pcm.physics import (
    default_sim,
    empty_geom,
    enthalpy,
    liquid_fraction,
    simulate,
    simulate_energy,
    slab_bcs,
    step,
    tube_bcs,
    two_tube_geom,
)
from pcm.published import GALLIUM, GAU_VISKANTA_1986, neumann_lambda, stefan_lambda

jax.config.update("jax_enable_x64", True)

# Mushy half-width and subcooling for slab tests. T0 = -4ε so f_ℓ(T0) ≈ 0.
# Fo and mesh are set so s ≫ ε and the lagged-C Jacobi actually converges.
NEUMANN_EPS = 0.01
NEUMANN_T0 = -0.04
NEUMANN_NX = 192
NEUMANN_NY = 8
NEUMANN_JACOBI = 240
NEUMANN_DT = 0.001
NEUMANN_FO_HEAD = 0.40
NEUMANN_FO = (0.32, 0.40)


def interface_x(T, eps, Lx, nx, j=None) -> float:
    """x where liquid fraction crosses 0.5. Mean in y, or a single column j."""
    f = np.asarray(liquid_fraction(T, eps))
    x = (np.arange(nx) + 0.5) * Lx / nx
    js = range(f.shape[1]) if j is None else [int(j)]
    xs = []
    for jj in js:
        col = f[:, jj]
        for i in range(nx - 1):
            if col[i] >= 0.5 and col[i + 1] < 0.5:
                t = (col[i] - 0.5) / (col[i] - col[i + 1] + 1e-12)
                xs.append(x[i] + t * (x[i + 1] - x[i]))
                break
            if col[i] <= 0.5 and col[i + 1] > 0.5:
                t = (0.5 - col[i]) / (col[i + 1] - col[i] + 1e-12)
                xs.append(x[i] + t * (x[i + 1] - x[i]))
                break
    return float(np.mean(xs)) if xs else float("nan")


def _neumann_case(ste: float, fo: float, *, eps: float = NEUMANN_EPS, t0: float = NEUMANN_T0) -> dict:
    n_steps = int(round(fo / NEUMANN_DT))
    sim = default_sim(
        NEUMANN_NX,
        NEUMANN_NY,
        dt=NEUMANN_DT,
        n_steps=n_steps,
        ste=ste,
        kappa=1.0,
        flow=False,
        n_jacobi_t=NEUMANN_JACOBI,
        eps=eps,
    )
    T, _, _ = simulate(
        jnp.full((NEUMANN_NX, NEUMANN_NY), t0),
        jnp.zeros((NEUMANN_NX, NEUMANN_NY)),
        empty_geom(NEUMANN_NX, NEUMANN_NY),
        sim,
        slab_bcs(T_left=1.0),
    )
    ste_s = ste * abs(t0)
    lam = neumann_lambda(ste, ste_s)
    lam1 = stefan_lambda(ste)
    s_exact = 2.0 * lam * math.sqrt(fo)
    s_one = 2.0 * lam1 * math.sqrt(fo)
    s_num = interface_x(T, sim.eps, sim.Lx, NEUMANN_NX)
    return {
        "ste": ste,
        "fo": fo,
        "eps": eps,
        "theta_init": t0,
        "ste_s": ste_s,
        "n_steps": n_steps,
        "nx": NEUMANN_NX,
        "ny": NEUMANN_NY,
        "lambda_two_phase": lam,
        "lambda_one_phase": lam1,
        "s_exact": s_exact,
        "s_one_phase": s_one,
        "s_numeric": s_num,
        "relative_error": abs(s_num - s_exact) / s_exact if s_exact else float("nan"),
        "relative_error_vs_one_phase": abs(s_num - s_one) / s_one if s_one else float("nan"),
        "mean_liquid_fraction": float(np.mean(np.asarray(liquid_fraction(T, sim.eps)))),
    }


def check_stefan() -> dict:
    row = _neumann_case(0.2, NEUMANN_FO_HEAD)
    row["note"] = (
        "Two-phase Neumann (Carslaw & Jaeger) with Ste_s = Ste |θ_init|; "
        "tanh mushy zone ε=0.01, θ_init=-4ε, Fo large enough that s ≫ ε."
    )
    return row


def check_neumann_series() -> dict:
    rows = [_neumann_case(ste, fo) for ste in (0.05, 0.1, 0.2) for fo in NEUMANN_FO]
    return {
        "source": "Carslaw & Jaeger (1959), two-phase Neumann, equal diffusivities",
        "eps": NEUMANN_EPS,
        "theta_init": NEUMANN_T0,
        "cases": rows,
        "max_relative_error": max(r["relative_error"] for r in rows),
    }


def check_energy() -> dict:
    """Steady linear profile; transient slab; tube freeze ΔH vs ∫q dt."""
    nx, ny = 48, 16
    dt = 0.002
    sim = default_sim(nx, ny, dt=dt, n_steps=20, ste=1.0e6, kappa=1.0, flow=False, n_jacobi_t=80)
    geom = empty_geom(nx, ny)
    gamma = jnp.zeros((nx, ny))
    dx = sim.Lx / nx
    dy = sim.Ly / ny
    x = (jnp.arange(nx) + 0.5) * dx
    T_lin = jnp.broadcast_to(1.0 - x[:, None], (nx, ny))
    bcs_lr = slab_bcs(T_left=1.0, T_right=0.0)
    T = T_lin
    for _ in range(20):
        z = jnp.zeros_like(T)
        T, _, _, _ = step(T, z, z, z, gamma, geom, sim, bcs_lr)
    rms = float(jnp.sqrt(jnp.mean((T - T_lin) ** 2)))
    H0 = float(jnp.sum(enthalpy(T_lin, gamma, sim)) * dx * dy)
    H1 = float(jnp.sum(enthalpy(T, gamma, sim)) * dx * dy)

    sim_t = default_sim(nx, ny, dt=dt, n_steps=40, ste=1.0e6, kappa=1.0, flow=False, n_jacobi_t=80)
    Tt = jnp.zeros((nx, ny))
    bcs_l = slab_bcs(T_left=1.0)
    Ht0 = float(jnp.sum(enthalpy(Tt, gamma, sim_t)) * dx * dy)
    Tt, _, _ = simulate(Tt, gamma, geom, sim_t, bcs_l)
    Ht1 = float(jnp.sum(enthalpy(Tt, gamma, sim_t)) * dx * dy)

    nxt, nyt = 32, 32
    fo_tube = 0.25
    dt_t = 0.005
    n_tube = int(round(fo_tube / dt_t))
    sim_tube = default_sim(nxt, nyt, dt=dt_t, n_steps=n_tube, ste=0.1, kappa=1.0, flow=False, n_jacobi_t=64)
    geom_t = two_tube_geom(nxt, nyt)
    gamma_t = jnp.where(geom_t.design, 0.0, 1.0)
    T0 = jnp.full((nxt, nyt), 1.0)
    bcs_tube = tube_bcs(charge_on=True, discharge_on=False)
    _, dH, acc = simulate_energy(T0, gamma_t, geom_t, sim_tube, bcs_tube)
    dH, acc = float(dH), float(acc)
    tube_rel = abs(dH - acc) / (abs(dH) + 1e-12)
    return {
        "steady_rms_drift": rms,
        "steady_dH": H1 - H0,
        "transient_dH": Ht1 - Ht0,
        "transient_heat_enters": bool(Ht1 > Ht0),
        "tube_freeze": {
            "Fo": n_tube * dt_t,
            "dH": dH,
            "int_q_dt": acc,
            "relative_error": tube_rel,
        },
        "note": "Steady: T=1-x. Transient: left hot, right insulated, Ste huge. Tube: freeze window ΔH vs wall+tube flux.",
    }


def check_cavity_ra() -> dict:
    """Same Fo, melt fraction at Ra=0 vs Ra=1e5. Trend test, not a literature benchmark."""
    nx, ny = 32, 32
    dt = 0.005
    n_steps = 120
    ste = 0.1
    fo = n_steps * dt
    out = {}
    for ra, flow, key in [(0.0, False, "ra_0"), (1.0e5, True, "ra_1e5")]:
        sim = default_sim(
            nx,
            ny,
            dt=dt,
            n_steps=n_steps,
            ste=ste,
            ra=ra,
            kappa=1.0,
            flow=flow,
            n_jacobi_t=32,
            n_jacobi_p=32,
        )
        T, u, v = simulate(
            jnp.full((nx, ny), -0.8), jnp.zeros((nx, ny)), empty_geom(nx, ny), sim, slab_bcs(T_left=1.0)
        )
        f = float(np.mean(np.asarray(liquid_fraction(T, sim.eps))))
        out[key] = {
            "mean_liquid_fraction": f,
            "u_max": float(np.max(np.abs(np.asarray(u)))),
            "v_max": float(np.max(np.abs(np.asarray(v)))),
            "fo": fo,
            "flow": flow,
            "Ra": ra,
        }
    f0 = out["ra_0"]["mean_liquid_fraction"]
    f1 = out["ra_1e5"]["mean_liquid_fraction"]
    out["melt_fraction_ratio_Ra1e5_over_Ra0"] = f1 / (f0 + 1e-12)
    out["note"] = "Vorticity–stream-function NS, implicit-upwind energy, 32x32, Fo=0.6."
    return out


def check_cavity_mesh() -> dict:
    dt = 0.005
    n_steps = 120
    rows = []
    for nx in (32, 48):
        out = {}
        for ra, flow, key in [(0.0, False, "ra_0"), (1.0e5, True, "ra_1e5")]:
            sim = default_sim(
                nx, nx, dt=dt, n_steps=n_steps, ste=0.1, ra=ra, kappa=1.0,
                flow=flow, n_jacobi_t=40, n_jacobi_p=40,
            )
            T, u, v = simulate(
                jnp.full((nx, nx), -0.8), jnp.zeros((nx, nx)), empty_geom(nx, nx), sim, slab_bcs(T_left=1.0)
            )
            out[key] = {
                "mean_liquid_fraction": float(np.mean(np.asarray(liquid_fraction(T, sim.eps)))),
                "u_max": float(np.max(np.abs(np.asarray(u)))),
                "v_max": float(np.max(np.abs(np.asarray(v)))),
            }
        r = out["ra_1e5"]["mean_liquid_fraction"] / (out["ra_0"]["mean_liquid_fraction"] + 1e-12)
        rows.append({"nx": nx, "ny": nx, "ratio": r, **out})
    r0, r1 = rows[0]["ratio"], rows[1]["ratio"]
    return {
        "Fo": n_steps * dt,
        "cases": rows,
        "ratio_rel_diff": abs(r1 - r0) / (0.5 * (r0 + r1) + 1e-12),
        "note": "Side-heated cavity melt-fraction ratio Ra=1e5 over Ra=0 at 32 and 48.",
    }


def check_cfl_mush() -> dict:
    """Energy CFL and mush constant: melt fraction at Fo=0.6, Ra=1e5."""
    nx = 32
    dt = 0.005
    n_steps = 120
    T0 = jnp.full((nx, nx), -0.8)
    geom = empty_geom(nx, nx)
    gamma = jnp.zeros((nx, nx))
    bcs = slab_bcs(T_left=1.0)

    def melt_frac(energy_cfl: float, mush: float) -> float:
        sim = default_sim(
            nx, nx, dt=dt, n_steps=n_steps, ste=0.1, ra=1.0e5, kappa=1.0,
            flow=True, n_jacobi_t=32, n_jacobi_p=32, energy_cfl=energy_cfl, mush=mush,
        )
        T, _, _ = simulate(T0, gamma, geom, sim, bcs)
        return float(np.mean(np.asarray(liquid_fraction(T, sim.eps))))

    cfl_rows = [{"energy_cfl": c, "mean_liquid_fraction": melt_frac(c, 1.0e5)} for c in (4.0, 8.0, 32.0)]
    mush_rows = [{"mush": m, "mean_liquid_fraction": melt_frac(8.0, m)} for m in (1.0e4, 1.0e5, 1.0e6)]
    f_cfl = [r["mean_liquid_fraction"] for r in cfl_rows]
    f_mush = [r["mean_liquid_fraction"] for r in mush_rows]
    cfl_span = (max(f_cfl) - min(f_cfl)) / (0.5 * (max(f_cfl) + min(f_cfl)) + 1e-12)
    mush_span = (max(f_mush) - min(f_mush)) / (0.5 * (max(f_mush) + min(f_mush)) + 1e-12)
    return {
        "Fo": n_steps * dt,
        "Ra": 1.0e5,
        "energy_cfl": cfl_rows,
        "mush": mush_rows,
        "cfl_relative_span": cfl_span,
        "mush_relative_span": mush_span,
        "note": "Detached energy CFL vs Kozeny–Carman mush at fixed mesh/Fo.",
    }


def _gallium_groups() -> dict:
    g = GALLIUM
    dT = g["T_hot"] - g["T_m"]
    alpha = g["k"] / (g["rho"] * g["cp"])
    nu = g["mu"] / g["rho"]
    ste_from_props = g["cp"] * dT / g["L_lat"]
    ra_from_props = 9.81 * g["beta"] * dT * g["H"] ** 3 / (nu * alpha)
    theta_init = (g["T_init"] - g["T_m"]) / dT
    return {
        "alpha": alpha,
        "t_ref_H": g["H"] ** 2 / alpha,
        "aspect_W_over_H": g["W"] / g["H"],
        "ste_from_props": ste_from_props,
        "ra_from_props": ra_from_props,
        "theta_init": theta_init,
        "dT": dT,
    }


def check_gallium() -> dict:
    """Gau–Viskanta / Brent gallium cavity. Published properties; no digitized fronts."""
    g = GALLIUM
    meta = _gallium_groups()
    Lx = meta["aspect_W_over_H"]
    nx, ny = 64, 48
    dt = 0.0015
    ste = g["Ste"]
    t0 = float(meta["theta_init"])
    bcs = slab_bcs(T_left=1.0, T_right=t0)
    T0 = jnp.full((nx, ny), t0)
    geom = empty_geom(nx, ny)
    gamma = jnp.zeros((nx, ny))

    def run(t_min: float, flow: bool, ra: float, energy_cfl: float) -> dict:
        fo = (t_min * 60.0) / meta["t_ref_H"]
        n_steps = max(2, int(round(fo / dt)))
        sim = default_sim(
            nx, ny, dt=dt, n_steps=n_steps, ste=ste, ra=ra, pr=g["Pr"], kappa=1.0,
            flow=flow, n_jacobi_t=48, n_jacobi_p=56, Lx=Lx, Ly=1.0, energy_cfl=energy_cfl,
        )
        T, u, v = simulate(T0, gamma, geom, sim, bcs)
        f = np.asarray(liquid_fraction(T, sim.eps))
        s_mean = interface_x(T, sim.eps, Lx, nx)
        s_mid = interface_x(T, sim.eps, Lx, nx, j=ny // 2)
        return {
            "t_min": t_min,
            "Fo_H": n_steps * dt,
            "n_steps": n_steps,
            "flow": flow,
            "Ra": ra,
            "energy_cfl": energy_cfl,
            "mean_liquid_fraction": float(f.mean()),
            "u_max": float(np.max(np.abs(np.asarray(u)))),
            "v_max": float(np.max(np.abs(np.asarray(v)))),
            "interface_x_over_W": s_mean / Lx,
            "interface_mid_x_over_W": s_mid / Lx,
        }

    t_early = GAU_VISKANTA_1986["conduction_window_min"]
    t_late = GAU_VISKANTA_1986["late_time_min"]
    cond_early = run(t_early, False, 0.0, 32.0)
    cond_late = run(t_late, False, 0.0, 32.0)
    flow_late = run(t_late, True, g["Ra"], 8.0)

    fo_W_early = (t_early * 60.0) * meta["alpha"] / g["W"] ** 2
    ste_s = ste * abs(t0)
    lam = neumann_lambda(ste, ste_s)
    s_over_W = 2.0 * lam * math.sqrt(fo_W_early)
    enhancement = flow_late["mean_liquid_fraction"] / (cond_late["mean_liquid_fraction"] + 1e-12)
    return {
        "source_properties": "Brent, Voller & Reid, Numer. Heat Transfer 13, 297–318 (1988), Table 1",
        "source_experiment": "Gau & Viskanta, J. Heat Transfer 108, 174–181 (1986)",
        "groups": meta,
        "brent_Ste": g["Ste"],
        "brent_Ra": g["Ra"],
        "brent_Pr": g["Pr"],
        "neumann_at_2min": {
            "Fo_W": fo_W_early,
            "lambda_two_phase": lam,
            "ste_s": ste_s,
            "s_over_W_exact": s_over_W,
            "note": "Two-phase Neumann using cavity width W, experimental subcooling, Ste=0.039.",
        },
        "cavity_conduction_2min": cond_early,
        "cavity_conduction_17min": cond_late,
        "cavity_ns_17min": flow_late,
        "cavity_darcy_17min": flow_late,
        "convection_volume_ratio_17min": enhancement,
        "gau_viskanta_late_volume_over_neumann": GAU_VISKANTA_1986["late_time_volume_over_neumann"],
        "note": (
            "NS at energy CFL 8 (vorticity CFL 1). Late-time volume ratio vs Gau–Viskanta ~1.75, "
            "not forced to match. Mid-cavity interface is reported alongside the y-mean."
        ),
    }


def check_flow_adjoint() -> dict:
    """Melt-loss reverse mode must stay finite with live NS adjoint."""
    from pcm.optimize import loss_melt, seed_psi

    nx = 16
    geom = two_tube_geom(nx, nx)
    sim = default_sim(
        nx, nx, dt=0.005, n_steps=16, ste=0.1, ra=1.0e5, kappa=20.0,
        flow=True, n_jacobi_t=16, n_jacobi_p=32,
    )
    psi = seed_psi(geom, 0.1, 0)

    def melt_grad(n_steps: int):
        s = sim._replace(n_steps=n_steps)
        return jax.grad(lambda p: loss_melt(p, geom, s, 0.1, 2.0))(psi)

    g1 = melt_grad(1)
    g = melt_grad(16)
    n1 = float(jnp.linalg.norm(g1))
    n16 = float(jnp.linalg.norm(g))
    return {
        "grad_norm": n16,
        "grad_norm_n1": n1,
        "grad_norm_n16_over_n1": n16 / (n1 + 1e-12),
        "has_nan": bool(jnp.isnan(g).any() | jnp.isnan(g1).any()),
        "has_inf": bool(jnp.isinf(g).any() | jnp.isinf(g1).any()),
        "note": "Live NS adjoint: implicit VJP through Helmholtz/Poisson, detached CFL scale.",
    }


def main() -> None:
    report = {
        "stefan": check_stefan(),
        "neumann_series": check_neumann_series(),
        "energy": check_energy(),
        "cavity": check_cavity_ra(),
        "cavity_mesh": check_cavity_mesh(),
        "cfl_mush": check_cfl_mush(),
        "gallium_gau_viskanta": check_gallium(),
    }
    adj = check_flow_adjoint()
    report["flow_adjoint"] = adj
    Path("results").mkdir(exist_ok=True)
    path = Path("results/validate.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")
    ste = report["stefan"]
    if not (ste["s_numeric"] == ste["s_numeric"]):
        raise SystemExit("stefan interface is NaN")
    if ste["relative_error"] > 0.05:
        raise SystemExit(f"stefan relative error too large: {ste['relative_error']}")
    neu = report["neumann_series"]
    if neu["max_relative_error"] > 0.05:
        raise SystemExit(f"neumann series max relative error {neu['max_relative_error']}")
    if report["energy"]["steady_rms_drift"] > 0.05:
        raise SystemExit(f"steady profile drifted: {report['energy']['steady_rms_drift']}")
    if not report["energy"]["transient_heat_enters"]:
        raise SystemExit("transient slab did not heat")
    if report["energy"]["tube_freeze"]["relative_error"] > 0.08:
        raise SystemExit(f"tube energy error: {report['energy']['tube_freeze']}")
    cav = report["cavity"]
    if not (cav["ra_1e5"]["mean_liquid_fraction"] == cav["ra_1e5"]["mean_liquid_fraction"]):
        raise SystemExit("cavity Ra=1e5 produced NaN")
    if cav["melt_fraction_ratio_Ra1e5_over_Ra0"] < 1.15:
        raise SystemExit(
            f"buoyancy did not increase melt: ratio={cav['melt_fraction_ratio_Ra1e5_over_Ra0']}"
        )
    mesh = report["cavity_mesh"]
    for case in mesh["cases"]:
        if case["ratio"] < 1.05:
            raise SystemExit(f"mesh cavity ratio too small: {case}")
    if mesh["ratio_rel_diff"] > 0.15:
        raise SystemExit(f"cavity mesh ratios disagree: {mesh}")
    print(
        "CFL relative span",
        report["cfl_mush"]["cfl_relative_span"],
        "(production energy CFL is 8; 32 over-melts)",
    )
    gal = report["gallium_gau_viskanta"]
    s_ex = gal["neumann_at_2min"]["s_over_W_exact"]
    s_nu = gal["cavity_conduction_2min"]["interface_x_over_W"]
    if abs(s_nu - s_ex) / s_ex > 0.08:
        raise SystemExit(f"gallium 2 min interface error: {s_nu} vs {s_ex}")
    if gal["convection_volume_ratio_17min"] < 1.4:
        raise SystemExit(f"gallium convection too weak: {gal['convection_volume_ratio_17min']}")
    if gal["convection_volume_ratio_17min"] != gal["convection_volume_ratio_17min"]:
        raise SystemExit("gallium NS produced NaN")
    if (
        adj["has_nan"]
        or adj["has_inf"]
        or adj["grad_norm"] <= 0.0
        or adj["grad_norm"] > 1e4
        or adj["grad_norm_n16_over_n1"] > 100.0
    ):
        raise SystemExit(f"flow adjoint failed: {adj}")


if __name__ == "__main__":
    main()
