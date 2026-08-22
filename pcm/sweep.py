"""Coarse (Λ, Ra_d) architecture sweep. Writes results/sweep.json from this run only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pcm.classify import dual_admitted, pick_winner
from pcm.design import volume_fraction
from pcm.optimize import BETAS, cross_eval, optimize_architecture
from pcm.physics import default_sim, two_tube_geom
from pcm.report import plot_classes, plot_map

jax.config.update("jax_enable_x64", True)


def make_sims(nx, ny, fo_c, fo_d, ra_d, dt=0.005, ste_c=0.1, ste_d=0.1, energy_cfl=8.0):
    n_c = max(2, int(round(fo_c / dt)))
    n_d = max(2, int(round(fo_d / dt)))
    sim_c = default_sim(
        nx, ny, dt=dt, n_steps=n_c, ste=ste_c, ra=0.0, kappa=20.0, flow=False, n_jacobi_t=32
    )
    sim_d = default_sim(
        nx,
        ny,
        dt=dt,
        n_steps=n_d,
        ste=ste_d,
        ra=ra_d,
        kappa=20.0,
        flow=ra_d > 0,
        n_jacobi_t=32,
        n_jacobi_p=64,
        energy_cfl=energy_cfl,
    )
    return sim_c, sim_d, n_c * dt, n_d * dt


def cycle_j(metrics: dict) -> float:
    return metrics["cycle_liquid_after_freeze"] + (1.0 - metrics["cycle_liquid_after_melt"])


def _slim(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "gamma" and not str(k).startswith("gamma")}


def run_cell(geom, sim_c, sim_d, names, phi, n_iter, seeds, match_volume) -> dict:
    seed_arch = []
    seed_extras = []
    for seed in seeds:
        arch = {}
        extras_keep = {}
        psi_warm = {}
        for name in names:
            warm = None
            if name == "dual" and "freeze" in psi_warm and "melt" in psi_warm:
                warm = {"psi_c": psi_warm["freeze"], "psi_d": psi_warm["melt"]}
            gamma, hist, extras = optimize_architecture(
                name,
                geom,
                sim_c,
                sim_d,
                phi=phi,
                n_iter=n_iter,
                seed=seed,
                match_volume=match_volume,
                betas=BETAS,
                warm=warm,
            )
            if "psi" in extras:
                psi_warm[name] = extras["psi"]
            metrics = cross_eval(gamma, geom, sim_c, sim_d)
            metrics["J_cycle"] = cycle_j(metrics)
            metrics["loss_hist"] = hist
            metrics["gamma"] = np.asarray(gamma).tolist()
            if "gamma_c" in extras:
                gc, gd = extras["gamma_c"], extras["gamma_d"]
                metrics["vol_c"] = float(volume_fraction(gc, geom))
                metrics["vol_d"] = float(volume_fraction(gd, geom))
                metrics["overlap"] = float(volume_fraction(gc * gd, geom))
                extras_keep[name] = {
                    "gamma_c": np.asarray(gc).tolist(),
                    "gamma_d": np.asarray(gd).tolist(),
                    "vol_c": metrics["vol_c"],
                    "vol_d": metrics["vol_d"],
                }
            extra = ""
            if "vol_c" in metrics:
                extra = f" vol_c={metrics['vol_c']:.3f} vol_d={metrics['vol_d']:.3f} ov={metrics['overlap']:.3f}"
            print(
                f"  seed {seed} {name}: J_cycle={metrics['J_cycle']:.4f} "
                f"vol={metrics['volume']:.3f} "
                f"metal_top={metrics['metal_top']:.3f} metal_bottom={metrics['metal_bottom']:.3f}{extra}",
                flush=True,
            )
            arch[name] = metrics
        seed_arch.append(arch)
        seed_extras.append(extras_keep)

    mean_j = {n: float(np.mean([a[n]["J_cycle"] for a in seed_arch])) for n in names}
    std_j = {n: float(np.std([a[n]["J_cycle"] for a in seed_arch])) for n in names}
    best_seed = int(np.argmin([min(a[n]["J_cycle"] for n in names) for a in seed_arch]))
    arch = seed_arch[best_seed]
    extras_keep = seed_extras[best_seed]
    winner = pick_winner(mean_j, scatter=std_j)
    winner_gamma_name = winner if winner in arch else min(names, key=lambda n: mean_j[n])
    freeze_incomplete = True
    if "cycle" in arch:
        freeze_incomplete = arch["cycle"]["liquid_after_freeze"] > 0.35
    elif "freeze" in arch:
        freeze_incomplete = arch["freeze"]["liquid_after_freeze"] > 0.35
    dJ = None
    if "dual" in mean_j and "cycle" in mean_j:
        dJ = mean_j["cycle"] - mean_j["dual"]
    dual_split = extras_keep.get("dual", {})
    dual_ok = dual_admitted(
        dual_split.get("vol_c", 0.0),
        dual_split.get("vol_d", 0.0),
        dual_split.get("gamma_c"),
        dual_split.get("gamma_d"),
        geom,
    )
    if winner == "dual" and not dual_ok:
        winner = pick_winner({k: v for k, v in mean_j.items() if k != "dual"}, scatter=std_j)
        winner_gamma_name = winner if winner in arch else min(names, key=lambda n: mean_j[n])
    return {
        "winner": winner,
        "mean_J": mean_j,
        "std_J": std_j,
        "n_seeds": len(seeds),
        "best_seed": seeds[best_seed],
        "melt_only_freeze_fail": freeze_incomplete,
        "dual_minus_cycle_J": dJ,
        "dual_two_networks": dual_ok,
        "dual_admitted": dual_ok,
        "architectures": {n: _slim(arch[n]) | {"J_cycle_mean": mean_j[n], "J_cycle_std": std_j[n]} for n in names},
        "winner_gamma": arch[winner_gamma_name]["gamma"],
        "dual_fields": dual_split,
        **({} if winner_gamma_name not in extras_keep else {"winner_dual": extras_keep[winner_gamma_name]}),
    }


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=28)
    p.add_argument("--phi", type=float, default=0.1)
    p.add_argument("--fo-d", type=float, default=0.6)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--energy-cfl", type=float, default=8.0)
    p.add_argument("--ijhmt", action="store_true", help="Longer Fo, volume hold, β=2→8, 2 seeds")
    p.add_argument("--grid-study", action="store_true", help="3×3 subset at --nx and --nx-fine")
    p.add_argument("--nx-fine", type=int, default=48)
    p.add_argument("--quick", action="store_true", help="2x2 grid, melt/freeze only, 8 iterations")
    p.add_argument("--spot-check", action="store_true", help="64² cycle vs dual at the two Λ=3 flip cells")
    p.add_argument("--no-match-volume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()
    lambdas = [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0]
    ras = [0.0, 3.0e4, 1.0e5, 3.0e5]
    names = ["melt", "freeze", "cycle", "dual"]
    fo_d = args.fo_d
    n_seeds = args.seeds
    match_volume = not args.no_match_volume
    energy_cfl = args.energy_cfl
    if args.ijhmt:
        fo_d = max(fo_d, 1.2)
        args.n_iter = max(args.n_iter, 36)
        n_seeds = max(n_seeds, 2)
        match_volume = True
        energy_cfl = 8.0
    if args.grid_study:
        lambdas = [1.0 / 3.0, 1.0, 3.0]
        ras = [0.0, 1.0e5, 3.0e5]
        n_seeds = max(n_seeds, 1)
    if args.spot_check:
        lambdas = [3.0]
        ras = [0.0, 3.0e5]
        names = ["melt", "freeze", "cycle", "dual"]
        args.nx = max(args.nx, 64)
        args.ny = args.nx
        args.n_iter = 24
        fo_d = max(fo_d, 1.2)
        n_seeds = 1
        match_volume = True
        energy_cfl = 8.0
    if args.quick:
        lambdas = [1.0, 3.0]
        ras = [0.0, 1.0e5]
        names = ["melt", "freeze"]
        args.n_iter = min(args.n_iter, 8)
        args.nx = min(args.nx, 20)
        args.ny = min(args.ny, 20)
        fo_d = min(fo_d, 0.4)
        n_seeds = 1

    meshes = [args.nx]
    if args.grid_study:
        meshes = [args.nx, args.nx_fine]

    seeds = list(range(n_seeds))
    by_mesh = {}
    for nx in meshes:
        ny = nx if args.grid_study else args.ny
        geom = two_tube_geom(nx, ny)
        cells = []
        for lam in lambdas:
            for ra in ras:
                fo_c = lam * fo_d
                sim_c, sim_d, fo_c_act, fo_d_act = make_sims(
                    nx, ny, fo_c, fo_d, ra, energy_cfl=energy_cfl
                )
                lam_act = fo_c_act / fo_d_act
                print(f"\n=== nx={nx} Λ={lam_act:.3f}  Ra_d={ra:g}  Fo_c={fo_c_act:.4f} Fo_d={fo_d_act:.4f} ===")
                cell = run_cell(geom, sim_c, sim_d, names, args.phi, args.n_iter, seeds, match_volume)
                cell.update(
                    {
                        "Lambda_target": lam,
                        "Lambda": lam_act,
                        "Ra_d": ra,
                        "Fo_c": fo_c_act,
                        "Fo_d": fo_d_act,
                        "nx": nx,
                        "ny": ny,
                    }
                )
                cells.append(cell)
        by_mesh[str(nx)] = cells

    Path("results").mkdir(exist_ok=True)
    cells = by_mesh[str(meshes[0])]
    flips = []
    if args.grid_study and len(meshes) == 2:
        a, b = by_mesh[str(meshes[0])], by_mesh[str(meshes[1])]
        for ca, cb in zip(a, b):
            if ca["winner"] != cb["winner"]:
                flips.append(
                    {
                        "Lambda_target": ca["Lambda_target"],
                        "Ra_d": ca["Ra_d"],
                        "coarse": ca["winner"],
                        "fine": cb["winner"],
                    }
                )
        Path("results/grid_study.json").write_text(
            json.dumps(
                {
                    "meshes": meshes,
                    "n_iter": args.n_iter,
                    "Fo_d": fo_d,
                    "n_seeds": n_seeds,
                    "energy_cfl": energy_cfl,
                    "by_mesh": {k: [{kk: vv for kk, vv in c.items() if kk != "winner_gamma"} for c in v] for k, v in by_mesh.items()},
                    "flips": flips,
                    "n_flips": len(flips),
                },
                indent=2,
            )
        )
        plot_map(b, Path("paper/figures/lambda_ra_map_grid.png"))
        plot_classes(b, Path("paper/figures/lambda_ra_classes_grid.png"))
        print(f"wrote results/grid_study.json with {len(flips)} winner flips")
        return

    report = {
        "nx": cells[0]["nx"],
        "ny": cells[0]["ny"],
        "n_iter": args.n_iter,
        "phi": args.phi,
        "Ste": 0.1,
        "Fo_d": fo_d,
        "kappa": 20.0,
        "energy_cfl": energy_cfl,
        "n_seeds": n_seeds,
        "match_volume": match_volume,
        "betas": list(BETAS),
        "grid_study": args.grid_study,
        "n_flips": len(flips),
        "flips": flips,
        "cells": cells,
        "spot_check": args.spot_check,
        "disclaimer": (
            "Vorticity-stream NS TO, implicit-upwind energy, live implicit-VJP adjoint, "
            f"beta continuation {BETAS[0]:g} to {BETAS[-1]:g}, volume shift, dual warm-start from melt/freeze. "
            "Winner requires ΔJ above seed scatter. Dual is admitted only as a two-network "
            "tube graph: Φ_c, Φ_d ≥ 0.02 and each metal is 4-connected to its tube. "
            "Numbers are from this run only."
        ),
    }
    out = Path("results/spot.json") if args.spot_check else Path("results/sweep.json")
    out.write_text(json.dumps(report, indent=2))
    fig_map = Path("paper/figures/lambda_ra_map_spot.png") if args.spot_check else Path("paper/figures/lambda_ra_map.png")
    fig_cls = Path("paper/figures/lambda_ra_classes_spot.png") if args.spot_check else Path("paper/figures/lambda_ra_classes.png")
    plot_map(cells, fig_map)
    plot_classes(cells, fig_cls)
    print(f"wrote {out} and {fig_map}")
    if args.grid_study:
        print(f"grid-study flips: {len(flips)} {flips}")
    _ = jnp.array(0.0)


if __name__ == "__main__":
    main()
