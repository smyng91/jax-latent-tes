"""Write paper macros and figures from results/*.json. No solver rerun."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from pcm.published import GAU_VISKANTA_1986

ROOT = Path(__file__).resolve().parents[1]
RESDIR = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"
MACRO_PATH = ROOT / "paper" / "generated_numbers.tex"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def _savefig(fig, path: Path, plt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = fig.get_layout_engine()
    if engine is not None:
        try:
            engine.set(w_pad=0.04, h_pad=0.06, wspace=0.04, hspace=0.06)
        except TypeError:
            pass
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    if bbox is None:
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.45)
    else:
        fig.savefig(path, dpi=200, bbox_inches=bbox.padded(0.45))
    plt.close(fig)


def _sci(x: float, digits: int = 2) -> str:
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    rounded = round(mant, digits)
    if rounded >= 10:
        rounded /= 10.0
        exp += 1
    if digits == 0 or abs(rounded - round(rounded)) < 0.5 * 10 ** (-max(digits, 1)):
        return rf"{int(round(rounded))}\times 10^{{{exp}}}"
    return rf"{rounded:.{digits}f}\times 10^{{{exp}}}"


def macros_from_json(validate: dict, sweep: dict, explore: dict) -> dict[str, str]:
    st = validate["stefan"]
    gal = validate["gallium_gau_viskanta"]
    s_ex = gal["neumann_at_2min"]["s_over_W_exact"]
    s_nu = gal["cavity_conduction_2min"]["interface_x_over_W"]
    winners = Counter(c["winner"] for c in sweep["cells"])
    dual = explore["dual"]
    pareto = explore["pareto"]
    humid = explore["humidity"]
    cop = [r for r in explore["cop"] if r["cop_w"] == 2.0]
    sched = explore["schedule"]
    tr = explore["transfer"]
    p0, p1 = pareto[0], pareto[-1]
    neu = validate.get("neumann_series", {})
    energy = validate["energy"]
    cfl = validate.get("cfl_mush", {})
    tube = energy.get("tube_freeze", {})
    macros = {
        "StefanSte": f"{st['ste']:g}",
        "StefanFo": f"{st['fo']:g}",
        "StefanSexact": f"{st['s_exact']:.4f}",
        "StefanSnum": f"{st['s_numeric']:.4f}",
        "StefanRelErr": f"{100 * st['relative_error']:.2f}",
        "NeumannMaxErr": f"{100 * neu.get('max_relative_error', st['relative_error']):.2f}",
        "EnergyRMS": _sci(energy["steady_rms_drift"], 1),
        "TubeEnergyErr": f"{100 * tube['relative_error']:.2f}" if tube else "---",
        "CavityMeltRatio": f"{validate['cavity']['melt_fraction_ratio_Ra1e5_over_Ra0']:.2f}",
        "CflSpan": f"{100 * cfl.get('cfl_relative_span', 0.0):.1f}",
        "MushSpan": f"{100 * cfl.get('mush_relative_span', 0.0):.1f}",
        "GalliumSteBrent": f"{gal['brent_Ste']:g}",
        "GalliumSteProps": f"{gal['groups']['ste_from_props']:.5f}",
        "GalliumRaBrent": _sci(gal["brent_Ra"], 0),
        "GalliumRaProps": _sci(gal["groups"]["ra_from_props"], 2),
        "GalliumSNeumann": f"{s_ex:.4f}",
        "GalliumSnum": f"{s_nu:.4f}",
        "GalliumSrel": f"{100 * abs(s_nu - s_ex) / s_ex:.2f}",
        "GalliumFcond": f"{gal['cavity_conduction_17min']['mean_liquid_fraction']:.4f}",
        "GalliumFdarcy": f"{gal['cavity_ns_17min']['mean_liquid_fraction']:.4f}",
        "GalliumRatio": f"{gal['convection_volume_ratio_17min']:.2f}",
        "GalliumGVfactor": f"{gal['gau_viskanta_late_volume_over_neumann']:g}",
        "GalliumCFL": f"{gal['cavity_ns_17min'].get('energy_cfl', 32):g}",
        "PhiTarget": f"{sweep['phi']:g}",
        "SweepNx": str(sweep["nx"]),
        "SweepIter": str(sweep["n_iter"]),
        "SweepFoD": f"{sweep['Fo_d']:g}",
        "SweepSeeds": str(sweep.get("n_seeds", 1)),
        "SweepBetaEnd": f"{(sweep.get('betas') or [4])[-1]:g}",
        "MapMelt": str(winners.get("melt", 0)),
        "MapFreeze": str(winners.get("freeze", 0)),
        "MapCycle": str(winners.get("cycle", 0)),
        "MapDual": str(winners.get("dual", 0)),
        "MapTie": str(winners.get("tie", 0)),
        "MapN": str(len(sweep["cells"])),
        "GridFlips": str(
            json.loads((RESDIR / "grid_study.json").read_text())["n_flips"]
            if (RESDIR / "grid_study.json").exists()
            else sweep.get("n_flips", 0)
        ),
        "ExploreNx": str(explore["nx"]),
        "ExploreIter": str(explore["n_iter"]),
        "ExploreFo": f"{explore['Fo_c']:g}",
        "ExploreCycleJ": f"{dual['cycle_phi0.1']['J_cycle']:.3f}",
        "ExploreDualJ": f"{dual['dual_phi0.1']['J_cycle']:.3f}",
        "ExploreDualPhiTwoJ": f"{dual['dual_phi0.2']['J_cycle']:.3f}",
        "ExploreDualPhiTwoVol": f"{dual['dual_phi0.2']['volume']:.3f}",
        "ExploreSwitchedJ": f"{dual['dual_switched_phi0.1']['J_switched']:.3f}",
        "ExploreParetoTopMelt": f"{p0['metal_top']:.3f}",
        "ExploreParetoTopFreeze": f"{p1['metal_top']:.3f}",
        "ExploreParetoBotMelt": f"{p0['metal_bottom']:.3f}",
        "ExploreParetoBotFreeze": f"{p1['metal_bottom']:.3f}",
        "ExploreHumidTopZero": f"{humid[0]['metal_top']:.3f}",
        "ExploreHumidTopOne": f"{humid[-1]['metal_top']:.3f}",
        "ExploreSteCLowJ": f"{cop[0]['J_cycle']:.3f}",
        "ExploreSteCHighJ": f"{cop[-1]['J_cycle']:.3f}",
        "ExploreFoCshortJ": f"{sched[0]['J_cycle']:.3f}",
        "ExploreFoClongJ": f"{sched[-1]['J_cycle']:.3f}",
        "ExploreMeltOnCycle": f"{tr['melt_on_cycle']:.3f}",
        "ExploreFreezeOnCycle": f"{tr['freeze_on_cycle']:.3f}",
        "ExploreLayoutGap": f"{abs(tr['layout_gap_top']):.3f}",
    }
    bpath = RESDIR / "baseline.json"
    if bpath.exists():
        baseline = json.loads(bpath.read_text())
        macros["BaselineNx"] = str(baseline["nx"])
        macros["BaselineIter"] = str(baseline["n_iter"])
        djs = [c["dJ_opt_minus_annular"] for c in baseline["cells"]]
        macros["BaselineDJMin"] = f"{min(djs):.3f}"
        macros["BaselineDJMax"] = f"{max(djs):.3f}"
        macros["BaselineHorizon"] = f"{baseline.get('Fo_horizon', 2.4):g}"

        def _at(lam, ra):
            return next(
                c
                for c in baseline["cells"]
                if abs(c["Lambda_target"] - lam) < 1e-9 and abs(c["Ra_d"] - ra) < 1.0
            )

        try:
            c_short = _at(1.0 / 3.0, 0.0)
            c_long = _at(3.0, 3.0e5)
            macros["BaselineAnnularJShort"] = f"{c_short['annular']['J_cycle']:.3f}"
            macros["BaselineCycleJShort"] = f"{c_short['optimized']['J_cycle']:.3f}"
            macros["BaselineAnnularJLong"] = f"{c_long['annular']['J_cycle']:.3f}"
            macros["BaselineCycleJLong"] = f"{c_long['optimized']['J_cycle']:.3f}"
        except StopIteration:
            pass
    c32 = RESDIR / "sweep_nx32.json"
    if c32.exists():
        coarse = json.loads(c32.read_text())
        macros["SweepNxCoarse"] = str(coarse["nx"])
        cw = Counter(c["winner"] for c in coarse["cells"])
        macros["CoarseDual"] = str(cw.get("dual", 0))
        macros["CoarseCycle"] = str(cw.get("cycle", 0))
        macros["CoarseTie"] = str(cw.get("tie", 0))
    else:
        macros["SweepNxCoarse"] = str(sweep["nx"])
    return macros


def _j(cell: dict, name: str) -> float:
    a = cell["architectures"][name]
    return float(a.get("J_cycle_mean", a["J_cycle"]))


def _lam_tex(x: float) -> str:
    for v, s in ((1.0 / 3.0, r"1/3"), (0.5, r"1/2"), (1.0, "1"), (2.0, "2"), (3.0, "3")):
        if abs(x - v) < 1e-9:
            return s
    return f"{x:g}"


def _ra_tex(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    if abs(x - 3.0e4) < 1.0:
        return r"3\times 10^{4}"
    if abs(x - 1.0e5) < 1.0:
        return r"10^{5}"
    if abs(x - 3.0e5) < 1.0:
        return r"3\times 10^{5}"
    return _sci(x, 0)


def write_map_tex(sweep: dict, path: Path | None = None) -> Path:
    path = path or (ROOT / "paper" / "generated_map.tex")
    lines = [
        r"\begin{tabular}{@{}rrlcccc@{}}",
        r"\toprule",
        r"$\Lambda$ & $\mathrm{Ra}_d$ & winner & $J_{\mathrm{melt}}$ & $J_{\mathrm{freeze}}$ & $J_{\mathrm{cycle}}$ & $J_{\mathrm{dual}}$ \\",
        r"\midrule",
    ]
    for c in sweep["cells"]:
        arch = c["architectures"]
        jm = jf = jc = jd = "---"
        if "melt" in arch:
            jm = f"{_j(c, 'melt'):.3f}"
        if "freeze" in arch:
            jf = f"{_j(c, 'freeze'):.3f}"
        if "cycle" in arch:
            jc = f"{_j(c, 'cycle'):.3f}"
        if "dual" in arch:
            jd = f"{_j(c, 'dual'):.3f}"
        lines.append(
            rf"${_lam_tex(c['Lambda_target'])}$ & ${_ra_tex(c['Ra_d'])}$ & {c['winner']} "
            rf"& ${jm}$ & ${jf}$ & ${jc}$ & ${jd}$ \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")
    return path


def write_neumann_tex(validate: dict, path: Path | None = None) -> Path:
    path = path or (ROOT / "paper" / "generated_neumann.tex")
    lines = [
        r"\begin{tabular}{@{}lllll@{}}",
        r"\toprule",
        r"$\mathrm{Ste}$ & $\mathrm{Fo}$ & $s/L$ exact & $s/L$ numeric & relative error (\%) \\",
        r"\midrule",
    ]
    for r in validate["neumann_series"]["cases"]:
        lines.append(
            rf"{r['ste']:.2f} & {r['fo']:.2f} & {r['s_exact']:.4f} & {r['s_numeric']:.4f} & {100 * r['relative_error']:.2f} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")
    return path


def write_baseline_tex(baseline: dict, path: Path | None = None) -> Path:
    path = path or (ROOT / "paper" / "generated_baseline.tex")
    lines = [
        r"\begin{tabular}{@{}llllllll@{}}",
        r"\toprule",
        r"$\Lambda$ & $\mathrm{Ra}_d$ & design & $J_{\mathrm{cycle}}$ & volume & $\mathrm{Fo}_{95,\mathrm{fr}}$ & $\mathrm{Fo}_{90,\mathrm{fr}}$ & $\mathrm{Fo}_{95,\mathrm{m}}$ \\",
        r"\midrule",
    ]

    def fo(x):
        return "---" if x is None else f"{x:.2f}"

    def row(c, name, d):
        return (
            rf"${_lam_tex(c['Lambda_target'])}$ & ${_ra_tex(c['Ra_d'])}$ & {name} "
            rf"& ${d['J_cycle']:.3f}$ & ${d['volume']:.3f}$ & ${fo(d['Fo_freeze_95'])}$ "
            rf"& ${fo(d.get('Fo_freeze_90'))}$ & ${fo(d['Fo_melt_95'])}$ \\"
        )

    for c in baseline["cells"]:
        lines.append(row(c, "annular", c["annular"]))
        if "spoke" in c:
            lines.append(row(c, "spokes", c["spoke"]))
        lines.append(row(c, "cycle TO", c["optimized"]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")
    return path


def write_generated_numbers(
    validate: dict | None = None,
    sweep: dict | None = None,
    explore: dict | None = None,
    path: Path | None = None,
) -> Path:
    validate = validate or json.loads((RESDIR / "validate.json").read_text())
    sweep = sweep or json.loads((RESDIR / "sweep.json").read_text())
    explore = explore or json.loads((RESDIR / "explore.json").read_text())
    if sweep.get("nx", 0) >= 48:
        from pcm.classify import reclassify_cell
        from pcm.physics import two_tube_geom

        geom = two_tube_geom(sweep["nx"], sweep["ny"])
        new_cells = [reclassify_cell(c, geom) for c in sweep["cells"]]
        sweep = {**sweep, "cells": new_cells}
        if any(a.get("winner") != b.get("winner") for a, b in zip(json.loads((RESDIR / "sweep.json").read_text())["cells"], new_cells)):
            (RESDIR / "sweep.json").write_text(json.dumps(sweep, indent=2))
    macros = macros_from_json(validate, sweep, explore)
    bpath = RESDIR / "baseline.json"
    if bpath.exists():
        baseline = json.loads(bpath.read_text())
        macros["BaselineNx"] = str(baseline["nx"])
        macros["BaselineIter"] = str(baseline["n_iter"])
        djs = [c["dJ_opt_minus_annular"] for c in baseline["cells"]]
        macros["BaselineDJMin"] = f"{min(djs):.3f}"
        macros["BaselineDJMax"] = f"{max(djs):.3f}"
        macros["BaselineHorizon"] = f"{baseline.get('Fo_horizon', 2.4):g}"
        if any("spoke" in c for c in baseline["cells"]):
            sjs = [c["dJ_opt_minus_spoke"] for c in baseline["cells"] if "dJ_opt_minus_spoke" in c]
            macros["BaselineSpokeDJMin"] = f"{min(sjs):.3f}"
            macros["BaselineSpokeDJMax"] = f"{max(sjs):.3f}"
        write_baseline_tex(baseline)
    gpath = RESDIR / "grid_study.json"
    if gpath.exists():
        gs = json.loads(gpath.read_text())
        macros["GridFlips"] = str(gs["n_flips"])
    spath = RESDIR / "spot.json"
    if spath.exists():
        spot = json.loads(spath.read_text())
        macros["SpotNx"] = str(spot["nx"])
        macros["SpotIter"] = str(spot["n_iter"])
        sw = Counter(c["winner"] for c in spot["cells"])
        macros["SpotDual"] = str(sw.get("dual", 0))
        macros["SpotCycle"] = str(sw.get("cycle", 0))
        macros["SpotN"] = str(len(spot["cells"]))
    path = path or MACRO_PATH
    lines = [
        "% Headline digits from results/*.json. Do not edit by hand.",
        *(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()),
    ]
    path.write_text("\n".join(lines) + "\n")
    write_map_tex(sweep)
    if "neumann_series" in validate:
        write_neumann_tex(validate)
    return path


METAL_THRESH = 0.5
CHARGE_RGB = (0.16, 0.42, 0.76)
DISCHARGE_RGB = (0.86, 0.40, 0.12)
TUBE_RGB = (0.12, 0.12, 0.12)
FIELD_CACHE = RESDIR / "winner_fields.npz"
FIELD_SPOTS = (
    (1.0 / 3.0, 0.0),
    (1.0 / 3.0, 3.0e5),
    (1.0, 1.0e5),
    (3.0, 3.0e5),
)


def _sorted_grid(cells):
    lams = sorted({c["Lambda_target"] for c in cells})
    ras = sorted({c["Ra_d"] for c in cells})
    return lams, ras


def _cell_at(cells, lam, ra):
    return next(
        c for c in cells if abs(c["Lambda_target"] - lam) < 1e-9 and abs(c["Ra_d"] - ra) < 1.0
    )


def _field_key(lam, ra) -> tuple[float, float]:
    return (round(float(lam), 10), round(float(ra), 4))


def _label_map_ax(ax, i, j, n_ra, lam, ra) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if i == n_ra - 1:
        ax.set_xlabel(rf"$\Lambda={lam:.2g}$")
    if j == 0:
        ax.set_ylabel(rf"$\mathrm{{Ra}}_d={ra:.1g}$")


def _dJ_title(cell: dict) -> str:
    dJ = cell.get("dual_minus_cycle_J")
    if dJ is None:
        return str(cell.get("winner", ""))
    return f"ΔJ={dJ:.3f}"


def dual_rgb(gc, gd, gamma):
    """Charge metal blue, discharge metal orange, tubes dark, PCM white."""
    import numpy as np

    gc = np.clip(np.asarray(gc, dtype=float), 0.0, 1.0)
    gd = np.clip(np.asarray(gd, dtype=float), 0.0, 1.0)
    g = np.asarray(gamma, dtype=float)
    rgb = np.ones(gc.shape + (3,))
    rgb = rgb * (1.0 - gc[..., None]) + np.asarray(CHARGE_RGB) * gc[..., None]
    rgb = rgb * (1.0 - gd[..., None]) + np.asarray(DISCHARGE_RGB) * gd[..., None]
    tubes = (g > 0.9) & (gc + gd < 0.1)
    rgb[tubes] = TUBE_RGB
    return np.clip(rgb, 0.0, 1.0)


def pcm_masked(field, gamma, thresh: float = METAL_THRESH):
    """NaN-out solid metal so field plots keep the dark topology overlay."""
    import numpy as np

    out = np.array(field, dtype=float, copy=True)
    out[np.asarray(gamma) >= thresh] = np.nan
    return out


def _fields_signature(sweep: dict) -> str:
    import numpy as np

    parts = [str(sweep["nx"]), str(sweep["ny"]), str(sweep["Fo_d"]), str(sweep.get("energy_cfl", 8.0))]
    for c in sweep["cells"]:
        g = np.asarray(c["winner_gamma"])
        parts.append(
            f"{c['Lambda_target']:.8g}:{c['Ra_d']:.8g}:{g.mean():.8g}:{g.std():.8g}:{g.max():.8g}"
        )
    return "|".join(parts)


def _load_field_cache(path: Path, signature: str):
    import numpy as np

    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if str(np.asarray(data["signature"]).item()) != signature:
        return None
    keys = [_field_key(k[0], k[1]) for k in data["keys"]]
    out = {}
    for i, key in enumerate(keys):
        out[key] = {name: data[f"{i}_{name}"] for name in ("gamma", "Tf", "Tm", "fl", "fm", "speed", "um", "vm")}
    return out


def _save_field_cache(path: Path, signature: str, fields: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signature": np.asarray(signature), "keys": np.array(list(fields.keys()), dtype=float)}
    for i, d in enumerate(fields.values()):
        for name, arr in d.items():
            payload[f"{i}_{name}"] = arr
    np.savez_compressed(path, **payload)


def replay_winner_fields(sweep: dict, cache: Path | None = FIELD_CACHE) -> dict:
    """Forward freeze-then-melt on each stored winner γ. Not a topology-optimization rerun."""
    import numpy as np

    signature = _fields_signature(sweep)
    if cache is not None:
        cached = _load_field_cache(cache, signature)
        if cached is not None:
            return cached

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from pcm.optimize import T_FREEZE0
    from pcm.physics import liquid_fraction, simulate, tube_bcs, two_tube_geom
    from pcm.sweep import make_sims

    nx, ny = int(sweep["nx"]), int(sweep["ny"])
    geom = two_tube_geom(nx, ny)
    energy_cfl = float(sweep.get("energy_cfl", 8.0))
    fields = {}
    for cell in sweep["cells"]:
        key = _field_key(cell["Lambda_target"], cell["Ra_d"])
        gamma = jnp.asarray(cell["winner_gamma"])
        sim_c, sim_d, *_ = make_sims(nx, ny, cell["Fo_c"], cell["Fo_d"], cell["Ra_d"], energy_cfl=energy_cfl)
        T0 = jnp.full((nx, ny), T_FREEZE0)
        Tf, _uf, _vf = simulate(T0, gamma, geom, sim_c, tube_bcs(charge_on=True, discharge_on=False))
        Tm, um, vm = simulate(Tf, gamma, geom, sim_d, tube_bcs(charge_on=False, discharge_on=True))
        g = np.asarray(gamma)
        Tf_np, Tm_np = np.asarray(Tf), np.asarray(Tm)
        um_np, vm_np = np.asarray(um), np.asarray(vm)
        eps = float(sim_c.eps)
        fl = np.asarray(liquid_fraction(jnp.asarray(Tf_np), eps)) * (1.0 - g)
        fm = np.asarray(liquid_fraction(jnp.asarray(Tm_np), eps)) * (1.0 - g)
        fields[key] = {
            "gamma": g,
            "Tf": Tf_np,
            "Tm": Tm_np,
            "fl": fl,
            "fm": fm,
            "speed": np.sqrt(um_np * um_np + vm_np * vm_np),
            "um": um_np,
            "vm": vm_np,
        }
        print(
            f"  fields Λ={cell['Lambda_target']:.3g} Ra_d={cell['Ra_d']:.1g} "
            f"θ_fr=[{Tf_np.min():.2f},{Tf_np.max():.2f}] |u|_max={fields[key]['speed'].max():.2f}",
            flush=True,
        )
    if cache is not None:
        _save_field_cache(cache, signature, fields)
    return fields


def plot_map(cells, path: Path) -> None:
    plt, np = _mpl()
    lams, ras = _sorted_grid(cells)
    fig, axes = plt.subplots(
        len(ras), len(lams), figsize=(2.2 * len(lams), 2.15 * len(ras)), squeeze=False, layout="constrained"
    )
    for i, ra in enumerate(ras[::-1]):
        for j, lam in enumerate(lams):
            cell = _cell_at(cells, lam, ra)
            ax = axes[i][j]
            g = np.asarray(cell["winner_gamma"])
            ax.imshow(g.T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=[0, 1, 0, 1])
            _label_map_ax(ax, i, j, len(ras), lam, ra)
            ax.set_title(_dJ_title(cell), fontsize=7)
    fig.suptitle(r"Combined metal $\gamma$ (black). $\Delta J=J_{\mathrm{cycle}}-J_{\mathrm{dual}}$", fontsize=10)
    _savefig(fig, path, plt)


def plot_dual_networks(cells, path: Path) -> None:
    plt, np = _mpl()
    lams, ras = _sorted_grid(cells)
    fig, axes = plt.subplots(
        len(ras), len(lams), figsize=(2.2 * len(lams), 2.25 * len(ras)), squeeze=False, layout="constrained"
    )
    for i, ra in enumerate(ras[::-1]):
        for j, lam in enumerate(lams):
            cell = _cell_at(cells, lam, ra)
            ax = axes[i][j]
            g = np.asarray(cell["winner_gamma"])
            split = cell.get("winner_dual") or cell.get("dual_fields") or {}
            if "gamma_c" in split and "gamma_d" in split:
                rgb = dual_rgb(split["gamma_c"], split["gamma_d"], g)
                ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=[0, 1, 0, 1])
            else:
                ax.imshow(g.T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=[0, 1, 0, 1])
            _label_map_ax(ax, i, j, len(ras), lam, ra)
            ax.set_title(_dJ_title(cell), fontsize=7)
    fig.suptitle(r"Blue: charge $\gamma_c$ (bottom tube). Orange: discharge $\gamma_d$ (top tube).", fontsize=10)
    _savefig(fig, path, plt)


def plot_field_map(
    cells,
    fields: dict,
    name: str,
    path: Path,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar: str,
    title: str,
    stat: str = "mean",
) -> None:
    plt, np = _mpl()
    lams, ras = _sorted_grid(cells)
    fig, axes = plt.subplots(
        len(ras), len(lams), figsize=(2.25 * len(lams), 2.2 * len(ras)), squeeze=False, layout="constrained"
    )
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.18")
    im = None
    for i, ra in enumerate(ras[::-1]):
        for j, lam in enumerate(lams):
            cell = _cell_at(cells, lam, ra)
            key = _field_key(cell["Lambda_target"], cell["Ra_d"])
            pack = fields[key]
            ax = axes[i][j]
            masked = pcm_masked(pack[name], pack["gamma"])
            im = ax.imshow(
                masked.T, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1]
            )
            _label_map_ax(ax, i, j, len(ras), lam, ra)
            pcm = pack["gamma"] < METAL_THRESH
            vals = pack[name][pcm]
            label = f"{np.nanmax(vals):.2f}" if stat == "max" else f"{np.nanmean(vals):.2f}"
            ax.set_title(label, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02, label=cbar)
    fig.suptitle(title, fontsize=10)
    _savefig(fig, path, plt)


def plot_optimal_fields(cells, fields: dict, path: Path, spots=FIELD_SPOTS) -> None:
    """Four representative winners: two-network density, θ, liquid fraction, speed."""
    plt, np = _mpl()
    rows = []
    for lam, ra in spots:
        try:
            cell = _cell_at(cells, lam, ra)
        except StopIteration:
            continue
        key = _field_key(cell["Lambda_target"], cell["Ra_d"])
        if key in fields:
            rows.append((cell, fields[key]))
    if not rows:
        return
    cmap_t = plt.get_cmap("RdBu_r").copy()
    cmap_f = plt.get_cmap("YlGnBu").copy()
    cmap_u = plt.get_cmap("viridis").copy()
    for c in (cmap_t, cmap_f, cmap_u):
        c.set_bad("0.18")
    nr = len(rows)
    fig = plt.figure(figsize=(14.8, 2.9 * nr), layout="constrained")
    gs = fig.add_gridspec(nr, 9, width_ratios=[1.08, 1, 0.055, 1, 0.055, 1, 0.055, 1, 0.055])
    axes = [
        [
            fig.add_subplot(gs[i, 0]),
            fig.add_subplot(gs[i, 1]),
            fig.add_subplot(gs[i, 3]),
            fig.add_subplot(gs[i, 5]),
            fig.add_subplot(gs[i, 7]),
        ]
        for i in range(nr)
    ]
    cax_t = fig.add_subplot(gs[:, 2])
    cax_f = fig.add_subplot(gs[:, 4])
    cax_tm = fig.add_subplot(gs[:, 6])
    cax_u = fig.add_subplot(gs[:, 8])
    umax = max(float(pack["speed"].max()) for _, pack in rows)
    umax = max(umax, 0.05)
    col_titles = (
        r"metal $\gamma_c,\gamma_d$",
        r"$\theta$ after freeze",
        r"$f_\ell$ after freeze",
        r"$\theta$ after melt",
        r"$|u|$ after melt",
    )
    last = {}
    for i, (cell, pack) in enumerate(rows):
        split = cell.get("winner_dual") or {}
        if "gamma_c" in split:
            rgb = dual_rgb(split["gamma_c"], split["gamma_d"], pack["gamma"])
            axes[i][0].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=[0, 1, 0, 1])
        else:
            axes[i][0].imshow(pack["gamma"].T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=[0, 1, 0, 1])
        last["t"] = axes[i][1].imshow(
            pcm_masked(pack["Tf"], pack["gamma"]).T,
            origin="lower",
            cmap=cmap_t,
            vmin=-1,
            vmax=1,
            extent=[0, 1, 0, 1],
        )
        last["f"] = axes[i][2].imshow(
            pcm_masked(pack["fl"], pack["gamma"]).T,
            origin="lower",
            cmap=cmap_f,
            vmin=0,
            vmax=1,
            extent=[0, 1, 0, 1],
        )
        last["tm"] = axes[i][3].imshow(
            pcm_masked(pack["Tm"], pack["gamma"]).T,
            origin="lower",
            cmap=cmap_t,
            vmin=-1,
            vmax=1,
            extent=[0, 1, 0, 1],
        )
        last["u"] = axes[i][4].imshow(
            pcm_masked(pack["speed"], pack["gamma"]).T,
            origin="lower",
            cmap=cmap_u,
            vmin=0,
            vmax=umax,
            extent=[0, 1, 0, 1],
        )
        if pack["speed"].max() > 0.05:
            n = pack["um"].shape[0]
            x = (np.arange(n) + 0.5) / n
            y = (np.arange(n) + 0.5) / n
            skip = max(1, n // 16)
            axes[i][4].streamplot(
                x[::skip],
                y[::skip],
                pack["um"].T[::skip, ::skip],
                pack["vm"].T[::skip, ::skip],
                color="w",
                density=0.55,
                linewidth=0.4,
                arrowsize=0.5,
            )
            axes[i][4].set_xlim(0, 1)
            axes[i][4].set_ylim(0, 1)
        axes[i][0].set_ylabel(
            rf"$\Lambda={cell['Lambda_target']:.2g}$, $\mathrm{{Ra}}_d={cell['Ra_d']:.1g}$",
            fontsize=8,
        )
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        if i == 0:
            for ax, lab in zip(axes[i], col_titles):
                ax.set_title(lab, fontsize=9)
    fig.colorbar(last["t"], cax=cax_t, label=r"$\theta$")
    fig.colorbar(last["f"], cax=cax_f, label=r"$f_\ell$")
    fig.colorbar(last["tm"], cax=cax_tm, label=r"$\theta$")
    fig.colorbar(last["u"], cax=cax_u, label=r"$|u|$")
    fig.suptitle(r"Winning dual topologies (dark = metal $\gamma\geq 0.5$)", fontsize=11)
    _savefig(fig, path, plt)


def plot_winner_fields(sweep: dict, figdir: Path) -> None:
    fields = replay_winner_fields(sweep)
    cells = sweep["cells"]
    umax = max(float(p["speed"].max()) for p in fields.values())
    umax = max(umax, 0.05)
    plot_field_map(
        cells,
        fields,
        "Tf",
        figdir / "lambda_ra_theta_freeze.png",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        cbar=r"$\theta$",
        title=r"Temperature $\theta$ after charge. Dark = metal.",
        stat="mean",
    )
    plot_field_map(
        cells,
        fields,
        "Tm",
        figdir / "lambda_ra_theta_melt.png",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        cbar=r"$\theta$",
        title=r"Temperature $\theta$ after sequential discharge. Dark = metal.",
        stat="mean",
    )
    plot_field_map(
        cells,
        fields,
        "fl",
        figdir / "lambda_ra_liquid_freeze.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=r"$f_\ell$",
        title=r"Liquid fraction $f_\ell$ after charge. Dark = metal.",
        stat="mean",
    )
    plot_field_map(
        cells,
        fields,
        "fm",
        figdir / "lambda_ra_liquid_melt.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=r"$f_\ell$",
        title=r"Liquid fraction $f_\ell$ after sequential discharge. Dark = metal.",
        stat="mean",
    )
    plot_field_map(
        cells,
        fields,
        "speed",
        figdir / "lambda_ra_speed_melt.png",
        cmap="viridis",
        vmin=0.0,
        vmax=umax,
        cbar=r"$|u|$",
        title=r"Discharge speed $|u|$ at end of melt. Dark = metal.",
        stat="max",
    )
    plot_optimal_fields(cells, fields, figdir / "optimal_fields.png")


def plot_classes(cells, path: Path) -> None:
    """Margin heatmap: how much dual beats a static cycle. Replaces a class chart when every cell is dual."""
    plt, np = _mpl()
    lams = sorted({c["Lambda_target"] for c in cells})
    ras = sorted({c["Ra_d"] for c in cells})
    z = np.full((len(ras), len(lams)), np.nan)
    for i, ra in enumerate(ras):
        for j, lam in enumerate(lams):
            cell = next(c for c in cells if c["Lambda_target"] == lam and c["Ra_d"] == ra)
            dJ = cell.get("dual_minus_cycle_J")
            z[i, j] = dJ if dJ is not None else np.nan
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    vmax = max(0.05, float(np.nanmax(np.abs(z))))
    if np.nanmin(z) < 0:
        im = ax.imshow(z, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    else:
        im = ax.imshow(z, origin="lower", cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="equal")
    for i, ra in enumerate(ras):
        for j, lam in enumerate(lams):
            val = z[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="k")
    ax.set_xticks(range(len(lams)), [f"{lam:.2g}" for lam in lams])
    ax.set_yticks(range(len(ras)), [f"{ra:.1g}" for ra in ras])
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$\mathrm{Ra}_d$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta J=J_{\mathrm{cycle}}-J_{\mathrm{dual}}$")
    ax.set_title("Dual margin over a static cycle (positive: dual pays)")
    _savefig(fig, path, plt)


def plot_explore(out: dict, figdir: Path) -> None:
    plt, np = _mpl()
    figdir.mkdir(parents=True, exist_ok=True)
    pareto = out["pareto"]
    dual = out["dual"]
    cop = out["cop"]
    sched = out["schedule"]

    fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
    ws = [r["w_fr"] for r in pareto]
    ax.plot(ws, [r["metal_top"] for r in pareto], "o-", label="metal top (discharge half)")
    ax.plot(ws, [r["metal_bottom"] for r in pareto], "s--", label="metal bottom (charge half)")
    ax.set_xlabel(r"freeze weight $w$")
    ax.set_ylabel("design-cell metal fraction")
    ax.set_title("Pareto $w$ moves metal toward the charge half")
    ax.legend(frameon=False, fontsize=8)
    _savefig(fig, figdir / "explore_pareto.png", plt)

    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    labels = list(dual.keys())
    js = [dual[k].get("J_switched", dual[k]["J_cycle"]) if "switched" in k else dual[k]["J_cycle"] for k in labels]
    hatches = ["///" if "switched" in k else "" for k in labels]
    ax.bar(np.arange(len(labels)), js, color="0.35", hatch=hatches, edgecolor="0.15")
    ax.set_xticks(np.arange(len(labels)), [k.replace("_", "\n") for k in labels], fontsize=7)
    ax.set_ylabel(r"$J$")
    ax.set_title("Switched dual vs valved dual; extra metal helps more")
    _savefig(fig, figdir / "explore_dual.png", plt)

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.5), layout="constrained")
    ax[0].plot([r["Fo_c"] for r in sched], [r["J_cycle"] for r in sched], "o-")
    ax[0].set_xlabel(r"$\mathrm{Fo}_c$ at fixed $\mathrm{Fo}_d$")
    ax[0].set_ylabel(r"$J_{\mathrm{cycle}}$")
    ax[0].set_title(r"Longer freeze window lowers $J_{\mathrm{cycle}}$")
    cop_phys = [r for r in cop if r["cop_w"] == 2.0]
    ax[1].plot([r["Ste_c"] for r in cop_phys], [r["J_cycle"] for r in cop_phys], "s-")
    ax[1].set_xlabel(r"$\mathrm{Ste}_c$ at fixed $\mathrm{Ste}_d=0.1$")
    ax[1].set_ylabel(r"$J_{\mathrm{cycle}}$")
    ax[1].set_title(r"Colder charge (higher $\mathrm{Ste}_c$)")
    _savefig(fig, figdir / "explore_schedule.png", plt)


def plot_validate(report: dict, figdir: Path) -> None:
    plt, np = _mpl()
    figdir.mkdir(parents=True, exist_ok=True)
    series = report["neumann_series"]
    gallium = report["gallium_gau_viskanta"]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for ste in sorted({c["ste"] for c in series["cases"]}):
        cs = [c for c in series["cases"] if c["ste"] == ste]
        ax.plot([c["fo"] for c in cs], [c["s_exact"] for c in cs], "o", mfc="none", label=f"Neumann Ste={ste}")
        ax.plot([c["fo"] for c in cs], [c["s_numeric"] for c in cs], "s", label=f"this code Ste={ste}")
    ax.set_xlabel(r"Fo $=\alpha t/L^2$")
    ax.set_ylabel(r"$s/L$")
    ax.set_title("Computed $s/L$ tracks the two-phase Neumann root")
    ax.legend(frameon=False, fontsize=8)
    _savefig(fig, figdir / "neumann_series.png", plt)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), layout="constrained")
    ne = gallium["neumann_at_2min"]["s_over_W_exact"]
    s2 = gallium["cavity_conduction_2min"]["interface_x_over_W"]
    c17 = gallium["cavity_conduction_17min"]["mean_liquid_fraction"]
    f17 = gallium["cavity_ns_17min"]["mean_liquid_fraction"]
    gv = GAU_VISKANTA_1986["late_time_volume_over_neumann"] * c17
    axes[0].bar([0, 1], [ne, s2], color="0.3")
    axes[0].set_xticks([0, 1], ["Neumann\n2 min", "conduction\n2 min"], fontsize=7)
    axes[0].set_ylabel(r"$s/W$")
    axes[0].set_title(r"$2\,\mathrm{min}$: conduction vs Neumann")
    axes[1].bar([0, 1, 2], [c17, f17, gv], color="0.3")
    axes[1].set_xticks(
        [0, 1, 2],
        ["conduction\n17 min", "NS\n17 min", r"1.75$\times$ conduction" + "\n(Gau–Viskanta)"],
        fontsize=7,
    )
    axes[1].set_ylabel("liquid fraction")
    axes[1].set_title(r"$17\,\mathrm{min}$: NS vs Gau--Viskanta")
    _savefig(fig, figdir / "gallium_gau_viskanta.png", plt)


def plot_baseline(report: dict, figdir: Path) -> None:
    plt, np = _mpl()
    figdir.mkdir(parents=True, exist_ok=True)
    keys = [("annular_gamma", "annular"), ("optimized_gamma", "cycle TO")]
    if report["cells"] and "spoke_gamma" in report["cells"][0]:
        keys = [("annular_gamma", "annular"), ("spoke_gamma", "spokes"), ("optimized_gamma", "cycle TO")]
    fig, axes = plt.subplots(
        len(report["cells"]),
        len(keys),
        figsize=(3.2 * len(keys), 3.2 * len(report["cells"])),
        squeeze=False,
        layout="constrained",
    )
    for i, c in enumerate(report["cells"]):
        for j, (key, title) in enumerate(keys):
            ax = axes[i][j]
            ax.imshow(np.asarray(c[key]).T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=[0, 1, 0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(rf"$\Lambda={c['Lambda_target']:.2g}$, $\mathrm{{Ra}}_d={c['Ra_d']:.1g}$" + "\n" + title, fontsize=8)
    _savefig(fig, figdir / "baseline_annular.png", plt)


def plot_from_json() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    sweep = json.loads((RESDIR / "sweep.json").read_text())
    plot_map(sweep["cells"], FIGDIR / "lambda_ra_map.png")
    plot_dual_networks(sweep["cells"], FIGDIR / "lambda_ra_dual.png")
    plot_classes(sweep["cells"], FIGDIR / "lambda_ra_classes.png")
    plot_winner_fields(sweep, FIGDIR)
    plot_explore(json.loads((RESDIR / "explore.json").read_text()), FIGDIR)
    plot_validate(json.loads((RESDIR / "validate.json").read_text()), FIGDIR)
    bpath = RESDIR / "baseline.json"
    if bpath.exists():
        plot_baseline(json.loads(bpath.read_text()), FIGDIR)
    print(f"wrote figures under {FIGDIR}")


def main_numbers() -> None:
    path = write_generated_numbers()
    print(f"wrote {path}")


def main_figures() -> None:
    plot_from_json()


if __name__ == "__main__":
    main_numbers()
    main_figures()
