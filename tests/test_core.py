"""Checks that fail if the Neumann root or paper macros drift off the JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pcm.published import neumann_lambda, stefan_lambda
from pcm.classify import pick_winner
from pcm.report import macros_from_json

ROOT = Path(__file__).resolve().parents[1]


def _macros_tex(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.startswith(r"\newcommand{"):
            continue
        rest = line[len(r"\newcommand{") :]
        name, _, rhs = rest.partition("}{")
        out[name.lstrip("\\")] = rhs[:-1]
    return out


def test_stefan_lambda_matches_carslaw():
    lam = stefan_lambda(0.2)
    lhs = lam * math.sqrt(math.pi) * math.erf(lam) * math.exp(lam * lam)
    assert abs(lhs - 0.2) < 1e-9
    assert abs(lam - 0.3064239053612112) < 1e-10
    assert abs(neumann_lambda(0.2, 0.0) - lam) < 1e-12
    # Subcooling reduces λ (slower melt).
    assert neumann_lambda(0.2, 0.04) < lam


def test_pick_winner_respects_margin():
    assert pick_winner({"melt": 0.80, "freeze": 0.81, "cycle": 0.90}) == "melt"
    assert pick_winner({"melt": 0.80, "freeze": 0.801}) == "tie"
    assert pick_winner({"melt": 0.80, "freeze": 0.82}, scatter={"melt": 0.03, "freeze": 0.03}) == "tie"


def test_dual_tube_graphs_require_attachment():
    from pcm.classify import dual_admitted, dual_tube_graphs
    from pcm.physics import two_tube_geom

    geom = two_tube_geom(24, 24)
    gc = np.zeros((24, 24))
    gd = np.zeros((24, 24))
    ok, _ = dual_tube_graphs(gc, gd, geom)
    assert not ok
    gc = np.asarray(geom.design, dtype=float) * np.asarray(geom.charge, dtype=float)
    # Charge metal only next to charge tube: dilate charge into design.
    charge = np.asarray(geom.charge)
    design = np.asarray(geom.design)
    near_c = np.zeros_like(charge)
    near_d = np.zeros_like(charge)
    for i in range(24):
        for j in range(24):
            if not design[i, j]:
                continue
            if any(
                charge[ii, jj]
                for ii in range(max(i - 1, 0), min(i + 2, 24))
                for jj in range(max(j - 1, 0), min(j + 2, 24))
            ):
                near_c[i, j] = 1.0
            disc = np.asarray(geom.discharge)
            if any(
                disc[ii, jj]
                for ii in range(max(i - 1, 0), min(i + 2, 24))
                for jj in range(max(j - 1, 0), min(j + 2, 24))
            ):
                near_d[i, j] = 1.0
    ok, stats = dual_tube_graphs(near_c, near_d, geom)
    assert ok, stats
    assert dual_admitted(0.05, 0.05, near_c, near_d, geom)
    assert not dual_admitted(0.01, 0.05, near_c, near_d, geom)


def test_macros_match_generated_numbers():
    validate = json.loads((ROOT / "results/validate.json").read_text())
    sweep = json.loads((ROOT / "results/sweep.json").read_text())
    explore = json.loads((ROOT / "results/explore.json").read_text())
    got = macros_from_json(validate, sweep, explore)
    listed = _macros_tex((ROOT / "paper/generated_numbers.tex").read_text())
    for k, v in got.items():
        assert listed.get(k) == v, (k, listed.get(k), v)


def test_highlights_use_headline_dual_count():
    h = (ROOT / "paper/highlights.tex").read_text().replace("$", "")
    listed = _macros_tex((ROOT / "paper/generated_numbers.tex").read_text())
    assert f"{listed['MapDual']} of {listed['MapN']}" in h
    if listed["CoarseDual"] != listed["MapDual"]:
        assert f"{listed['CoarseDual']} of {listed['MapN']}" not in h


def test_highlights_fit_elsevier_limit():
    items = [
        ln.split(r"\item", 1)[1].strip()
        for ln in (ROOT / "paper/highlights.tex").read_text().splitlines()
        if r"\item" in ln
    ]
    assert 3 <= len(items) <= 5
    for item in items:
        assert len(item) <= 85, (len(item), item)


def test_neumann_table_reports_percent():
    tex = (ROOT / "paper/generated_neumann.tex").read_text()
    listed = _macros_tex((ROOT / "paper/generated_numbers.tex").read_text())
    assert r"relative error (\%)" in tex
    assert listed["StefanRelErr"] in tex
    assert listed["NeumannMaxErr"] in tex


def test_baseline_claim_allows_annular_win():
    tex = (ROOT / "paper/main.tex").read_text()
    assert "consistently outperform equal-volume annular" not in tex
    assert r"J_{\mathrm{TO}}-J_{\mathrm{annular}}" in tex


def test_map_winners_are_named_classes():
    sweep = json.loads((ROOT / "results/sweep.json").read_text())
    allowed = {"melt", "freeze", "cycle", "dual", "tie"}
    for cell in sweep["cells"]:
        assert cell["winner"] in allowed
    assert len(sweep["cells"]) >= 4


def test_generated_map_matches_json():
    path = ROOT / "paper/generated_map.tex"
    if not path.exists():
        return
    sweep = json.loads((ROOT / "results/sweep.json").read_text())
    rows = [ln.strip() for ln in path.read_text().splitlines() if " & " in ln and "winner" not in ln]
    assert len(rows) == len(sweep["cells"])
    tex = (ROOT / "paper/main.tex").read_text()
    assert r"generated_map" in tex


def test_dual_rgb_paints_two_networks():
    from pcm.report import CHARGE_RGB, DISCHARGE_RGB, TUBE_RGB, dual_rgb, pcm_masked

    gc = np.zeros((6, 6))
    gd = np.zeros((6, 6))
    g = np.zeros((6, 6))
    gc[1, 1] = 1.0
    gd[4, 4] = 1.0
    g[1, 1] = 1.0
    g[4, 4] = 1.0
    g[2, 5] = 1.0
    rgb = dual_rgb(gc, gd, g)
    assert rgb.shape == (6, 6, 3)
    assert np.allclose(rgb[1, 1], CHARGE_RGB)
    assert np.allclose(rgb[4, 4], DISCHARGE_RGB)
    assert np.allclose(rgb[2, 5], TUBE_RGB)
    assert np.allclose(rgb[0, 0], (1.0, 1.0, 1.0))
    masked = pcm_masked(np.ones((6, 6)), g)
    assert np.isnan(masked[1, 1])
    assert masked[0, 0] == 1.0


def test_plot_dual_and_field_maps(tmp_path=None):
    from pathlib import Path

    from pcm.report import plot_dual_networks, plot_field_map, plot_map

    out = Path("/tmp/pcm-map-fig-test") if tmp_path is None else Path(tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    g = np.zeros((8, 8))
    g[3:5, 3:5] = 1.0
    gc = np.zeros((8, 8))
    gd = np.zeros((8, 8))
    gc[2, 2] = 0.9
    gd[6, 6] = 0.8
    g[2, 2] = 0.9
    g[6, 6] = 0.8
    cell = {
        "Lambda_target": 1.0,
        "Ra_d": 0.0,
        "winner": "dual",
        "dual_minus_cycle_J": 0.05,
        "winner_gamma": g.tolist(),
        "winner_dual": {"gamma_c": gc.tolist(), "gamma_d": gd.tolist()},
    }
    cells = [cell]
    plot_map(cells, out / "map.png")
    plot_dual_networks(cells, out / "dual.png")
    fields = {
        (1.0, 0.0): {
            "gamma": g,
            "Tf": np.linspace(-1, 1, 64).reshape(8, 8),
            "Tm": np.ones((8, 8)),
            "fl": 0.3 * (1.0 - g),
            "fm": 0.9 * (1.0 - g),
            "speed": 0.1 * np.ones((8, 8)),
            "um": np.zeros((8, 8)),
            "vm": np.zeros((8, 8)),
        }
    }
    plot_field_map(
        cells,
        fields,
        "Tf",
        out / "theta.png",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        cbar=r"$\theta$",
        title="test",
    )
    assert (out / "map.png").is_file()
    assert (out / "dual.png").is_file()
    assert (out / "theta.png").is_file()


if __name__ == "__main__":
    test_stefan_lambda_matches_carslaw()
    test_pick_winner_respects_margin()
    test_dual_tube_graphs_require_attachment()
    test_macros_match_generated_numbers()
    test_highlights_use_headline_dual_count()
    test_highlights_fit_elsevier_limit()
    test_neumann_table_reports_percent()
    test_baseline_claim_allows_annular_win()
    test_map_winners_are_named_classes()
    test_generated_map_matches_json()
    test_dual_rgb_paints_two_networks()
    test_plot_dual_and_field_maps()
    print("ok")
