"""Melt-only vs freeze-only TO on one (Λ, Ra_d) cell. Saves gamma maps from this run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcm.optimize import cross_eval, optimize_architecture
from pcm.physics import default_sim, two_tube_geom


def main() -> None:
    nx = ny = 24
    n_iter = 8
    phi = 0.1
    fo_d, dt = 0.03, 0.002
    n_d = int(round(fo_d / dt))
    geom = two_tube_geom(nx, ny)
    sim_c = default_sim(nx, ny, dt=dt, n_steps=n_d, ste=0.1, ra=0.0, kappa=20.0, flow=False, n_jacobi_t=30)
    sim_d = default_sim(
        nx, ny, dt=dt, n_steps=n_d, ste=0.1, ra=1e4, kappa=20.0, flow=True, n_jacobi_t=30, n_jacobi_p=30
    )
    print(f"Fo_c={sim_c.n_steps*sim_c.dt:.4f} Fo_d={sim_d.n_steps*sim_d.dt:.4f} Ra_d={sim_d.ra:g}")
    maps = {}
    metrics = {}
    for name in ("melt", "freeze"):
        gamma, hist, _ = optimize_architecture(name, geom, sim_c, sim_d, phi=phi, n_iter=n_iter)
        maps[name] = np.asarray(gamma)
        metrics[name] = cross_eval(gamma, geom, sim_c, sim_d)
        metrics[name]["loss_final"] = hist[-1]
        print(name, metrics[name])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, name in zip(axes, ("melt", "freeze")):
        ax.imshow(maps[name].T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=[0, 1, 0, 1])
        m = metrics[name]
        ax.set_title(
            f"{name}\nvol={m['volume']:.3f} top={m['metal_top']:.3f} bot={m['metal_bottom']:.3f}"
        )
        ax.set_xlabel("$x/L$")
    axes[0].set_ylabel("$y/L$")
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "melt_vs_freeze_to.png", dpi=200)
    Path("results").mkdir(exist_ok=True)
    np.savez("results/melt_vs_freeze.npz", melt=maps["melt"], freeze=maps["freeze"])
    print("wrote paper/figures/melt_vs_freeze_to.png")


if __name__ == "__main__":
    main()
