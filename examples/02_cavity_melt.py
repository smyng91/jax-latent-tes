"""Buoyancy trend in a square cavity. Not a Gau–Viskanta validation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pcm.physics import default_sim, empty_geom, liquid_fraction, simulate, slab_bcs

jax.config.update("jax_enable_x64", True)


def run(ra, flow, nx=32, n_steps=40, dt=0.002):
    sim = default_sim(
        nx, nx, dt=dt, n_steps=n_steps, ste=0.1, ra=ra, kappa=1.0, flow=flow, n_jacobi_t=40, n_jacobi_p=40
    )
    T, u, v = simulate(
        jnp.full((nx, nx), -0.8), jnp.zeros((nx, nx)), empty_geom(nx, nx), sim, slab_bcs(T_left=1.0)
    )
    return np.asarray(T), np.asarray(u), np.asarray(v), sim


def main() -> None:
    T0, _, _, sim = run(0.0, False)
    T1, u1, v1, _ = run(1e4, True)
    f0 = float(np.mean(np.asarray(liquid_fraction(T0, sim.eps))))
    f1 = float(np.mean(np.asarray(liquid_fraction(T1, sim.eps))))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, T, title in zip(axes, [T0, T1], [f"Ra=0, f={f0:.3f}", f"Ra=$10^4$, f={f1:.3f}"]):
        im = ax.imshow(T.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1, extent=[0, 1, 0, 1])
        ax.set_title(title)
        ax.set_xlabel("$x/L$")
    axes[0].set_ylabel("$y/L$")
    fig.colorbar(im, ax=axes, fraction=0.046, label=r"$\theta$")
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cavity_ra.png", dpi=200, bbox_inches="tight")
    print(f"f(Ra=0)={f0:.4f} f(Ra=1e4)={f1:.4f} ratio={f1/f0:.3f}")
    print(f"u_max={np.max(np.abs(u1)):.3e} v_max={np.max(np.abs(v1)):.3e}")
    print("wrote paper/figures/cavity_ra.png")


if __name__ == "__main__":
    main()
