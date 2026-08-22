"""Two-phase Neumann slab. Writes paper/figures/stefan.png and prints measured error."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcm.physics import default_sim, empty_geom, liquid_fraction, simulate, slab_bcs
from pcm.published import neumann_lambda
from pcm.validate import (
    NEUMANN_DT,
    NEUMANN_EPS,
    NEUMANN_FO_HEAD,
    NEUMANN_JACOBI,
    NEUMANN_NX,
    NEUMANN_NY,
    NEUMANN_T0,
    interface_x,
)
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main() -> None:
    ste, fo, t0, eps = 0.2, NEUMANN_FO_HEAD, NEUMANN_T0, NEUMANN_EPS
    n_steps = int(round(fo / NEUMANN_DT))
    nx, ny = NEUMANN_NX, NEUMANN_NY
    sim = default_sim(
        nx, ny, dt=NEUMANN_DT, n_steps=n_steps, ste=ste, kappa=1.0, flow=False, n_jacobi_t=NEUMANN_JACOBI, eps=eps
    )
    T, _, _ = simulate(jnp.full((nx, ny), t0), jnp.zeros((nx, ny)), empty_geom(nx, ny), sim, slab_bcs(T_left=1.0))
    T = np.asarray(T)
    f = np.asarray(liquid_fraction(T, sim.eps))
    x = (np.arange(nx) + 0.5) / nx
    lam = neumann_lambda(ste, ste * abs(t0))
    s_ex = 2 * lam * fo**0.5
    s_nu = interface_x(T, sim.eps, 1.0, nx)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(x, f.mean(axis=1), color="black", label="mean liquid fraction")
    ax.axvline(s_ex, color="0.4", ls="--", label=f"two-phase Neumann $s/L={s_ex:.3f}$")
    ax.axvline(s_nu, color="black", ls=":", label=f"numeric $s/L={s_nu:.3f}$")
    ax.set_xlabel("$x/L$")
    ax.set_ylabel("liquid fraction")
    ax.set_title(f"Ste={ste}, Fo={fo}, relative error={abs(s_nu-s_ex)/s_ex:.2%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stefan.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    print(f"s_exact={s_ex:.4f} s_numeric={s_nu:.4f} rel={abs(s_nu-s_ex)/s_ex:.4f}")
    print(f"wrote {out / 'stefan.png'}")


if __name__ == "__main__":
    main()
