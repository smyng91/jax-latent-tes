"""Adam topology optimization for melt, freeze, cycle, and dual-loop architectures."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from pcm.design import (
    density_from_logits,
    dual_from_logits,
    shift_dual_to_volume,
    shift_logits_to_volume,
    volume_fraction,
)
from pcm.physics import Geom, Sim, liquid_fraction, simulate, tube_bcs, two_tube_geom

Array = jnp.ndarray

BETAS = (2.0, 4.0, 8.0)
VOL_W = 200.0
T_FREEZE0 = 1.0
T_MELT0 = -1.0


def pcm_liquid(T: Array, gamma: Array, geom: Geom, eps: float) -> Array:
    w = geom.design.astype(T.dtype) * (1.0 - gamma)
    f = liquid_fraction(T, eps) * w
    return jnp.sum(f) / (jnp.sum(w) + 1e-12)


def run_freeze(gamma: Array, geom: Geom, sim: Sim) -> Array:
    T0 = jnp.full((sim.nx, sim.ny), T_FREEZE0)
    T, _, _ = simulate(T0, gamma, geom, sim, tube_bcs(charge_on=True, discharge_on=False))
    return T


def run_melt(gamma: Array, geom: Geom, sim: Sim) -> Array:
    T0 = jnp.full((sim.nx, sim.ny), T_MELT0)
    T, _, _ = simulate(T0, gamma, geom, sim, tube_bcs(charge_on=False, discharge_on=True))
    return T


def run_cycle(gamma: Array, geom: Geom, sim_c: Sim, sim_d: Sim) -> tuple[Array, Array]:
    T0 = jnp.full((sim_c.nx, sim_c.ny), T_FREEZE0)
    Tf, _, _ = simulate(T0, gamma, geom, sim_c, tube_bcs(charge_on=True, discharge_on=False))
    Tm, _, _ = simulate(Tf, gamma, geom, sim_d, tube_bcs(charge_on=False, discharge_on=True))
    return Tf, Tm


def vol_penalty(gamma: Array, geom: Geom, phi: float, w: float = VOL_W) -> Array:
    return w * (volume_fraction(gamma, geom) - phi) ** 2


def overlap_penalty(gc: Array, gd: Array, geom: Geom, w: float = 40.0) -> Array:
    return w * volume_fraction(gc * gd, geom)


def loss_melt(
    psi: Array, geom: Geom, sim: Sim, phi: float, beta: float = 4.0, n_blur: int = 2, vol_w: float = VOL_W
) -> Array:
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    T = run_melt(gamma, geom, sim)
    return (1.0 - pcm_liquid(T, gamma, geom, sim.eps)) + vol_penalty(gamma, geom, phi, vol_w)


def loss_freeze(
    psi: Array, geom: Geom, sim: Sim, phi: float, beta: float = 4.0, n_blur: int = 2, vol_w: float = VOL_W
) -> Array:
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    T = run_freeze(gamma, geom, sim)
    return pcm_liquid(T, gamma, geom, sim.eps) + vol_penalty(gamma, geom, phi, vol_w)


def loss_cycle(
    psi: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle(gamma, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gamma, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gamma, geom, sim_d.eps)
    return j_fr + j_m + vol_penalty(gamma, geom, phi, vol_w)


def loss_dual(
    psi_c: Array,
    psi_d: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    gc, gd, gamma = dual_from_logits(psi_c, psi_d, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle(gamma, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gamma, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gamma, geom, sim_d.eps)
    return j_fr + j_m + vol_penalty(gamma, geom, phi, vol_w) + overlap_penalty(gc, gd, geom)


def run_cycle_switched(gc: Array, gd: Array, geom: Geom, sim_c: Sim, sim_d: Sim) -> tuple[Array, Array]:
    """Charge uses only charge-metal; discharge uses only discharge-metal (reconfigurable)."""
    T0 = jnp.full((sim_c.nx, sim_c.ny), T_FREEZE0)
    Tf, _, _ = simulate(T0, gc, geom, sim_c, tube_bcs(charge_on=True, discharge_on=False))
    Tm, _, _ = simulate(Tf, gd, geom, sim_d, tube_bcs(charge_on=False, discharge_on=True))
    return Tf, Tm


def loss_dual_switched(
    psi_c: Array,
    psi_d: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    gc, gd, gamma = dual_from_logits(psi_c, psi_d, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle_switched(gc, gd, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gc, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gd, geom, sim_d.eps)
    return j_fr + j_m + vol_penalty(gamma, geom, phi, vol_w) + overlap_penalty(gc, gd, geom)


def loss_pareto(
    psi: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    w_fr: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle(gamma, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gamma, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gamma, geom, sim_d.eps)
    return w_fr * j_fr + (1.0 - w_fr) * j_m + vol_penalty(gamma, geom, phi, vol_w)


def loss_humid(
    psi: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    h: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    """Discharge residual weighted by (1+h); h is building latent/sensible cooling fraction."""
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle(gamma, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gamma, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gamma, geom, sim_d.eps)
    return j_fr + (1.0 + h) * j_m + vol_penalty(gamma, geom, phi, vol_w)


def loss_cop(
    psi: Array,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float,
    cop_w: float,
    beta: float = 4.0,
    n_blur: int = 2,
    vol_w: float = VOL_W,
) -> Array:
    """Carnot-lift proxy: colder charge (larger Ste_c) costs cop_w * Ste_c."""
    gamma = density_from_logits(psi, geom, beta=beta, n_blur=n_blur)
    Tf, Tm = run_cycle(gamma, geom, sim_c, sim_d)
    j_fr = pcm_liquid(Tf, gamma, geom, sim_c.eps)
    j_m = 1.0 - pcm_liquid(Tm, gamma, geom, sim_d.eps)
    return j_fr + j_m + cop_w * sim_c.ste + vol_penalty(gamma, geom, phi, vol_w)


def _adam_loop(loss_fn: Callable, params, n_iter: int, lr: float, match: Callable):
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    opt_state = opt.init(params)
    hist = []

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = match(params)
        return params, opt_state, loss

    for i in range(n_iter):
        params, opt_state, loss = step(params, opt_state)
        hist.append(float(loss))
        print(f"  iter {i:03d}  J={float(loss):.4f}", flush=True)
    return params, hist


def _fit_beta(loss_at_beta: Callable, params, n_iter: int, lr: float, match_at_beta: Callable, betas=BETAS):
    n_b = len(betas)
    chunks = [n_iter // n_b] * n_b
    chunks[-1] += n_iter - sum(chunks)
    hist: list[float] = []
    for beta, n in zip(betas, chunks):
        if n <= 0:
            continue
        params, h = _adam_loop(
            lambda p, b=beta: loss_at_beta(p, b),
            params,
            n,
            lr,
            lambda p, b=beta: match_at_beta(p, b),
        )
        hist += h
    return params, hist


def _match_single(params, geom, phi, beta, n_blur, enabled):
    if not enabled:
        return params
    return params + shift_logits_to_volume(params, geom, phi, beta, n_blur)


def _match_dual(params, geom, phi, beta, n_blur, enabled):
    if not enabled:
        return params
    c = shift_dual_to_volume(params[0], params[1], geom, phi, beta, n_blur)
    return params[0] + c, params[1] + c


def seed_psi(geom: Geom, phi: float, key: int = 0) -> Array:
    rng = jax.random.PRNGKey(key)
    noise = 0.08 * jax.random.normal(rng, geom.design.shape)
    near = (geom.charge | geom.discharge).astype(jnp.float32)
    near = near + jnp.roll(near, 1, 0) + jnp.roll(near, -1, 0)
    near = near + jnp.roll(near, 1, 1) + jnp.roll(near, -1, 1)
    return (-1.15 + noise + 0.2 * near) * geom.design.astype(jnp.float32)


def seed_dual(geom: Geom, which: str, key: int) -> Array:
    """Charge metal biased to the bottom tube; discharge metal to the top tube."""
    psi = seed_psi(geom, 0.05, key)
    ny = geom.design.shape[1]
    y = (jnp.arange(ny) + 0.5) / ny
    bias = 2.0 * ((y - 0.5) if which == "d" else (0.5 - y))
    return psi + bias[None, :] * geom.design.astype(psi.dtype)


def optimize_architecture(
    name: str,
    geom: Geom,
    sim_c: Sim,
    sim_d: Sim,
    phi: float = 0.1,
    n_iter: int = 15,
    lr: float = 0.08,
    w_fr: float = 0.5,
    humidity: float = 0.0,
    cop_w: float = 0.0,
    seed: int = 0,
    n_blur: int = 2,
    match_volume: bool = True,
    betas: tuple[float, ...] = BETAS,
    vol_w: float = VOL_W,
    warm: dict | None = None,
):
    print(f"TO {name} seed={seed}", flush=True)
    kw = dict(n_blur=n_blur, vol_w=vol_w)
    beta_end = float(betas[-1])

    def match_s(p, b):
        return _match_single(p, geom, phi, b, n_blur, match_volume)

    def match_d(p, b):
        return _match_dual(p, geom, phi, b, n_blur, match_volume)

    if name == "melt":
        psi = seed_psi(geom, phi, 1 + 17 * seed)
        psi, hist = _fit_beta(lambda p, b: loss_melt(p, geom, sim_d, phi, b, **kw), psi, n_iter, lr, match_s, betas)
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi}
    elif name == "freeze":
        psi = seed_psi(geom, phi, 2 + 17 * seed)
        psi, hist = _fit_beta(lambda p, b: loss_freeze(p, geom, sim_c, phi, b, **kw), psi, n_iter, lr, match_s, betas)
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi}
    elif name == "cycle":
        psi = seed_psi(geom, phi, 3 + 17 * seed)
        psi, hist = _fit_beta(
            lambda p, b: loss_cycle(p, geom, sim_c, sim_d, phi, b, **kw), psi, n_iter, lr, match_s, betas
        )
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi}
    elif name == "pareto":
        psi = seed_psi(geom, phi, 6 + 17 * seed)
        psi, hist = _fit_beta(
            lambda p, b: loss_pareto(p, geom, sim_c, sim_d, phi, w_fr, b, **kw), psi, n_iter, lr, match_s, betas
        )
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi, "w_fr": w_fr}
    elif name == "humid":
        psi = seed_psi(geom, phi, 7 + 17 * seed)
        psi, hist = _fit_beta(
            lambda p, b: loss_humid(p, geom, sim_c, sim_d, phi, humidity, b, **kw), psi, n_iter, lr, match_s, betas
        )
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi, "humidity": humidity}
    elif name == "cop":
        psi = seed_psi(geom, phi, 8 + 17 * seed)
        psi, hist = _fit_beta(
            lambda p, b: loss_cop(p, geom, sim_c, sim_d, phi, cop_w, b, **kw), psi, n_iter, lr, match_s, betas
        )
        gamma = density_from_logits(psi, geom, beta=beta_end, n_blur=n_blur)
        extras = {"psi": psi, "cop_w": cop_w}
    elif name == "dual":
        if warm is not None and "psi_c" in warm and "psi_d" in warm:
            params = (warm["psi_c"], warm["psi_d"])
        else:
            params = (seed_dual(geom, "c", 4 + 17 * seed), seed_dual(geom, "d", 5 + 17 * seed))
        params, hist = _fit_beta(
            lambda p, b: loss_dual(p[0], p[1], geom, sim_c, sim_d, phi, b, **kw),
            params,
            n_iter,
            lr,
            match_d,
            betas,
        )
        gc, gd, gamma = dual_from_logits(params[0], params[1], geom, beta=beta_end, n_blur=n_blur)
        extras = {"gamma_c": gc, "gamma_d": gd, "psi_c": params[0], "psi_d": params[1]}
    elif name == "dual_switched":
        params = (seed_dual(geom, "c", 9 + 17 * seed), seed_dual(geom, "d", 10 + 17 * seed))
        params, hist = _fit_beta(
            lambda p, b: loss_dual_switched(p[0], p[1], geom, sim_c, sim_d, phi, b, **kw),
            params,
            n_iter,
            lr,
            match_d,
            betas,
        )
        gc, gd, gamma = dual_from_logits(params[0], params[1], geom, beta=beta_end, n_blur=n_blur)
        extras = {"gamma_c": gc, "gamma_d": gd, "mode": "switched"}
    else:
        raise ValueError(name)
    return gamma, hist, extras


def cross_eval(gamma: Array, geom: Geom, sim_c: Sim, sim_d: Sim) -> dict:
    Tf = run_freeze(gamma, geom, sim_c)
    Tm = run_melt(gamma, geom, sim_d)
    Tf2, Tm2 = run_cycle(gamma, geom, sim_c, sim_d)
    return {
        "liquid_after_freeze": float(pcm_liquid(Tf, gamma, geom, sim_c.eps)),
        "liquid_after_melt": float(pcm_liquid(Tm, gamma, geom, sim_d.eps)),
        "cycle_liquid_after_freeze": float(pcm_liquid(Tf2, gamma, geom, sim_c.eps)),
        "cycle_liquid_after_melt": float(pcm_liquid(Tm2, gamma, geom, sim_d.eps)),
        "volume": float(volume_fraction(gamma, geom)),
        "metal_bottom": float(
            jnp.sum(gamma[:, : gamma.shape[1] // 2] * geom.design[:, : gamma.shape[1] // 2])
            / (jnp.sum(geom.design[:, : gamma.shape[1] // 2]) + 1e-12)
        ),
        "metal_top": float(
            jnp.sum(gamma[:, gamma.shape[1] // 2 :] * geom.design[:, gamma.shape[1] // 2 :])
            / (jnp.sum(geom.design[:, gamma.shape[1] // 2 :]) + 1e-12)
        ),
    }


def default_geom(nx: int, ny: int) -> Geom:
    return two_tube_geom(nx, ny)
