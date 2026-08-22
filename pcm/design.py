"""Density filter, tanh projection, and SIMP interpolation."""

import jax
import jax.numpy as jnp

from pcm.physics import Geom

Array = jnp.ndarray


def _smooth(rho: Array) -> Array:
    """3x3 binomial blur with Neumann edges."""
    k = jnp.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    p = jnp.pad(rho, 1, mode="edge")
    out = (
        k[0, 0] * p[:-2, :-2]
        + k[0, 1] * p[:-2, 1:-1]
        + k[0, 2] * p[:-2, 2:]
        + k[1, 0] * p[1:-1, :-2]
        + k[1, 1] * p[1:-1, 1:-1]
        + k[1, 2] * p[1:-1, 2:]
        + k[2, 0] * p[2:, :-2]
        + k[2, 1] * p[2:, 1:-1]
        + k[2, 2] * p[2:, 2:]
    )
    return out


def filter_density(rho: Array, n_blur: int = 2) -> Array:
    def body(_, r):
        return _smooth(r)

    return jax.lax.fori_loop(0, n_blur, body, rho)


def project_density(rho: Array, beta: float = 8.0, eta: float = 0.5) -> Array:
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


def density_from_logits(psi: Array, geom: Geom, beta: float = 4.0, n_blur: int = 2) -> Array:
    rho = jax.nn.sigmoid(filter_density(psi, n_blur=n_blur))
    gamma = project_density(rho, beta=beta)
    return jnp.where(geom.design, gamma, 1.0)


def dual_from_logits(
    psi_c: Array, psi_d: Array, geom: Geom, beta: float = 4.0, n_blur: int = 2
) -> tuple[Array, Array, Array]:
    """Three-phase softmax on filtered logits: charge metal, discharge metal, PCM."""
    zc = filter_density(psi_c, n_blur=n_blur)
    zd = filter_density(psi_d, n_blur=n_blur)
    zp = jnp.zeros_like(zc)
    w = jax.nn.softmax(jnp.stack([zc, zd, zp], axis=0), axis=0)
    gc = project_density(w[0], beta=beta)
    gd = project_density(w[1], beta=beta)
    gc = jnp.where(geom.design, gc, 0.0)
    gd = jnp.where(geom.design, gd, 0.0)
    gamma = jnp.clip(gc + gd, 0.0, 1.0)
    gamma = jnp.where(geom.design, gamma, 1.0)
    return gc, gd, gamma


def volume_fraction(gamma: Array, geom: Geom) -> Array:
    w = geom.design.astype(gamma.dtype)
    return jnp.sum(gamma * w) / jnp.sum(w)


def shift_logits_to_volume(psi: Array, geom: Geom, phi: float, beta: float, n_blur: int = 2) -> Array:
    """Detached constant added to logits so projected volume matches phi."""

    def vol(c):
        return volume_fraction(density_from_logits(psi + c, geom, beta=beta, n_blur=n_blur), geom)

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        too_small = vol(mid) < phi
        lo = jnp.where(too_small, mid, lo)
        hi = jnp.where(too_small, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 18, body, (jnp.array(-6.0), jnp.array(6.0)))
    return jax.lax.stop_gradient(0.5 * (lo + hi))


def shift_dual_to_volume(
    psi_c: Array, psi_d: Array, geom: Geom, phi: float, beta: float, n_blur: int = 2
) -> Array:
    def vol(c):
        _, _, g = dual_from_logits(psi_c + c, psi_d + c, geom, beta=beta, n_blur=n_blur)
        return volume_fraction(g, geom)

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        too_small = vol(mid) < phi
        lo = jnp.where(too_small, mid, lo)
        hi = jnp.where(too_small, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 18, body, (jnp.array(-6.0), jnp.array(6.0)))
    return jax.lax.stop_gradient(0.5 * (lo + hi))
