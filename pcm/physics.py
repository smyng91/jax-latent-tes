"""2D enthalpy-porosity with vorticity–stream-function Boussinesq NS.

Dimensionless form uses L = domain height, t_ref = L^2/alpha_PCM, theta = (T-T_m)/DeltaT
with independent Ste_c, Ste_d on charge and discharge windows. Outer walls are insulated unless a slab test sets
Dirichlet. Tubes are Dirichlet when the corresponding loop is active.

Momentum is unsteady incompressible NS in stream-function/vorticity form with
Kozeny–Carman mushy drag (enthalpy-porosity). Pressure is never formed.
Poisson/Helmholtz solves use implicit VJPs (custom_linear_solve), not unrolled CG.
Temperature advection is implicit upwind in the diffusion Jacobi. Vorticity advection
stays explicit at CFL 1; energy uses a detached CFL 8 so a buoyancy gyre can melt
without the CFL-32 over-melt documented in the validation suite.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

Array = jnp.ndarray


class Sim(NamedTuple):
    nx: int
    ny: int
    Lx: float
    Ly: float
    dt: float
    n_steps: int
    ste: float
    ra: float
    pr: float
    kappa: float
    eps: float
    simp_p: float
    mush: float
    n_jacobi_t: int
    n_jacobi_p: int
    flow: bool
    energy_cfl: float
    vorticity_cfl: float


class BCs(NamedTuple):
    left_dirichlet: bool
    right_dirichlet: bool
    T_left: float
    T_right: float
    T_charge: float
    T_discharge: float
    charge_on: bool
    discharge_on: bool


class Geom(NamedTuple):
    charge: Array
    discharge: Array
    design: Array


def two_tube_geom(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0, r: float = 0.08) -> Geom:
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    charge = (X - 0.5 * Lx) ** 2 + (Y - 0.22 * Ly) ** 2 <= r**2
    discharge = (X - 0.5 * Lx) ** 2 + (Y - 0.78 * Ly) ** 2 <= r**2
    tubes = charge | discharge
    return Geom(
        charge=jnp.asarray(charge),
        discharge=jnp.asarray(discharge),
        design=jnp.asarray(~tubes),
    )


def empty_geom(nx: int, ny: int) -> Geom:
    z = jnp.zeros((nx, ny), dtype=bool)
    return Geom(charge=z, discharge=z, design=jnp.ones((nx, ny), dtype=bool))


def liquid_fraction(T: Array, eps: float) -> Array:
    return 0.5 * (1.0 + jnp.tanh(T / eps))


def conductivity(gamma: Array, kappa: float, simp_p: float) -> Array:
    return 1.0 + (kappa - 1.0) * gamma**simp_p


def _face_k_x(k: Array) -> Array:
    return 2.0 * k[1:] * k[:-1] / (k[1:] + k[:-1] + 1e-12)


def _face_k_y(k: Array) -> Array:
    return 2.0 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1] + 1e-12)


def _neighbor_T(T: Array, geom: Geom, bcs: BCs) -> tuple[Array, Array, Array, Array]:
    """East, west, north, south neighbor temperatures (same shape as T)."""
    Tbc = apply_dirichlet(T, geom, bcs)
    Te = jnp.concatenate([Tbc[1:], Tbc[-1:]], axis=0)
    Tw = jnp.concatenate([Tbc[:1], Tbc[:-1]], axis=0)
    Tn = jnp.concatenate([Tbc[:, 1:], Tbc[:, -1:]], axis=1)
    Ts = jnp.concatenate([Tbc[:, :1], Tbc[:, :-1]], axis=1)
    if bcs.left_dirichlet:
        Tw = Tw.at[0].set(bcs.T_left)
    if bcs.right_dirichlet:
        Te = Te.at[-1].set(bcs.T_right)
    return Te, Tw, Tn, Ts


def apply_dirichlet(T: Array, geom: Geom, bcs: BCs) -> Array:
    T = jnp.where(bcs.charge_on & geom.charge, bcs.T_charge, T)
    T = jnp.where(bcs.discharge_on & geom.discharge, bcs.T_discharge, T)
    return T


def implicit_temperature(
    T: Array,
    gamma: Array,
    u: Array,
    v: Array,
    geom: Geom,
    sim: Sim,
    bcs: BCs,
) -> Array:
    dx = sim.Lx / sim.nx
    dy = sim.Ly / sim.ny
    T = apply_dirichlet(T, geom, bcs)
    k = conductivity(gamma, sim.kappa, sim.simp_p)
    f = liquid_fraction(T, sim.eps) * (1.0 - gamma)
    sech2 = 1.0 - jnp.tanh(T / sim.eps) ** 2
    dfdT = (0.5 / sim.eps) * sech2 * (1.0 - gamma)
    C = 1.0 + dfdT / sim.ste
    a = C / sim.dt
    ke = _face_k_x(k)
    kn = _face_k_y(k)
    # Pad face k with boundary faces using cell k.
    ke_w = jnp.concatenate([k[:1], ke], axis=0)
    ke_e = jnp.concatenate([ke, k[-1:]], axis=0)
    kn_s = jnp.concatenate([k[:, :1], kn], axis=1)
    kn_n = jnp.concatenate([kn, k[:, -1:]], axis=1)
    # Insulated outer faces: zero flux => zero that face k in the stencil.
    if not bcs.left_dirichlet:
        ke_w = ke_w.at[0].set(0.0)
    else:
        ke_w = ke_w.at[0].set(k[0])
    if not bcs.right_dirichlet:
        ke_e = ke_e.at[-1].set(0.0)
    else:
        ke_e = ke_e.at[-1].set(k[-1])
    kn_s = kn_s.at[:, 0].set(0.0)
    kn_n = kn_n.at[:, -1].set(0.0)

    # Implicit first-order upwind: lagged u,v, live T in the Jacobi.
    aw = jnp.maximum(u, 0.0) / dx
    ae = jnp.maximum(-u, 0.0) / dx
    as_ = jnp.maximum(v, 0.0) / dy
    an = jnp.maximum(-v, 0.0) / dy
    adv_diag = aw + ae + as_ + an
    rhs0 = a * T

    dirichlet_cell = jnp.zeros_like(T, dtype=bool)
    if bcs.charge_on:
        dirichlet_cell = dirichlet_cell | geom.charge
    if bcs.discharge_on:
        dirichlet_cell = dirichlet_cell | geom.discharge

    dx2 = dx * dx
    dy2 = dy * dy
    # Dirichlet walls: flux 2 k (T - Twall) / dx  => extra diag 2k/dx2
    extra_diag = jnp.zeros_like(T)
    extra_rhs = jnp.zeros_like(T)
    if bcs.left_dirichlet:
        extra_diag = extra_diag.at[0].add(2.0 * k[0] / dx2)
        extra_rhs = extra_rhs.at[0].add(2.0 * k[0] / dx2 * bcs.T_left)
        ke_w = ke_w.at[0].set(0.0)
    if bcs.right_dirichlet:
        extra_diag = extra_diag.at[-1].add(2.0 * k[-1] / dx2)
        extra_rhs = extra_rhs.at[-1].add(2.0 * k[-1] / dx2 * bcs.T_right)
        ke_e = ke_e.at[-1].set(0.0)

    diag = a + ke_e / dx2 + ke_w / dx2 + kn_n / dy2 + kn_s / dy2 + extra_diag + adv_diag
    rhs0 = rhs0 + extra_rhs

    def body(_, Tcur):
        Tcur = apply_dirichlet(Tcur, geom, bcs)
        Te, Tw, Tn, Ts = _neighbor_T(Tcur, geom, bcs)
        num = (
            rhs0
            + ke_e * Te / dx2
            + ke_w * Tw / dx2
            + kn_n * Tn / dy2
            + kn_s * Ts / dy2
            + ae * Te
            + aw * Tw
            + an * Tn
            + as_ * Ts
        )
        Tnew = num / (diag + 1e-12)
        Tnew = jnp.where(dirichlet_cell, Tcur, Tnew)
        return apply_dirichlet(Tnew, geom, bcs)

    return jax.lax.fori_loop(0, sim.n_jacobi_t, body, T)


def _grad(f: Array, dx: float, dy: float) -> tuple[Array, Array]:
    fe = jnp.concatenate([f[1:], f[-1:]], 0)
    fw = jnp.concatenate([f[:1], f[:-1]], 0)
    fn = jnp.concatenate([f[:, 1:], f[:, -1:]], 1)
    fs = jnp.concatenate([f[:, :1], f[:, :-1]], 1)
    return 0.5 * (fe - fw) / dx, 0.5 * (fn - fs) / dy


def _upwind_grad(q: Array, u: Array, v: Array, dx: float, dy: float) -> tuple[Array, Array]:
    qe = jnp.concatenate([q[1:], q[-1:]], 0)
    qw = jnp.concatenate([q[:1], q[:-1]], 0)
    qn = jnp.concatenate([q[:, 1:], q[:, -1:]], 1)
    qs = jnp.concatenate([q[:, :1], q[:, :-1]], 1)
    dqx = jnp.where(u >= 0.0, (q - qw) / dx, (qe - q) / dx)
    dqy = jnp.where(v >= 0.0, (q - qs) / dy, (qn - q) / dy)
    return dqx, dqy


def _locked(solid: Array) -> Array:
    locked = solid
    locked = locked.at[0].set(True).at[-1].set(True)
    locked = locked.at[:, 0].set(True).at[:, -1].set(True)
    return jax.lax.stop_gradient(locked)


def _spd_laplace(phi: Array, locked: Array, dx2: float, dy2: float) -> Array:
    """-∇² with identity rows on locked cells (Dirichlet)."""
    diag = 2.0 / dx2 + 2.0 / dy2
    pe = jnp.concatenate([phi[1:], phi[-1:]], 0)
    pw = jnp.concatenate([phi[:1], phi[:-1]], 0)
    pn = jnp.concatenate([phi[:, 1:], phi[:, -1:]], 1)
    ps = jnp.concatenate([phi[:, :1], phi[:, :-1]], 1)
    out = diag * phi - pe / dx2 - pw / dx2 - pn / dy2 - ps / dy2
    return jnp.where(locked, phi, out)


def _linear_solve(matvec, b: Array, maxiter: int, precond=None) -> Array:
    def solver(mv, rhs):
        x, _ = cg(mv, rhs, M=precond, maxiter=maxiter, tol=1e-6)
        return x

    return jax.lax.custom_linear_solve(matvec, b, solver, solver, symmetric=True)


def _poisson(rhs: Array, locked: Array, dx2: float, dy2: float, maxiter: int) -> Array:
    b = jnp.where(locked, 0.0, rhs)
    shape = rhs.shape
    invdiag = jnp.where(locked, 1.0, 1.0 / (2.0 / dx2 + 2.0 / dy2))

    def matvec(x):
        return _spd_laplace(x.reshape(shape), locked, dx2, dy2).ravel()

    def precond(x):
        return (x.reshape(shape) * invdiag).ravel()

    phi = _linear_solve(matvec, b.ravel(), maxiter, precond).reshape(shape)
    return jnp.where(locked, 0.0, phi)


def _helmholtz(
    rhs: Array,
    locked: Array,
    lam: Array,
    dx2: float,
    dy2: float,
    dt_pr: float,
    maxiter: int,
) -> Array:
    """(I + dt Pr λ I − dt Pr ∇²) ω = rhs, Dirichlet on locked cells."""
    b = jnp.where(locked, 0.0, rhs)
    shape = rhs.shape
    invdiag = jnp.where(locked, 1.0, 1.0 / (1.0 + dt_pr * lam + dt_pr * (2.0 / dx2 + 2.0 / dy2)))

    def matvec(x):
        phi = x.reshape(shape)
        visc = _spd_laplace(phi, locked, dx2, dy2)
        return jnp.where(locked, phi, phi + dt_pr * lam * phi + dt_pr * visc).ravel()

    def precond(x):
        return (x.reshape(shape) * invdiag).ravel()

    omega = _linear_solve(matvec, b.ravel(), maxiter, precond).reshape(shape)
    return jnp.where(locked, 0.0, omega)


def ns_boussinesq(
    T: Array,
    omega: Array,
    u: Array,
    v: Array,
    gamma: Array,
    geom: Geom,
    sim: Sim,
) -> tuple[Array, Array, Array]:
    """Unsteady NS: ∂ω/∂t + u·∇ω = Pr ∇²ω + Pr Ra ∂θ/∂x − Pr λ ω, ∇²ψ = −ω.

    λ = mush (1−f)² / (f³+ε) is Kozeny–Carman drag. Fully liquid (f=1) is NS–Boussinesq.
    Solid is damped by λ, not a discrete freeze mask. CFL scale is detached so reverse-mode
    does not see the clip; Poisson/Helmholtz use implicit VJPs, not unrolled CG.
    """
    dx = sim.Lx / sim.nx
    dy = sim.Ly / sim.ny
    dx2, dy2 = dx * dx, dy * dy
    f = liquid_fraction(T, sim.eps) * (1.0 - gamma)
    tubes = geom.charge | geom.discharge
    locked = _locked(tubes)
    lam = sim.mush * (1.0 - f) ** 2 / (f**3 + 1e-3)
    dqx, dqy = _upwind_grad(omega, u, v, dx, dy)
    adv = u * dqx + v * dqy
    dTdx, _ = _grad(T, dx, dy)
    buoy = sim.pr * sim.ra * dTdx
    rhs = omega + sim.dt * (buoy - adv)
    omega = _helmholtz(rhs, locked, lam, dx2, dy2, sim.dt * sim.pr, sim.n_jacobi_p)
    psi = _poisson(omega, locked, dx2, dy2, sim.n_jacobi_p)
    dpsidx, dpsidy = _grad(psi, dx, dy)
    u = jnp.where(locked, 0.0, dpsidy)
    v = jnp.where(locked, 0.0, -dpsidx)
    return u, v, omega


def step(
    T: Array,
    u: Array,
    v: Array,
    omega: Array,
    gamma: Array,
    geom: Geom,
    sim: Sim,
    bcs: BCs,
) -> tuple[Array, Array, Array, Array]:
    T = apply_dirichlet(T, geom, bcs)
    if sim.flow:
        u, v, omega = ns_boussinesq(T, omega, u, v, gamma, geom, sim)
        dx = sim.Lx / sim.nx
        dy = sim.Ly / sim.ny
        umax = jnp.max(jnp.sqrt(u * u + v * v))
        h = jnp.minimum(dx, dy)
        # Explicit vorticity advection needs CFL~1; energy is implicit-upwind and can take more.
        fac_w = jnp.minimum(1.0, (sim.vorticity_cfl * h / sim.dt) / (umax + 1e-12))
        fac_e = jnp.minimum(1.0, (sim.energy_cfl * h / sim.dt) / (umax + 1e-12))
        fac_w = jax.lax.stop_gradient(fac_w)
        fac_e = jax.lax.stop_gradient(fac_e)
        T = implicit_temperature(T, gamma, u * fac_e, v * fac_e, geom, sim, bcs)
        u, v = u * fac_w, v * fac_w
    else:
        u, v, omega = jnp.zeros_like(T), jnp.zeros_like(T), jnp.zeros_like(T)
        T = implicit_temperature(T, gamma, u, v, geom, sim, bcs)
    return T, u, v, omega


def enthalpy(T: Array, gamma: Array, sim: Sim) -> Array:
    f = liquid_fraction(T, sim.eps) * (1.0 - gamma)
    return T + f / sim.ste


@partial(jax.jit, static_argnames=("sim", "bcs"))
def simulate(
    T0: Array,
    gamma: Array,
    geom: Geom,
    sim: Sim,
    bcs: BCs,
) -> tuple[Array, Array, Array]:
    u0 = jnp.zeros_like(T0)
    v0 = jnp.zeros_like(T0)
    w0 = jnp.zeros_like(T0)
    T0 = apply_dirichlet(T0, geom, bcs)

    def body(state, _):
        def inner(state):
            T, u, v, omega = state
            return step(T, u, v, omega, gamma, geom, sim, bcs)

        return jax.checkpoint(inner)(state), None

    (T, u, v, _omega), _ = jax.lax.scan(body, (T0, u0, v0, w0), None, length=sim.n_steps)
    return T, u, v


@partial(jax.jit, static_argnames=("sim", "bcs"))
def simulate_liquid_trace(
    T0: Array,
    gamma: Array,
    geom: Geom,
    sim: Sim,
    bcs: BCs,
) -> Array:
    """PCM-averaged liquid fraction after every step (design cells if present)."""
    u0 = jnp.zeros_like(T0)
    v0 = jnp.zeros_like(T0)
    w0 = jnp.zeros_like(T0)
    T0 = apply_dirichlet(T0, geom, bcs)
    w = geom.design.astype(T0.dtype) * (1.0 - gamma)
    den = jnp.sum(w) + 1e-12

    def body(state, _):
        def inner(state):
            T, u, v, omega = state
            return step(T, u, v, omega, gamma, geom, sim, bcs)

        T, u, v, omega = jax.checkpoint(inner)(state)
        f = liquid_fraction(T, sim.eps) * w
        return (T, u, v, omega), jnp.sum(f) / den

    _, hist = jax.lax.scan(body, (T0, u0, v0, w0), None, length=sim.n_steps)
    return hist


@partial(jax.jit, static_argnames=("sim", "bcs"))
def simulate_energy(
    T0: Array,
    gamma: Array,
    geom: Geom,
    sim: Sim,
    bcs: BCs,
) -> tuple[Array, Array, Array]:
    """Final field, ΔH, and ∫q dt (dimensionless) for a discrete energy check."""
    u0 = jnp.zeros_like(T0)
    v0 = jnp.zeros_like(T0)
    w0 = jnp.zeros_like(T0)
    T0 = apply_dirichlet(T0, geom, bcs)
    dx = sim.Lx / sim.nx
    dy = sim.Ly / sim.ny
    h0 = jnp.sum(enthalpy(T0, gamma, sim)) * dx * dy

    def body(state, _):
        def inner(state):
            T, u, v, omega, acc = state
            q = net_boundary_flux(T, gamma, geom, sim, bcs)
            T, u, v, omega = step(T, u, v, omega, gamma, geom, sim, bcs)
            return T, u, v, omega, acc + sim.dt * q

        return jax.checkpoint(inner)(state), None

    (T, _u, _v, _omega, acc), _ = jax.lax.scan(body, (T0, u0, v0, w0, 0.0), None, length=sim.n_steps)
    h1 = jnp.sum(enthalpy(T, gamma, sim)) * dx * dy
    return T, h1 - h0, acc


def net_boundary_flux(T: Array, gamma: Array, geom: Geom, sim: Sim, bcs: BCs) -> Array:
    """Dimensionless heat input into PCM (walls + active tubes)."""
    dx = sim.Lx / sim.nx
    dy = sim.Ly / sim.ny
    k = conductivity(gamma, sim.kappa, sim.simp_p)
    flux = 0.0
    if bcs.left_dirichlet:
        flux = flux + jnp.sum(2.0 * k[0] * (bcs.T_left - T[0]) / dx * dy)
    if bcs.right_dirichlet:
        flux = flux + jnp.sum(2.0 * k[-1] * (bcs.T_right - T[-1]) / dx * dy)
    active = jnp.zeros_like(T, dtype=bool)
    Twall = jnp.zeros_like(T)
    if bcs.charge_on:
        active = active | geom.charge
        Twall = jnp.where(geom.charge, bcs.T_charge, Twall)
    if bcs.discharge_on:
        active = active | geom.discharge
        Twall = jnp.where(geom.discharge, bcs.T_discharge, Twall)
    pcm = (~active) & geom.design
    ke = _face_k_x(k)
    kn = _face_k_y(k)
    # Face flux into the PCM cell from an active-tube neighbor.
    west_from_tube = active[:-1] & pcm[1:]
    east_from_tube = active[1:] & pcm[:-1]
    south_from_tube = active[:, :-1] & pcm[:, 1:]
    north_from_tube = active[:, 1:] & pcm[:, :-1]
    flux = flux + jnp.sum(ke * (Twall[:-1] - T[1:]) / dx * dy * west_from_tube)
    flux = flux + jnp.sum(ke * (Twall[1:] - T[:-1]) / dx * dy * east_from_tube)
    flux = flux + jnp.sum(kn * (Twall[:, :-1] - T[:, 1:]) / dy * dx * south_from_tube)
    flux = flux + jnp.sum(kn * (Twall[:, 1:] - T[:, :-1]) / dy * dx * north_from_tube)
    return flux


def default_sim(
    nx: int = 48,
    ny: int = 48,
    *,
    dt: float = 2e-3,
    n_steps: int = 40,
    ste: float = 0.1,
    ra: float = 0.0,
    pr: float = 1.0,
    kappa: float = 20.0,
    flow: bool = False,
    n_jacobi_t: int = 40,
    n_jacobi_p: int = 64,
    Lx: float = 1.0,
    Ly: float = 1.0,
    mush: float = 1.0e5,
    eps: float = 0.05,
    energy_cfl: float = 8.0,
    vorticity_cfl: float = 1.0,
) -> Sim:
    return Sim(
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        n_steps=n_steps,
        ste=ste,
        ra=ra,
        pr=pr,
        kappa=kappa,
        eps=eps,
        simp_p=3.0,
        mush=mush,
        n_jacobi_t=n_jacobi_t,
        n_jacobi_p=n_jacobi_p,
        flow=flow,
        energy_cfl=energy_cfl,
        vorticity_cfl=vorticity_cfl,
    )


def tube_bcs(*, charge_on: bool, discharge_on: bool, T_charge: float = -1.0, T_discharge: float = 1.0) -> BCs:
    return BCs(
        left_dirichlet=False,
        right_dirichlet=False,
        T_left=0.0,
        T_right=0.0,
        T_charge=T_charge,
        T_discharge=T_discharge,
        charge_on=charge_on,
        discharge_on=discharge_on,
    )


def slab_bcs(*, T_left: float = 1.0, T_right: float | None = None) -> BCs:
    return BCs(
        left_dirichlet=True,
        right_dirichlet=T_right is not None,
        T_left=T_left,
        T_right=0.0 if T_right is None else T_right,
        T_charge=-1.0,
        T_discharge=1.0,
        charge_on=False,
        discharge_on=False,
    )
