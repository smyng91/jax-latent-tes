"""Published constants used for validation. Do not invent experimental digits.

Sources
-------
Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., Oxford, 1959.
    One-phase Neumann/Stefan solution: s = 2 λ √(α t) with
    λ √π erf(λ) exp(λ²) = Ste.

Gau & Viskanta, *J. Heat Transfer* 108, 174–181 (1986).
    Gallium melting on a vertical wall. Early melting (~2 min) is
    conduction-dominated (V/V0 ∝ τ^{1/2}). Near the end of their tests,
    melted volume is about 1.75 times the Neumann (no-convection) value.
    Interface traces at 2, 6, 10, 17 min are graphical in their Fig. 2;
    we do not digitize unpublished point lists.

Brent, Voller & Reid, *Numer. Heat Transfer* 13, 297–318 (1988), Table 1
    and Fig. 1: property set and cavity used to reproduce Gau–Viskanta.
    X = 8.89 cm (width), Y = 6.35 cm (height), T_hot = 38°C, T_init = 28.3°C,
    T_m = 29.78°C, Ste = 0.039, Ra = 6e5, Pr = 0.0216.
"""

import math

# Brent et al. (1988) Table 1, SI.
GALLIUM = {
    "rho": 6093.0,  # kg/m^3
    "cp": 381.5,  # J/kg/K
    "k": 32.0,  # W/m/K
    "mu": 1.81e-3,  # Pa s
    "beta": 1.2e-4,  # 1/K
    "L_lat": 80160.0,  # J/kg
    "T_m": 29.78,  # deg C
    "T_hot": 38.0,
    "T_init": 28.3,
    "W": 0.0889,  # m, cavity width (Brent X)
    "H": 0.0635,  # m, cavity height (Brent Y)
    "Ste": 0.039,
    "Ra": 6.0e5,
    "Pr": 0.0216,
}

# Gau & Viskanta (1986), prose results we cite without digitizing figures.
GAU_VISKANTA_1986 = {
    "conduction_window_min": 2.0,
    "late_time_volume_over_neumann": 1.75,
    "late_time_min": 17.0,
}


def stefan_lambda(ste: float) -> float:
    """Root of λ√π erf(λ) exp(λ²) = Ste (one-phase Neumann, solid at fusion)."""
    return neumann_lambda(ste, 0.0)


def neumann_lambda(ste_l: float, ste_s: float = 0.0) -> float:
    """Two-phase Neumann λ with equal liquid/solid diffusivity (Carslaw & Jaeger).

    Wall superheat Ste_l = c_p (T_w − T_m)/L_lat, initial subcooling
    Ste_s = c_p (T_m − T_i)/L_lat. Interface s = 2 λ √(α t). Recovers the
    one-phase root when Ste_s = 0.
    """
    lo, hi = 1e-12, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        em = math.exp(mid * mid)
        rhs = ste_l / max(math.erf(mid), 1e-16) - ste_s / max(math.erfc(mid), 1e-16)
        val = mid * math.sqrt(math.pi) * em - rhs
        if val > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
