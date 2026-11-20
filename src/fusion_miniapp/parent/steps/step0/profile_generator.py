"""
Synthetic DIII-D-like edge profiles for n_e and T_e.

Generates ne(psi) and Te(psi) on a psi grid using:
  - three_region: Core + edge (mtanh) + SOL (image 1)
  - snyder: P.B. Snyder tanh + Heaviside core (image 2)

Both models support H-mode (steep pedestal) and L-mode (smooth) via kind="H"|"L".

Units: n_e in 1e20 m^-3, T_e in keV.

Usage:
  from profile_generator import generate_profile, plot_demo
  prof = generate_profile(psi, kind="H", model="three_region")
  prof = generate_profile(psi, kind="L", model="snyder")
  plot_demo(model="three_region")
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

# matplotlib imported lazily in plot_* functions only (avoids ~30s load on generate_profile)

try:
    from scipy.stats import qmc as _qmc
    _HAS_SCIPY_QMC = hasattr(_qmc, "LatinHypercube")
except (ImportError, AttributeError):
    _qmc = None
    _HAS_SCIPY_QMC = False

# Default psi grid: core (0) to edge/separatrix (1.05)
PSI_MIN = 0.0
PSI_MAX = 1.05
DEFAULT_PSI_N = 256

# Parameter ranges for LHS (min, max) per model and kind
PARAM_RANGES_THREE_REGION_H = {
    "a1_ne": (2.5, 4.5),
    "gamma_ne": (0.25, 0.45),
    "Delta": (0.04, 0.10),
    "a2_ne": (1.0, 2.5),
    "a3": (0.9, 1.3),
    "a4": (0.9, 1.4),
    "bcore_ne": (0.0, 0.4),
    "core_linear_frac_ne": (0.0, 0.5),
    "a1_Te": (0.5, 0.9),
    "gamma_Te": (0.08, 0.20),
    "a2_Te": (0.4, 0.9),
    "bcore_Te": (0.0, 0.5),
    "core_linear_frac_Te": (0.3, 0.9),
}
PARAM_RANGES_THREE_REGION_L = {
    "a1_ne": (1.2, 2.2),
    "gamma_ne": (0.45, 0.65),
    "Delta": (0.12, 0.25),
    "a2_ne": (0.3, 1.0),
    "a3": (0.8, 1.2),
    "a4": (0.8, 1.3),
    "bcore_ne": (0.0, 0.25),
    "core_linear_frac_ne": (0.0, 0.3),
    "a1_Te": (0.2, 0.45),
    "gamma_Te": (0.25, 0.45),
    "a2_Te": (0.15, 0.4),
    "bcore_Te": (0.0, 0.3),
    "core_linear_frac_Te": (0.15, 0.7),
}
PARAM_RANGES_SNYDER_H = {
    "ne_sep": (0.15, 0.25),
    "a0_ne": (0.5, 1.0),
    "a1_ne": (0.3, 0.6),
    "Te_sep": (0.05, 0.15),
    "a0_Te": (0.4, 0.8),
    "a1_Te": (0.5, 1.5),
    "Delta": (0.04, 0.10),
}
PARAM_RANGES_SNYDER_L = {
    "ne_sep": (0.15, 0.25),
    "a0_ne": (0.2, 0.5),
    "a1_ne": (0.1, 0.35),
    "Te_sep": (0.1, 0.2),
    "a0_Te": (0.15, 0.4),
    "a1_Te": (0.2, 0.6),
    "Delta": (0.12, 0.25),
}


# -----------------------------------------------------------------------------
# Model 1: Three-region (core / edge / SOL) from image 1
# -----------------------------------------------------------------------------

def three_region_profile(
    psi: np.ndarray,
    a1: float,
    gamma: float,
    Delta: float,
    a2: float,
    a3: float = 2.0,
    a4: float = 1.0,
    bcore: float = 0.0,
    core_linear_frac: float = 0.0,
) -> np.ndarray:
    """
    Three-region profile: core, edge (mtanh pedestal), SOL.

    Core (psi <= psi_t):     y = y_edge(psi_t) + a2 * core_term
    Edge (psi_t < psi <= 1): y = a1 * [gamma + (1-gamma) * g_mtanH(...)]
    SOL (psi > 1):           y = y_edge(1) * (4 - 3*psi)

    psi_t = 1 - Delta, psi_m = 1 - Delta/2.
    a1 = pedestal top, gamma = separatrix factor (y(1) = a1*gamma).
    bcore controls the modified-tanh skew on the core/inboard side.
    core_linear_frac blends the power-law core with a linear core:
      core_term = (1-core_linear_frac)*|1-(psi/psi_t)^a3|^a4 + core_linear_frac*(1-psi/psi_t)
    Larger core_linear_frac gives earlier roll-off in the core (closer to experimental red curve).

    Returns values in same units as a1 (use 1e19 m^-3 for ne, then scale to 1e20).
    """
    psi = np.asarray(psi)
    psi_t = 1.0 - Delta
    psi_m = 1.0 - Delta / 2.0

    # Edge formula (valid for psi_t <= psi <= 1): modified tanh with one-sided core skew.
    # mtanh(x; bcore) = ((1 + bcore*x) e^x - e^{-x}) / (e^x + e^{-x})
    # We normalize so g(-1)=1 and g(1)=0, preserving pedestal and separatrix anchors.
    def _mtanh(x: np.ndarray, b: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ex = np.exp(np.clip(x, -50.0, 50.0))
        emx = np.exp(np.clip(-x, -50.0, 50.0))
        return ((1.0 + b * x) * ex - emx) / (ex + emx + 1e-12)

    x_edge = (psi - psi_m) / (Delta / 2.0 + 1e-12)
    m_hi = _mtanh(1.0, bcore)
    m_lo = _mtanh(-1.0, bcore)
    g_edge = (m_hi - _mtanh(x_edge, bcore)) / (m_hi - m_lo + 1e-12)
    y_edge = a1 * (gamma + (1 - gamma) * g_edge)

    # Evaluate edge at join points
    x_lo = (psi_t - psi_m) / (Delta / 2.0 + 1e-12)  # nominally -1
    g_lo = (m_hi - _mtanh(x_lo, bcore)) / (m_hi - m_lo + 1e-12)
    y_edge_at_psi_t = a1 * (gamma + (1 - gamma) * g_lo)
    y_edge_at_1 = a1 * gamma  # at psi=1

    out = np.empty_like(psi, dtype=float)

    mask_core = psi <= psi_t
    mask_edge = (psi > psi_t) & (psi <= 1.0)
    mask_sol = psi > 1.0

    # Core: blend nonlinear power core with linear-in-psi core to control interior slope.
    psi_core = np.clip(psi[mask_core] / (psi_t + 1e-12), 0.0, 1.0)
    a3_eff = max(float(a3), 1e-6)
    a4_eff = max(float(a4), 1e-6)
    core_linear_frac = float(np.clip(core_linear_frac, 0.0, 1.0))
    power_term = np.abs(1.0 - psi_core ** a3_eff) ** a4_eff
    linear_term = 1.0 - psi_core
    core_term = (1.0 - core_linear_frac) * power_term + core_linear_frac * linear_term
    out[mask_core] = y_edge_at_psi_t + a2 * core_term

    # Edge
    out[mask_edge] = y_edge[mask_edge]

    # SOL: y = y_edge(1) * (4 - 3*psi)
    out[mask_sol] = y_edge_at_1 * (4.0 - 3.0 * psi[mask_sol])

    return out


# -----------------------------------------------------------------------------
# Model 2: P.B. Snyder pedestal model (tanh + Heaviside core)
# -----------------------------------------------------------------------------

def snyder_profile(
    psi: np.ndarray,
    f_sep: float,
    a0: float,
    a1: float,
    Delta: float,
    alpha_core1: float = 1.0,
    alpha_core2: float = 1.0,
) -> np.ndarray:
    """
    P.B. Snyder pedestal profile:
      f(psi) = f_sep + a0 * { tanh[2(1-psi_mid)/Delta] - tanh[2(psi-psi_mid)/Delta] }
               + a1 * H(1 - psi/psi_ped) * [1 - (psi/psi_ped)^alpha_core1]^alpha_core2

    psi_mid = 1 - Delta/2, psi_ped = 1 - Delta.
    H = Heaviside (1 when psi <= psi_ped, else 0).

    Returns profile values in same units as f_sep, a0, a1.
    """
    psi = np.asarray(psi)
    psi_mid = 1.0 - Delta / 2.0
    psi_ped = 1.0 - Delta

    # Pedestal term: a0 * { tanh[2(1-psi_mid)/Delta] - tanh[2(psi-psi_mid)/Delta] }
    tanh_const = np.tanh(2.0 * (1.0 - psi_mid) / (Delta + 1e-12))
    tanh_var = np.tanh(2.0 * (psi - psi_mid) / (Delta + 1e-12))
    pedestal_term = a0 * (tanh_const - tanh_var)

    # Core term: a1 * H(1 - psi/psi_ped) * [1 - (psi/psi_ped)^alpha_core1]^alpha_core2
    ratio = np.clip(psi / (psi_ped + 1e-12), 0.0, 1.0)
    heaviside = (psi <= psi_ped).astype(float)
    core_inner = np.clip(1.0 - ratio ** alpha_core1, 0.0, 1.0)
    core_term = a1 * heaviside * (core_inner ** alpha_core2)

    return f_sep + pedestal_term + core_term


# -----------------------------------------------------------------------------
# Modified tanh (mtanh) edge model — DEPRECATED, use three_region or snyder
# -----------------------------------------------------------------------------

def mtanh_edge(
    psi: np.ndarray,
    f_ped: float,
    f_sep: float,
    psi0: float,
    w: float,
    core_slope: float = 0.0,
    core_curv: float = 0.0,
    psi_join: Optional[float] = None,
    f_core: Optional[float] = None,
) -> np.ndarray:
    """
    Modified tanh edge profile with optional core blend.

    Edge (psi >= psi_join): mtanh
        f(psi) = (f_ped + f_sep)/2 - (f_ped - f_sep)/2 * tanh((psi - psi0) / w)
    Core (psi < psi_join): quadratic joining with C1 continuity at psi_join.

    Args:
        psi: Normalized flux grid
        f_ped: Pedestal top value (high side)
        f_sep: Separatrix value (low side, psi >> psi0)
        psi0: Pedestal center (steepest gradient location)
        w: Pedestal width (smaller = steeper)
        core_slope: If psi_join set, overrides derivative at join (optional)
        core_curv: Curvature for core quadratic (optional)
        psi_join: Join point; if None, use full mtanh (no core blend)
        f_core: Core value at psi_min; used to determine core quadratic if psi_join set

    Returns:
        Profile values, monotonic decreasing toward psi=1 (for f_ped > f_sep).

    .. deprecated:: Use three_region_profile or snyder_profile instead.
    """
    warnings.warn(
        "mtanh_edge is deprecated. Use generate_profile(model='three_region') or "
        "generate_profile(model='snyder') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    psi = np.asarray(psi)
    out = np.empty_like(psi, dtype=float)

    # Edge mtanh: f = (f_ped+f_sep)/2 - (f_ped-f_sep)/2 * tanh((psi-psi0)/w)
    x = (psi - psi0) / (w + 1e-12)
    mtanh_val = (f_ped + f_sep) / 2 - (f_ped - f_sep) / 2 * np.tanh(x)
    mtanh_deriv = -(f_ped - f_sep) / (2 * w) * (1 - np.tanh(x) ** 2)

    if psi_join is None or np.all(psi >= psi_join):
        return mtanh_val

    mask_edge = psi >= psi_join
    mask_core = ~mask_edge

    out[mask_edge] = mtanh_val[mask_edge]

    # Core: quadratic f(psi) = a + b*(psi - psi_join) + c*(psi - psi_join)^2
    # C1 at psi_join: f(psi_join)=f_join, f'(psi_join)=f_join'
    psi_join_val = np.clip(psi_join, psi.min(), psi.max())
    idx_join = np.searchsorted(psi, psi_join_val)
    f_join = (f_ped + f_sep) / 2 - (f_ped - f_sep) / 2 * np.tanh((psi_join - psi0) / w)
    f_join_prime = -(f_ped - f_sep) / (2 * w) * (1 - np.tanh((psi_join - psi0) / w) ** 2)

    if core_slope != 0.0:
        f_join_prime = core_slope

    psi_core = psi[mask_core]
    dpsi = psi_core - psi_join

    if f_core is not None and psi.min() < psi_join:
        # Quadratic through (psi_min, f_core) and (psi_join, f_join) with slope f_join' at join.
        # If the quadratic would be non-monotonic (internal hump), fall back to linear core
        # to avoid flattening artifacts from monotonic clamping.
        psi_min = psi.min()
        d_join = psi_min - psi_join
        if abs(d_join) > 1e-10:
            c = (f_core - f_join - f_join_prime * d_join) / (d_join ** 2)
            # Monotonic: derivative at psi_min must be <= 0 (decreasing toward psi_join).
            # f' = f_join_prime + 2*c*d_join. Need f' <= 0 => c >= -f_join_prime/(2*d_join).
            # When c is too negative, the quadratic has an internal hump.
            c_min = -f_join_prime / (2 * d_join) if d_join != 0 else 0.0
            if c < c_min:
                # Use linear core: monotonic, no hump (avoids flattening from clamp)
                out[mask_core] = f_core + (f_join - f_core) * (psi_core - psi_min) / (psi_join - psi_min)
            else:
                out[mask_core] = f_join + f_join_prime * dpsi + c * dpsi ** 2
        else:
            out[mask_core] = f_join + f_join_prime * dpsi + core_curv * dpsi ** 2
    else:
        # Linear or quadratic with given curvature
        out[mask_core] = f_join + f_join_prime * dpsi + core_curv * dpsi ** 2

    return out


# -----------------------------------------------------------------------------
# Parameter sampling for three_region and snyder (H-mode and L-mode)
# -----------------------------------------------------------------------------

def _sample_params_three_region(kind: str, rng: np.random.Generator) -> Dict:
    """Sample params for three-region model. kind='H' or 'L'."""
    kind = kind.upper()
    if kind == "H":
        Delta = float(rng.uniform(0.04, 0.10))
        return {
            "a1_ne": float(rng.uniform(2.5, 4.5)),
            "gamma_ne": float(rng.uniform(0.25, 0.45)),
            "Delta": Delta,
            "a2_ne": float(rng.uniform(1.0, 2.5)),
            "bcore_ne": float(rng.uniform(0.0, 0.4)),
            "core_linear_frac_ne": float(rng.uniform(0.0, 0.5)),
            "a3": float(rng.uniform(0.9, 1.3)),
            "a4": float(rng.uniform(0.9, 1.4)),
            "a1_Te": float(rng.uniform(0.5, 0.9)),
            "gamma_Te": float(rng.uniform(0.08, 0.20)),
            "a2_Te": float(rng.uniform(0.4, 0.9)),
            "bcore_Te": float(rng.uniform(0.0, 0.5)),
            "core_linear_frac_Te": float(rng.uniform(0.3, 0.9)),
        }
    else:
        # L-mode: broader Delta, lower pedestal, higher gamma (more gradual)
        Delta = float(rng.uniform(0.12, 0.25))
        return {
            "a1_ne": float(rng.uniform(1.2, 2.2)),
            "gamma_ne": float(rng.uniform(0.45, 0.65)),
            "Delta": Delta,
            "a2_ne": float(rng.uniform(0.3, 1.0)),
            "bcore_ne": float(rng.uniform(0.0, 0.25)),
            "core_linear_frac_ne": float(rng.uniform(0.0, 0.3)),
            "a3": float(rng.uniform(0.8, 1.2)),
            "a4": float(rng.uniform(0.8, 1.3)),
            "a1_Te": float(rng.uniform(0.2, 0.45)),
            "gamma_Te": float(rng.uniform(0.25, 0.45)),
            "a2_Te": float(rng.uniform(0.15, 0.4)),
            "bcore_Te": float(rng.uniform(0.0, 0.3)),
            "core_linear_frac_Te": float(rng.uniform(0.15, 0.7)),
        }


def _sample_params_snyder(kind: str, rng: np.random.Generator) -> Dict:
    """Sample params for Snyder model. kind='H' or 'L'."""
    kind = kind.upper()
    if kind == "H":
        Delta = float(rng.uniform(0.04, 0.10))
        return {
            "ne_sep": float(rng.uniform(0.15, 0.25)),
            "a0_ne": float(rng.uniform(0.5, 1.0)),
            "a1_ne": float(rng.uniform(0.3, 0.6)),
            "Te_sep": float(rng.uniform(0.05, 0.15)),
            "a0_Te": float(rng.uniform(0.4, 0.8)),
            "a1_Te": float(rng.uniform(0.5, 1.5)),
            "Delta": Delta,
            "alpha_core1": 1.0,
            "alpha_core2": 1.0,
        }
    else:
        # L-mode: broader Delta, lower amplitudes, higher separatrix
        Delta = float(rng.uniform(0.12, 0.25))
        return {
            "ne_sep": float(rng.uniform(0.15, 0.25)),
            "a0_ne": float(rng.uniform(0.2, 0.5)),
            "a1_ne": float(rng.uniform(0.1, 0.35)),
            "Te_sep": float(rng.uniform(0.1, 0.2)),
            "a0_Te": float(rng.uniform(0.15, 0.4)),
            "a1_Te": float(rng.uniform(0.2, 0.6)),
            "Delta": Delta,
            "alpha_core1": 1.0,
            "alpha_core2": 1.0,
        }


def _lhs_sample(n_samples: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """Latin Hypercube Sampling in [0,1)^d. Uses scipy if available, else numpy."""
    rng = np.random.default_rng(seed)
    if _HAS_SCIPY_QMC:
        sampler = _qmc.LatinHypercube(d=d, seed=seed)
        return sampler.random(n=n_samples)
    # Numpy fallback: stratified sampling per dimension
    samples = np.zeros((n_samples, d))
    for j in range(d):
        strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        samples[:, j] = rng.permutation(strata)
    return samples


def sample_params_lhs(
    model: str,
    kind: str,
    n_samples: int = 20,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Latin Hypercube Sampling of parameters for profile generation.

    Args:
        model: "three_region" or "snyder"
        kind: "H" or "L"
        n_samples: Number of parameter sets to sample
        seed: Random seed for reproducibility

    Returns:
        List of n_samples param dicts.
    """
    model = model.lower()
    kind = kind.upper()
    if model == "three_region":
        ranges = PARAM_RANGES_THREE_REGION_H if kind == "H" else PARAM_RANGES_THREE_REGION_L
    else:
        ranges = PARAM_RANGES_SNYDER_H if kind == "H" else PARAM_RANGES_SNYDER_L

    param_names = list(ranges.keys())
    d = len(param_names)
    l_bounds = np.array([ranges[k][0] for k in param_names])
    u_bounds = np.array([ranges[k][1] for k in param_names])

    samples = _lhs_sample(n_samples, d, seed)
    samples = l_bounds + samples * (u_bounds - l_bounds)

    result = []
    for i in range(n_samples):
        params = {k: float(samples[i, j]) for j, k in enumerate(param_names)}
        params["kind"] = kind
        params["model"] = model
        if model == "three_region":
            params.setdefault("a3", 1.0)
            params.setdefault("a4", 1.0)
            params.setdefault("bcore_ne", 0.0)
            params.setdefault("bcore_Te", 0.0)
            params.setdefault("core_linear_frac_ne", 0.0)
            params.setdefault("core_linear_frac_Te", 0.0)
        else:
            params["alpha_core1"] = 1.0
            params["alpha_core2"] = 1.0
        result.append(params)
    return result


def generate_profiles_lhs(
    n_profiles: int = 20,
    model: str = "three_region",
    kind: Optional[str] = None,
    psi: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Manufacture profiles using Latin Hypercube Sampling of parameters.

    Args:
        n_profiles: Number of profiles to generate (default 20)
        model: "three_region" or "snyder"
        kind: "H", "L", or None. If None, generates n_profiles//2 H and n_profiles//2 L
        psi: Radial grid (default: linspace(0.75, 1.05, 256))
        seed: Random seed for LHS

    Returns:
        List of profile dicts (each with psi, ne, Te, params, model, kind).
    """
    if psi is None:
        psi = np.linspace(PSI_MIN, PSI_MAX, DEFAULT_PSI_N)
    psi = np.asarray(psi)

    if kind is not None:
        param_sets = sample_params_lhs(model, kind, n_samples=n_profiles, seed=seed)
    else:
        n_h = n_profiles // 2
        n_l = n_profiles - n_h
        param_sets = sample_params_lhs(model, "H", n_samples=n_h, seed=seed)
        param_sets += sample_params_lhs(model, "L", n_samples=n_l, seed=seed + 1 if seed is not None else None)

    profiles = []
    for params in param_sets:
        prof = generate_profile(psi=psi, model=params["model"], kind=params["kind"], params=params)
        profiles.append(prof)
    return profiles


# -----------------------------------------------------------------------------
# Unified profile generation (three_region and snyder, H-mode and L-mode)
# -----------------------------------------------------------------------------

def generate_profile(
    psi: Optional[np.ndarray] = None,
    kind: str = "H",
    model: str = "three_region",
    params: Optional[Dict] = None,
    seed: Optional[int] = None,
    n_points: int = DEFAULT_PSI_N,
) -> Dict:
    """
    Generate synthetic n_e and T_e profiles.

    Args:
        psi: Radial grid (default: np.linspace(0.75, 1.05, n_points))
        kind: "H" (H-mode) or "L" (L-mode)
        model: "three_region" or "snyder"
        params: Optional parameter overrides; merged onto sampled defaults for the chosen model
        seed: Random seed for parameter sampling
        n_points: Number of grid points if psi not provided

    Returns:
        Dict with keys: psi, ne, Te, params, model, kind. ne in 1e20 m^-3, Te in keV.
    """
    if psi is None:
        psi = np.linspace(PSI_MIN, PSI_MAX, n_points)
    psi = np.asarray(psi)
    kind = kind.upper()
    model = model.lower()

    if model not in ("three_region", "snyder"):
        raise ValueError(f"model must be 'three_region' or 'snyder', got {model!r}")

    rng = np.random.default_rng(seed)
    sampled = _sample_params_three_region(kind, rng) if model == "three_region" else _sample_params_snyder(kind, rng)
    if params is None:
        params = sampled
    else:
        merged = dict(sampled)
        merged.update(params)
        params = merged
    params["kind"] = kind
    params["model"] = model

    if model == "three_region":
        ne_1e19 = three_region_profile(
            psi, params["a1_ne"], params["gamma_ne"], params["Delta"],
            params["a2_ne"], params.get("a3", 2.0), params.get("a4", 1.0),
            params.get("bcore_ne", 0.0),
            params.get("core_linear_frac_ne", params.get("core_linear_frac", 0.0)),
        )
        ne = ne_1e19 * 0.1  # 1e19 -> 1e20 m^-3
        Te = three_region_profile(
            psi, params["a1_Te"], params["gamma_Te"], params["Delta"],
            params["a2_Te"], params.get("a3", 2.0), params.get("a4", 1.0),
            params.get("bcore_Te", 0.0),
            params.get("core_linear_frac_Te", params.get("core_linear_frac", 0.0)),
        )
    else:
        ne = snyder_profile(
            psi, params["ne_sep"], params["a0_ne"], params["a1_ne"], params["Delta"],
            params.get("alpha_core1", 1.0), params.get("alpha_core2", 1.0),
        )
        Te = snyder_profile(
            psi, params["Te_sep"], params["a0_Te"], params["a1_Te"], params["Delta"],
            params.get("alpha_core1", 1.0), params.get("alpha_core2", 1.0),
        )
        ne = np.maximum(ne, 0.0)
        Te = np.maximum(Te, 0.0)

    return {
        "psi": psi,
        "ne": ne,
        "Te": Te,
        "params": params,
        "model": model,
        "kind": kind,
    }


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

def export_profile_dict(prof: Dict) -> Dict:
    """
    Export profile for downstream use.

    Returns dict with psi, ne, Te arrays and params. Suitable for JSON/YAML
    serialization of params (arrays excluded from serialization in typical use).
    """
    return {
        "psi": np.asarray(prof["psi"]),
        "ne": np.asarray(prof["ne"]),
        "Te": np.asarray(prof["Te"]),
        "params": dict(prof["params"]),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_profiles(
    prof_H: Dict,
    prof_L: Dict,
    ax_ne=None,
    ax_Te=None,
    psi_ped_band: Tuple[float, float] = (0.92, 0.99),
):
    """
    Plot n_e and T_e profiles (H-mode vs L-mode) in DIII-D reference style.

    Returns (ax_ne, ax_Te).
    """
    import matplotlib.pyplot as plt
    if ax_ne is None or ax_Te is None:
        fig, (ax_ne, ax_Te) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    psi_H, ne_H, Te_H = prof_H["psi"], prof_H["ne"], prof_H["Te"]
    psi_L, ne_L, Te_L = prof_L["psi"], prof_L["ne"], prof_L["Te"]

    # (a) n_e
    ax_ne.plot(psi_H, ne_H, "k-", linewidth=2.5, label="H-mode")
    ax_ne.plot(psi_L, ne_L, "b-", linewidth=2.5, label="L-mode")
    ax_ne.axvspan(psi_ped_band[0], psi_ped_band[1], alpha=0.2, color="orange")
    ax_ne.set_ylabel(r"$n_e$ ($10^{20}$ m$^{-3}$)")
    ax_ne.set_ylim(0, None)
    ax_ne.legend(loc="upper right")
    ax_ne.grid(True, alpha=0.3)
    ax_ne.set_title("(a) $n_e$")

    # (b) T_e
    ax_Te.plot(psi_H, Te_H, "k-", linewidth=2.5, label="H-mode")
    ax_Te.plot(psi_L, Te_L, "b-", linewidth=2.5, label="L-mode")
    ax_Te.axvspan(psi_ped_band[0], psi_ped_band[1], alpha=0.2, color="orange")
    ax_Te.set_xlabel("Radius (ψ)")
    ax_Te.set_ylabel(r"$T_e$ (keV)")
    ax_Te.set_ylim(0, None)
    ax_Te.legend(loc="upper right")
    ax_Te.grid(True, alpha=0.3)
    ax_Te.set_title("(b) $T_e$")

    plt.tight_layout()
    return ax_ne, ax_Te


def plot_demo(
    seed: Optional[int] = 42,
    model: str = "three_region",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Produce a figure comparing H-mode and L-mode n_e and T_e profiles.

    Args:
        model: "three_region" or "snyder"
    """
    import matplotlib.pyplot as plt
    psi = np.linspace(PSI_MIN, PSI_MAX, 256)
    prof_H = generate_profile(psi, kind="H", model=model, seed=seed)
    prof_L = generate_profile(psi, kind="L", model=model, seed=seed + 1)

    fig, (ax_ne, ax_Te) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    plot_profiles(prof_H, prof_L, ax_ne=ax_ne, ax_Te=ax_Te)
    fig.suptitle(f"Synthetic DIII-D-like edge profiles ({model} model)", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()


def plot_model_comparison(
    seed: Optional[int] = 42,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Compare three-region and Snyder models (H-mode and L-mode).

    Produces a 2x1 figure (ne, Te) with four curves: H/L for each model.
    """
    import matplotlib.pyplot as plt
    psi = np.linspace(0.5, 1.2, 256)

    prof_three_H = generate_profile(psi, kind="H", model="three_region", seed=seed)
    prof_three_L = generate_profile(psi, kind="L", model="three_region", seed=seed + 1)
    prof_snyder_H = generate_profile(psi, kind="H", model="snyder", seed=seed)
    prof_snyder_L = generate_profile(psi, kind="L", model="snyder", seed=seed + 1)

    fig, (ax_ne, ax_Te) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_ne.plot(prof_three_H["psi"], prof_three_H["ne"], "k-", lw=2, label="Three-region H")
    ax_ne.plot(prof_three_L["psi"], prof_three_L["ne"], "k--", lw=1.5, label="Three-region L")
    ax_ne.plot(prof_snyder_H["psi"], prof_snyder_H["ne"], "b-", lw=2, label="Snyder H")
    ax_ne.plot(prof_snyder_L["psi"], prof_snyder_L["ne"], "b--", lw=1.5, label="Snyder L")
    ax_ne.axvline(1.0, color="gray", ls="--", alpha=0.7)
    ax_ne.set_ylabel(r"$n_e$ ($10^{20}$ m$^{-3}$)")
    ax_ne.set_ylim(0, None)
    ax_ne.legend(loc="upper right")
    ax_ne.grid(True, alpha=0.3)
    ax_ne.set_title("(a) $n_e$ – three-region vs Snyder (H/L mode)")

    ax_Te.plot(prof_three_H["psi"], prof_three_H["Te"], "k-", lw=2, label="Three-region H")
    ax_Te.plot(prof_three_L["psi"], prof_three_L["Te"], "k--", lw=1.5, label="Three-region L")
    ax_Te.plot(prof_snyder_H["psi"], prof_snyder_H["Te"], "b-", lw=2, label="Snyder H")
    ax_Te.plot(prof_snyder_L["psi"], prof_snyder_L["Te"], "b--", lw=1.5, label="Snyder L")
    ax_Te.axvline(1.0, color="gray", ls="--", alpha=0.7)
    ax_Te.set_xlabel("Radius (ψ)")
    ax_Te.set_ylabel(r"$T_e$ (keV)")
    ax_Te.set_ylim(0, None)
    ax_Te.legend(loc="upper right")
    ax_Te.grid(True, alpha=0.3)
    ax_Te.set_title("(b) $T_e$ – three-region vs Snyder (H/L mode)")

    fig.suptitle("Profile model comparison: three-region vs P.B. Snyder", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_lhs_demo(
    n_profiles: int = 20,
    model: str = "three_region",
    seed: Optional[int] = 42,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Generate and plot LHS-sampled profiles (10 H + 10 L)."""
    import matplotlib.pyplot as plt
    profiles = generate_profiles_lhs(n_profiles=n_profiles, model=model, kind=None, seed=seed)
    psi = profiles[0]["psi"]
    fig, (ax_ne, ax_Te) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for prof in profiles:
        c = "k" if prof["kind"] == "H" else "b"
        ax_ne.plot(psi, prof["ne"], c=c, alpha=0.5, lw=1)
        ax_Te.plot(psi, prof["Te"], c=c, alpha=0.5, lw=1)
    ax_ne.set_ylabel(r"$n_e$ ($10^{20}$ m$^{-3}$)")
    ax_Te.set_ylabel(r"$T_e$ (keV)")
    ax_Te.set_xlabel("Radius (ψ)")
    ax_ne.set_title(f"LHS {n_profiles} profiles ({model})")
    ax_Te.set_title("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    import argparse
    import matplotlib
    parser = argparse.ArgumentParser(description="Synthetic DIII-D edge profile generator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="three_region", choices=["three_region", "snyder"])
    parser.add_argument("--save", type=str, default=None, help="Save figure path")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare three-region vs Snyder (H/L)")
    parser.add_argument("--lhs", type=int, default=None, metavar="N", help="LHS: manufacture N profiles (default 20)")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    if args.lhs is not None:
        run_lhs_demo(n_profiles=args.lhs, model=args.model, seed=args.seed, save_path=args.save, show=not args.no_show)
    elif args.compare:
        plot_model_comparison(seed=args.seed, save_path=args.save, show=not args.no_show)
    else:
        plot_demo(seed=args.seed, model=args.model, save_path=args.save, show=not args.no_show)
