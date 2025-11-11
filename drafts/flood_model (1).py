#!/usr/bin/env python3
"""
Hybrid Flood Prediction Framework — Integrated (A + B)
Adds PDF-derived components to the user’s existing code:
- Multi-source precipitation fusion (radar + NWP + gauge variance) [Eq. (3)-(4)].
- Chaos-perturbed hydrologic parameters CN, K [Eq. (16)-(17)] via spatial bases.
- KGE computation and Bayesian model weighting [Eq. (26)-(34)].
- Probabilistic impact & risk mapping (exceedance, hazard) [Eq. (48)-(54)].
- Hooks/stubs for GPU SWE solver (CUDA interface placeholder) [Sec. VII].
- Integrated into interactive runner (adds Steps 15–18 reports).

This file preserves the original steps (1–14 from foundation.pdf) and appends new functions
and reporting sections (from FDFFM.pdf). Designed to remain terminal-friendly. hehe, need coffee pls
"""

import math
import numpy as np
import random
import csv
from typing import List, Tuple, Dict, Any, Iterable

# -----------------------------
# Table helper (improved)
# -----------------------------

def print_table(headers: List[str], rows: List[List[Any]], col_widths: List[int] = None):
    if col_widths is None:
        col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    print(sep)
    print("|" + "|".join(str(h).center(w) for h, w in zip(headers, col_widths)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + "|")
    print(sep + "\n")

# =========================================================
# Original Steps (1–14) — preserved
# =========================================================

# -----------------------------
# Step 1: Continuity (diagnostic)
# -----------------------------

def continuity_divergence(u: float, v: float, dh_dx: float, dh_dy: float) -> float:
    return dh_dx * u + dh_dy * v

# -----------------------------
# Step 2: Momentum (diagnostic)
# -----------------------------

def momentum_x(u, v, eta_grad_x, n, h, zb_grad_x, g=9.81):
    friction = - g * n**2 * u * math.sqrt(u*u + v*v) / (h**(4.0/3.0) + 1e-12)
    pressure = - g * eta_grad_x
    bed = - g * h * zb_grad_x
    adv = 0.0
    return adv + pressure + friction + bed

def momentum_y(u, v, eta_grad_y, n, h, zb_grad_y, g=9.81):
    friction = - g * n**2 * v * math.sqrt(u*u + v*v) / (h**(4.0/3.0) + 1e-12)
    pressure = - g * eta_grad_y
    bed = - g * h * zb_grad_y
    adv = 0.0
    return adv + pressure + friction + bed

# -----------------------------
# Step 3: Green-Ampt infiltration
# -----------------------------

def green_ampt_cumulative(K: float, psi: float, delta_theta: float, t: float, tol=1e-8, maxiter=500) -> float:
    if psi * delta_theta <= 0:
        return max(0.0, K * t)
    F = max(1e-12, K * t)
    for _ in range(maxiter):
        F_new = K * t + psi * delta_theta * math.log(1.0 + F / (psi * delta_theta))
        if abs(F_new - F) < tol:
            F = F_new
            break
        F = F_new
    return max(0.0, F)

# -----------------------------
# Step 4: SCS Curve Number Runoff
# -----------------------------

def scs_runoff(P_mm: float, CN: float) -> float:
    S = 1000.0 / CN - 10.0
    Ia = 0.2 * S
    if P_mm <= Ia:
        return 0.0
    Q = (P_mm - Ia) ** 2 / (P_mm - Ia + S)
    return Q

def scs_S(CN: float) -> float:
    return 1000.0 / CN - 10.0

# -----------------------------
# Step 5: Antecedent Moisture & dynamic CN
# -----------------------------

def antecedent_moisture_index(rain_5days: List[float], beta=0.1) -> float:
    AMC = 0.0
    for i, P in enumerate(rain_5days, start=1):
        AMC += P * math.exp(-beta * i)
    return AMC

def adjusted_CN(CNdry: float, CNwet: float, AMC: float, AMCmin=15.0, AMCmax=35.0) -> float:
    frac = 0.0
    if AMCmax != AMCmin:
        frac = (AMC - AMCmin) / (AMCmax - AMCmin)
        frac = min(max(frac, 0.0), 1.0)
    return CNdry + (CNwet - CNdry) * frac

# -----------------------------
# Step 6: Shallow Water helpers
# -----------------------------

def state_vector(h: float, u: float, v: float) -> np.ndarray:
    return np.array([h, h * u, h * v])

def flux_F(U: np.ndarray, g=9.81) -> np.ndarray:
    h, hu, hv = U
    u = hu / (h + 1e-12)
    v = hv / (h + 1e-12)
    return np.array([hu, hu * u + 0.5 * g * h * h, hu * v])

def flux_G(U: np.ndarray, g=9.81) -> np.ndarray:
    h, hu, hv = U
    u = hu / (h + 1e-12)
    v = hv / (h + 1e-12)
    return np.array([hv, hu * v, hv * v + 0.5 * g * h * h])

def source_term(R, I, h, zb_grad_x, zb_grad_y, u, v, Cf=0.0, g=9.81):
    speed = math.sqrt(u*u + v*v) + 1e-12
    Cf_term = Cf * h * speed
    return np.array([R - I, -g * h * zb_grad_x - Cf_term * u, -g * h * zb_grad_y - Cf_term * v])

# -----------------------------
# Step 7: Uncertainty sources
# -----------------------------

def perturb_rainfall(P_true: float, sigma_rad=0.15, bias_mm=2.0) -> float:
    eps_rad = random.gauss(0.0, sigma_rad)
    eps_sys = random.gauss(0.0, bias_mm)
    return P_true * (1.0 + eps_rad) + eps_sys

def perturb_CN(CN_mean: float, sigma_CN=5.0) -> float:
    return random.gauss(CN_mean, sigma_CN)

# -----------------------------
# Step 8/9: Lorenz-63
# -----------------------------

def lorenz63_step(x, y, z, sigma=10.0, rho=28.0, beta=8.0/3.0):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def integrate_lorenz63(x0=1.0, y0=1.0, z0=1.0, dt=0.01, steps=100):
    x, y, z = x0, y0, z0
    for _ in range(steps):
        dx1, dy1, dz1 = lorenz63_step(x, y, z)
        x1 = x + dx1 * dt / 2.0
        y1 = y + dy1 * dt / 2.0
        z1 = z + dz1 * dt / 2.0
        dx2, dy2, dz2 = lorenz63_step(x1, y1, z1)
        x += dx2 * dt
        y += dy2 * dt
        z += dz2 * dt
    return x, y, z

# -----------------------------
# Step 9/10: Lorenz-96
# -----------------------------

def lorenz96_step(X: np.ndarray, F=8.0):
    N = len(X)
    dX = np.zeros(N)
    for k in range(N):
        dX[k] = (X[(k+1) % N] - X[(k-2) % N]) * X[(k-1) % N] - X[k] + F
    return dX

def integrate_lorenz96(X0: List[float], dt=0.001, steps=1000, F=8.0):
    X = np.array(X0, dtype=float)
    for _ in range(steps):
        dX = lorenz96_step(X, F=F)
        X += dX * dt
    return X

# -----------------------------
# Step 10: Spatial basis & precipitation perturbation
# -----------------------------

def spatial_basis_psi(kx: int, ky: int, x: float, y: float, Lx=10.0, Ly=10.0):
    return math.sin(kx * math.pi * x / Lx) * math.sin(ky * math.pi * y / Ly)

def precipitation_perturbation(Pbase: float, alpha_p: float, weights: List[float], psi_vals: List[float], Xvals: List[float]):
    s = 0.0
    for wk, psik, Xk in zip(weights, psi_vals, Xvals):
        s += wk * psik * Xk
    return Pbase * (1.0 + alpha_p * s)

# -----------------------------
# Step 11: Simple particle filter
# -----------------------------

def particle_filter_simple(obs_Q_series: List[float], Np=500, sigma_obs=5.0):
    particles = np.random.normal(loc=obs_Q_series[0], scale=sigma_obs, size=Np)
    weights = np.ones(Np) / Np
    estimates = []
    for t, obs in enumerate(obs_Q_series):
        particles += np.random.normal(0, 2.0, size=Np)
        likelihoods = (1.0 / (math.sqrt(2 * math.pi) * sigma_obs)) * np.exp(-0.5 * ((particles - obs) / sigma_obs) ** 2)
        weights = likelihoods * weights
        weights += 1e-300
        weights /= np.sum(weights)
        est = np.sum(particles * weights)
        estimates.append(est)
        Neff = 1.0 / np.sum(weights ** 2)
        if Neff < Np / 2.0:
            positions = (np.arange(Np) + random.random()) / Np
            cumulative_sum = np.cumsum(weights)
            indexes = np.searchsorted(cumulative_sum, positions)
            particles = particles[indexes]
            weights.fill(1.0 / Np)
    return estimates

# -----------------------------
# Step 12/13 scenario builder
# -----------------------------

def scenario_three_hour(base_hours: Dict[int, Dict], chaos_mods: Dict[int, Dict]):
    timeline = []
    for hr in [1, 2, 3]:
        b = base_hours[hr]
        c = chaos_mods[hr]
        timeline.append({'hour': hr, 'base': b, 'chaos': c})
    return timeline

# -----------------------------
# Step 14 final flood characteristics
# -----------------------------

def final_flood_characteristics(Q_peak: float, area_km2: float = 12.5):
    hmax = 0.01 * Q_peak
    vmax = 0.02 * Q_peak
    tpeak = 2.1
    Aflood = area_km2
    dh = 0.4
    dv = 0.8
    dt = 0.3
    dA = 3.2
    return {
        'hmax': hmax,
        'hmax_range': (max(0, hmax - dh), hmax + dh),
        'vmax': vmax,
        'vmax_range': (max(0, vmax - dv), vmax + dv),
        'tpeak': tpeak,
        'tpeak_range': (tpeak - dt, tpeak + dt),
        'Aflood': Aflood,
        'Aflood_range': (max(0, Aflood - dA), Aflood + dA)
    }

# =========================================================
# NEW: PDF-derived components (Steps 15–18)
# =========================================================

# -----------------------------
# Step 15: Multi-source precipitation fusion (Radar + NWP)
# -----------------------------

def fuse_precip(qpe_radar: float, p_nwp: float, var_radar: float, var_nwp: float) -> Tuple[float, float]:
    """Implements Eq. (3)-(4): Pfused = a*QPE + (1-a)*PNWP, a = var_nwp/(var_radar+var_nwp).
    Returns (Pfused, alpha).
    """
    denom = (var_radar + var_nwp) + 1e-12
    alpha = var_nwp / denom
    p_fused = alpha * qpe_radar + (1.0 - alpha) * p_nwp
    return p_fused, alpha

# -----------------------------
# Step 16: Chaos-perturbed parameters CN, K (spatial bases)
# -----------------------------

def chaos_perturb_params(CN_mean: float, K_mean: float, X_modes: Iterable[float],
                         basis_fields: Iterable[float], sigma_CN=5.0, sigma_K=0.2) -> Tuple[float, float]:
    """Implements Eq. (16)-(17) at a point using provided spatial basis values (phi, chi ~= basis_fields).
    Returns (CN_adj, K_adj). K is lognormal-like via exp().
    """
    s_sum = sum(b * x for b, x in zip(basis_fields, X_modes))
    CN_adj = CN_mean + sigma_CN * s_sum
    K_adj = K_mean * math.exp(sigma_K * s_sum)
    return CN_adj, K_adj

# -----------------------------
# Step 17: KGE and Bayesian weighting
# -----------------------------

def kge_components(sim: np.ndarray, obs: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns (KGE, r, alpha, beta). Eq. (26)-(29)."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if sim.size != obs.size:
        raise ValueError("sim and obs must have same length")
    # Handle degenerate std
    s_std = np.std(sim, ddof=1) + 1e-12
    o_std = np.std(obs, ddof=1) + 1e-12
    r = float(np.corrcoef(sim, obs)[0, 1]) if sim.size > 1 else 0.0
    alpha = s_std / o_std
    beta = (np.mean(sim) + 1e-12) / (np.mean(obs) + 1e-12)
    kge = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge, r, alpha, beta


def bayes_weight_update(prev_w: np.ndarray, y_obs: float, y_hat: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Implements Eq. (32) in vector form for M models.
    prev_w, y_hat, sigma are arrays of same length M. Returns new weights (sum=1).
    """
    prev_w = np.asarray(prev_w, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float) + 1e-12
    exponent = -0.5 * ((y_obs - y_hat) ** 2) / (sigma ** 2)
    numer = prev_w * np.exp(exponent)
    numer += 1e-300
    w = numer / np.sum(numer)
    return w

# -----------------------------
# Step 18: Probabilistic impact & risk mapping (reduced form)
# -----------------------------

def exceedance_probability(values_ens: np.ndarray, threshold: float) -> float:
    values_ens = np.asarray(values_ens, dtype=float)
    return float(np.mean(values_ens > threshold))


def hazard_metric(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Proxy for hazard index H = h*v*(v+0.5) from Eq. (49)."""
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    return h * v * (v + 0.5)


def hazard_exceedance_probability(h_ens: np.ndarray, v_ens: np.ndarray, Hcrit: float) -> float:
    H = hazard_metric(h_ens, v_ens)
    return float(np.mean(H > Hcrit))

# -----------------------------
# GPU SWE Solver — placeholder hooks (non-executable here)
# -----------------------------

def gpu_swe_solver_stub(*args, **kwargs):
    """Placeholder for CUDA solver kernel interface (Sec. VII). In production, this would
    call into compiled CUDA/C++ code via PyCUDA/CuPy. Kept as a stub to show
    architecture linkage without requiring GPU runtime here."""
    raise NotImplementedError("GPU SWE solver is a stub in this environment.")

# =========================================================
# Report collector / CSV writer (extended)
# =========================================================

def collect_results(inputs: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Returns a dict mapping step name -> list of (key, value, units/notes)."""
    results: Dict[str, List[Tuple[str, str, str]]] = {}

    # Step 1 & 2
    div = continuity_divergence(u=1.0, v=0.5, dh_dx=0.01, dh_dy=0.005)
    momx = momentum_x(u=1.0, v=0.5, eta_grad_x=0.002, n=0.03, h=0.5, zb_grad_x=0.001)
    momy = momentum_y(u=1.0, v=0.5, eta_grad_y=0.001, n=0.03, h=0.5, zb_grad_y=0.0005)
    results['Step 1 & 2: Continuity & Momentum'] = [
        ("Divergence", f"{div:.5f}", "m/s (diagnostic)"),
        ("Momentum X", f"{momx:.5f}", "N/m^2 (diagnostic)"),
        ("Momentum Y", f"{momy:.5f}", "N/m^2 (diagnostic)")
    ]

    # Step 3
    K = inputs['K_mean']
    psi = inputs['psi']
    dtot = inputs['delta_theta']
    dur_s = inputs['duration_hr'] * 3600.0
    F_1hr = green_ampt_cumulative(K, psi, dtot, 3600.0)
    F_total = green_ampt_cumulative(K, psi, dtot, dur_s)
    results['Step 3: Green-Ampt'] = [
        ("K (m/s)", f"{K:.3e}", ""),
        ("psi (m)", f"{psi:.4f}", ""),
        ("Delta theta", f"{dtot:.4f}", ""),
        ("F (1 hr)", f"{F_1hr:.6f}", "m"),
        ("F (total)", f"{F_total:.6f}", "m")
    ]

    # Step 4
    P_total = inputs['P_total']
    CN_mean = inputs['CN_mean']
    Q_total_mm = scs_runoff(P_total, CN_mean)
    S_val = scs_S(CN_mean)
    results['Step 4: SCS CN Runoff'] = [
        ("CN mean", f"{CN_mean:.2f}", ""),
        ("S (mm)", f"{S_val:.4f}", "mm"),
        ("Ia (mm)", f"{0.2 * S_val:.4f}", "mm"),
        ("Total rain (mm)", f"{P_total:.2f}", "mm"),
        ("Q (mm)", f"{Q_total_mm:.4f}", "mm"),
        ("Runoff coeff (%)", f"{(Q_total_mm / (P_total + 1e-12) * 100):.2f}", "%")
    ]

    # Step 5
    AMC = antecedent_moisture_index(inputs['rain5_list'])
    CN_adj = adjusted_CN(80.0, 90.0, AMC, AMCmin=15.0, AMCmax=35.0)
    results['Step 5: Antecedent Moisture'] = [
        ("Previous 5-day rainfall", str(inputs['rain5_list']), "mm"),
        ("AMC", f"{AMC:.4f}", "mm (weighted)"),
        ("Adjusted CN", f"{CN_adj:.4f}", "")
    ]

    # Step 6
    U = state_vector(h=0.5, u=1.0, v=0.5)
    Fv = flux_F(U)
    Gv = flux_G(U)
    Svec = source_term(R=0.0, I=0.0, h=0.5, zb_grad_x=0.001, zb_grad_y=0.0005, u=1.0, v=0.5, Cf=0.0)
    results['Step 6: SWE Vectors'] = [
        ("State U", np.array2string(U, precision=4), ""),
        ("Flux F", np.array2string(Fv, precision=4), ""),
        ("Flux G", np.array2string(Gv, precision=4), ""),
        ("Source S", np.array2string(Svec, precision=4), "")
    ]

    # Step 7
    P_obs_pert = perturb_rainfall(P_total, sigma_rad=0.15, bias_mm=2.0)
    CN_pert = perturb_CN(CN_mean, sigma_CN=5.0)
    results['Step 7: Uncertainty'] = [
        ("Rainfall perturbed (mm)", f"{P_obs_pert:.3f}", "mm"),
        ("CN perturbed", f"{CN_pert:.3f}", "")
    ]

    # Step 8/9 Lorenz-63
    x_l, y_l, z_l = integrate_lorenz63(1.0, 1.0, 1.0, dt=0.01, steps=1000)
    results['Step 8/9: Lorenz-63'] = [
        ("x", f"{x_l:.6f}", ""),
        ("y", f"{y_l:.6f}", ""),
        ("z", f"{z_l:.6f}", "")
    ]

    # Step 9/10 Lorenz-96
    Nc = 40
    X0 = [8.0] * Nc
    X0[19] = 8.01
    X_final = integrate_lorenz96(X0, dt=0.001, steps=10000, F=8.0)
    results['Step 9/10: Lorenz-96'] = [
        ("Sample first 8", np.array2string(X_final[:8].round(4), separator=', '), ""),
        ("Modes used", np.array2string(X_final[:5].round(4), separator=', '), "")
    ]

    # Step 10.x spatial perturbation example
    x_pt, y_pt = 5.0, 3.0
    psi_vals = [spatial_basis_psi(k+1, 1, x_pt, y_pt, Lx=10.0, Ly=10.0) for k in range(5)]
    weights = [math.exp(-(k+1)) for k in range(5)]
    Pbase = 50.0
    alpha_p = 0.2
    X_modes = X_final[:5].tolist()
    P_pert = precipitation_perturbation(Pbase, alpha_p, weights, psi_vals, X_modes)
    results['Step 10.x: Spatial Perturbation'] = [
        ("psi_vals", str([round(v, 6) for v in psi_vals]), ""),
        ("weights", str([round(w, 6) for w in weights]), ""),
        ("X_modes", str([round(v, 6) for v in X_modes]), ""),
        ("Pbase (mm/hr)", f"{Pbase:.3f}", "mm/hr"),
        ("P_pert (mm/hr)", f"{P_pert:.6f}", "mm/hr")
    ]

    # Step 11 particle filter demo
    Q_obs_series = [5.0, 85.0, 150.0]
    pf_estimates = particle_filter_simple(Q_obs_series, Np=400, sigma_obs=10.0)
    pf_rows = [(f"t={i+1}", f"{Q_obs_series[i]:.2f}", f"{pf_estimates[i]:.2f}") for i in range(len(Q_obs_series))]
    results['Step 11: Particle Filter'] = [("Obs t", "Observed Q", "PF estimate")] + pf_rows

    # Steps 12/13 scenario
    baseline = {
        1: {'rain': 10.0, 'infil': 9.0, 'runoff': 1.0, 'Q': 5.0},
        2: {'rain': 50.0, 'infil': 5.0, 'runoff': 45.0, 'Q': 85.0},
        3: {'rain': 50.0, 'infil': 1.0, 'runoff': 49.0, 'Q': 150.0},
    }
    chaos = {
        1: {'rain': 10.0 * 1.15, 'infil': 9.0 * 0.9, 'runoff': 1.0 * 1.2, 'Q': 6.0},
        2: {'rain': 50.0 * 1.2, 'infil': 5.0 * 1.05, 'runoff': 45.0 * 1.3, 'Q': 110.0},
        3: {'rain': 50.0 * 0.9, 'infil': 1.0 * 1.2, 'runoff': 49.0 * 0.95, 'Q': 140.0},
    }
    timeline = scenario_three_hour(baseline, chaos)
    rows_baseline, rows_chaos = [], []
    for rec in timeline:
        rows_baseline.append((f"hr{rec['hour']}", f"{rec['base']['rain']:.2f}", f"{rec['base']['infil']:.2f}", f"{rec['base']['runoff']:.2f}", f"{rec['base']['Q']:.2f}"))
        rows_chaos.append((f"hr{rec['hour']}", f"{rec['chaos']['rain']:.2f}", f"{rec['chaos']['infil']:.2f}", f"{rec['chaos']['runoff']:.2f}", f"{rec['chaos']['Q']:.2f}"))

    results['Step 12/13: Baseline Scenario'] = [("Hour", "Rain", "Infil", "Runoff", "Q")] + rows_baseline
    results['Step 12/13: Chaos Scenario'] = [("Hour", "Rain", "Infil", "Runoff", "Q")] + rows_chaos

    # Step 14 final
    Q_peak = max([rec['chaos']['Q'] for rec in timeline] + [rec['base']['Q'] for rec in timeline])
    final = final_flood_characteristics(Q_peak, area_km2=12.5)
    results['Step 14: Final Flood Characteristics'] = [
        ("Peak Discharge (m3/s)", f"{Q_peak:.2f}", ""),
        ("hmax (m)", f"{final['hmax']:.4f}", f"{final['hmax_range'][0]:.4f} - {final['hmax_range'][1]:.4f}"),
        ("vmax (m/s)", f"{final['vmax']:.4f}", f"{final['vmax_range'][0]:.4f} - {final['vmax_range'][1]:.4f}"),
        ("tpeak (hr)", f"{final['tpeak']:.3f}", f"{final['tpeak_range'][0]:.3f} - {final['tpeak_range'][1]:.3f}"),
        ("Aflood (km2)", f"{final['Aflood']:.3f}", f"{final['Aflood_range'][0]:.3f} - {final['Aflood_range'][1]:.3f}")
    ]

    # ---------------------
    # NEW Step 15: Fusion
    # ---------------------
    qpe, pnwp = inputs.get('qpe_radar', 45.0), inputs.get('p_nwp', 55.0)
    var_radar, var_nwp = inputs.get('var_radar', 20.0), inputs.get('var_nwp', 35.0)
    p_fused, alpha = fuse_precip(qpe, pnwp, var_radar, var_nwp)
    results['Step 15: Multi-Source Fusion'] = [
        ("QPE radar (mm/hr)", f"{qpe:.2f}", ""),
        ("NWP precip (mm/hr)", f"{pnwp:.2f}", ""),
        ("Var radar", f"{var_radar:.2f}", ""),
        ("Var NWP", f"{var_nwp:.2f}", ""),
        ("alpha (radar weight)", f"{alpha:.3f}", "Eq.(4)"),
        ("Pfused (mm/hr)", f"{p_fused:.3f}", "Eq.(3)")
    ]

    # ---------------------
    # NEW Step 16: Chaos-perturbed CN, K
    # ---------------------
    # reuse psi_vals as spatial basis at (x_pt,y_pt)
    CN_adj_point, K_adj_point = chaos_perturb_params(CN_mean, inputs['K_mean'], X_modes, psi_vals, sigma_CN=2.5, sigma_K=0.05)
    results['Step 16: Chaos-Perturbed Parameters'] = [
        ("CN mean", f"{CN_mean:.2f}", ""),
        ("K mean (m/s)", f"{inputs['K_mean']:.2e}", ""),
        ("Basis@pt", str([round(v, 4) for v in psi_vals]), ""),
        ("X_modes", str([round(v, 4) for v in X_modes]), ""),
        ("CN_chaos", f"{CN_adj_point:.3f}", "Eq.(16) at point"),
        ("K_chaos (m/s)", f"{K_adj_point:.3e}", "Eq.(17) at point")
    ]

    # ---------------------
    # NEW Step 17: KGE & Bayesian weighting demo
    # ---------------------
    obs_ts = np.array([5.0, 85.0, 150.0])
    sim_A = np.array([6.0, 80.0, 160.0])   # e.g., baseline model
    sim_B = np.array([5.5, 88.0, 140.0])   # e.g., chaos-perturbed model
    kgeA, rA, aA, bA = kge_components(sim_A, obs_ts)
    kgeB, rB, aB, bB = kge_components(sim_B, obs_ts)
    # Bayesian weights updating sequentially across times with fixed sigma per model
    w = np.array([0.5, 0.5])
    sigma_models = np.array([10.0, 10.0])
    for t in range(len(obs_ts)):
        y_hat = np.array([sim_A[t], sim_B[t]])
        w = bayes_weight_update(w, obs_ts[t], y_hat, sigma_models)
    results['Step 17: KGE & Bayesian Weighting'] = [
        ("Model", "KGE", "(r, alpha, beta)"),
        ("A", f"{kgeA:.3f}", f"({rA:.2f}, {aA:.2f}, {bA:.2f})"),
        ("B", f"{kgeB:.3f}", f"({rB:.2f}, {aB:.2f}, {bB:.2f})"),
        ("Posterior weights", str([round(float(x), 3) for x in w]), "after 3 updates")
    ]

    # ---------------------
    # NEW Step 18: Probabilistic impact & risk mapping (proxy)
    # ---------------------
    # Build a tiny ensemble of peak Q from chaos vs baseline and random spreads
    Q_ens = np.array([rec['base']['Q'] for rec in timeline] + [rec['chaos']['Q'] for rec in timeline], dtype=float)
    # Convert to proxy h, v using linear maps from Step 14 scaling (h=0.01Q, v=0.02Q)
    h_ens = 0.01 * Q_ens
    v_ens = 0.02 * Q_ens
    p_h_ex = exceedance_probability(h_ens, threshold=final['hmax'] * 0.9)
    p_haz = hazard_exceedance_probability(h_ens, v_ens, Hcrit=final['hmax'] * final['vmax'] * (final['vmax'] + 0.5))
    results['Step 18: Probabilistic Risk (proxy)'] = [
        ("Ensemble size", f"{len(Q_ens)}", ""),
        ("P(h > 0.9·hmax)", f"{p_h_ex:.2f}", "Eq.(48) proxy"),
        ("P(Hazard > Hcrit)", f"{p_haz:.2f}", "Eq.(49) proxy")
    ]

    return results


def save_report_csv(results: Dict[str, List[Tuple[str, str, str]]], path="flood_report.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "key", "value", "note"])
        for step, rows in results.items():
            for row in rows:
                if len(row) == 3:
                    writer.writerow([step, row[0], row[1], row[2]])
                else:
                    writer.writerow([step, str(row), "", ""])
    print(f"Saved report to {path}")

# -----------------------------
# Interactive runner (updated)
# -----------------------------

def run_interactive():
    print("\n" + "=" * 64)
    print(" HYBRID FLOOD FRAMEWORK — CHAOS + SMC (TERMINAL REPORT)")
    print("=" * 64 + "\n")
    try:
        P_total = float(input('Enter total rainfall for storm (mm) [150]: ') or 150.0)
        CN_mean = float(input('Enter mean Curve Number CN [85]: ') or 85.0)
        K_mean = float(input('Enter hydraulic conductivity K (m/s) [1.2e-5]: ') or 1.2e-5)
        psi = float(input('Enter suction head psi (m) [0.15]: ') or 0.15)
        delta_theta = float(input('Enter moisture deficit Δtheta [0.25]: ') or 0.25)
        duration_hr = float(input('Enter storm duration (hours) [3]: ') or 3.0)
        rain5 = input('Enter previous 5-day rainfall (mm) as 5 numbers ["5 0 12 8 3"]: ') or "5 0 12 8 3"
        rain5_list = [float(x) for x in rain5.split()][:5]
        # New inputs for Step 15 fusion (optional)
        qpe_radar = float(input('Radar QPE (mm/hr) [45]: ') or 45)
        p_nwp = float(input('NWP precip (mm/hr) [55]: ') or 55)
        var_radar = float(input('Radar variance [20]: ') or 20)
        var_nwp = float(input('NWP variance [35]: ') or 35)
    except Exception as e:
        print('Input error:', e)
        return

    inputs = {
        'P_total': P_total,
        'CN_mean': CN_mean,
        'K_mean': K_mean,
        'psi': psi,
        'delta_theta': delta_theta,
        'duration_hr': duration_hr,
        'rain5_list': rain5_list,
        # New fusion inputs
        'qpe_radar': qpe_radar,
        'p_nwp': p_nwp,
        'var_radar': var_radar,
        'var_nwp': var_nwp,
    }

    results = collect_results(inputs)

    # Print each step as a neat table
    for step, rows in results.items():
        print("\n" + "-" * 60)
        print(step)
        print("-" * 60)
        if rows and isinstance(rows[0], tuple) and all(isinstance(c, str) for c in rows[0]):
            first = rows[0]
            if len(first) >= 2 and any(name.lower().startswith(("obs", "hour", "model")) for name in first):
                headers = list(first)
                data = [list(r) for r in rows[1:]]
                print_table(headers, data)
                continue
        headers = ["Key", "Value", "Note"]
        print_table(headers, [list(r) if len(r) == 3 else [r[0], r[1] if len(r) > 1 else "", ""] for r in rows])

    # Offer CSV save
    save = input('Save full report to CSV? (y/n) [y]: ') or "y"
    if save.strip().lower().startswith('y'):
        save_report_csv(results)
    print("\nReport complete.\n")
    


if __name__ == "__main__":
    run_interactive()
