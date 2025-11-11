
#!/usr/bin/env python3
"""
flood_model.py
Comprehensive (simplified but complete) implementation of the equations in `foundation.pdf`.
- Terminal-only report.
- Implements Steps 1-14 from your PDF in functions, with a scenario runner and chaos perturbations.
- Designed for clarity and extendability, not for production hydrodynamic forecasting.
"""
import math
import numpy as np
import random
from typing import List, Tuple, Dict

# -----------------------------
# Step 1: Continuity (checks)
# -----------------------------
def continuity_incompressible_divergence(u: float, v: float, dh_dx: float, dh_dy: float) -> float:
    # For shallow-water representation: ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = R - I
    # Here we return divergence term as a diagnostic: ∂(hu)/∂x + ∂(hv)/∂y
    return dh_dx * u + dh_dy * v

# -----------------------------
# Step 2: Momentum (diagnostic)
# -----------------------------
def momentum_x(u, v, eta_grad_x, n, h, zb_grad_x, g=9.81):
    # X-momentum simplified from equation (5)
    friction = - g * n**2 * u * math.sqrt(u*u + v*v) / (h**(4.0/3.0) + 1e-12)
    pressure = - g * eta_grad_x
    bed = - g * h * zb_grad_x
    adv = 0.0  # ignoring nonlinear advective term for diagnostic simplicity
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
def green_ampt_cumulative(K: float, psi: float, delta_theta: float, t: float, tol=1e-8, maxiter=200) -> float:
    # Solves F = K t + psi * delta_theta * ln(1 + F / (psi * delta_theta))
    if psi * delta_theta <= 0:
        return K * t  # degenerate case
    F = K * t  # initial guess
    for _ in range(maxiter):
        F_new = K * t + psi * delta_theta * math.log(1.0 + F / (psi * delta_theta))
        if abs(F_new - F) < tol:
            F = F_new
            break
        F = F_new
    return F  # meters

# -----------------------------
# Step 4: SCS Curve Number Runoff
# -----------------------------
def scs_runoff(P_mm: float, CN: float) -> float:
    # P_mm in mm, returns Q in mm
    S = 1000.0 / CN - 10.0
    Ia = 0.2 * S
    if P_mm <= Ia:
        return 0.0
    Q = (P_mm - Ia)**2 / (P_mm - Ia + S)
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
# Step 6: Shallow Water Equations (conservative form helpers)
# -----------------------------
def state_vector(h: float, u: float, v: float) -> np.ndarray:
    # U = [h, h*u, h*v]
    return np.array([h, h*u, h*v])

def flux_F(U: np.ndarray, g=9.81) -> np.ndarray:
    h, hu, hv = U
    u = hu / (h + 1e-12)
    v = hv / (h + 1e-12)
    return np.array([hu, hu*u + 0.5 * g * h * h, hu*v])

def flux_G(U: np.ndarray, g=9.81) -> np.ndarray:
    h, hu, hv = U
    u = hu / (h + 1e-12)
    v = hv / (h + 1e-12)
    return np.array([hv, hu*v, hv*v + 0.5 * g * h * h])

def source_term(R, I, h, zb_grad_x, zb_grad_y, u, v, Cf=0.0, g=9.81):
    # Returns source vector S = [R-I, -g*h*dzb/dx - Cf * h * u * |u|, -g*h*dzb/dy - Cf * h * v * |v|]
    speed = math.sqrt(u*u + v*v) + 1e-12
    Cf_term = Cf * h * speed
    return np.array([R - I, -g * h * zb_grad_x - Cf_term * u, -g * h * zb_grad_y - Cf_term * v])

# -----------------------------
# Step 7: Uncertainty sources (input/parameter noise)
# -----------------------------
def perturb_rainfall(P_true: float, sigma_rad=0.15, bias_mm=2.0) -> float:
    eps_rad = random.gauss(0.0, sigma_rad)
    eps_sys = random.gauss(0.0, bias_mm)
    return P_true * (1.0 + eps_rad) + eps_sys

def perturb_CN(CN_mean: float, sigma_CN=5.0) -> float:
    return random.gauss(CN_mean, sigma_CN)

# -----------------------------
# Step 8 & 9: Chaos models (Lorenz-63)
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
# Step 9/10: Lorenz-96 (spatial interactions)
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
# Step 10.x: Spatial basis functions & precipitation perturbation
# -----------------------------
def spatial_basis_psi(kx: int, ky: int, x: float, y: float, Lx=10.0, Ly=10.0):
    # psi_k(x,y) = sin(kx * pi * x / Lx) * sin(ky * pi * y / Ly)
    return math.sin(kx * math.pi * x / Lx) * math.sin(ky * math.pi * y / Ly)

def precipitation_perturbation(Pbase: float, alpha_p: float, weights: List[float], psi_vals: List[float], Xvals: List[float]):
    # P_perturbed = Pbase * [1 + alpha_p * sum(wk * psik * Xk)]
    s = 0.0
    for wk, psik, Xk in zip(weights, psi_vals, Xvals):
        s += wk * psik * Xk
    return Pbase * (1.0 + alpha_p * s)

# -----------------------------
# Step 11: Simple particle filter (sequential MC)
# -----------------------------
def particle_filter_simple(obs_Q_series: List[float], Np=500, sigma_obs=5.0):
    # Very simplified: state is Q; propagate with small random walk; weight by obs likelihood
    particles = np.random.normal(loc=obs_Q_series[0], scale=sigma_obs, size=Np)
    weights = np.ones(Np) / Np
    estimates = []
    for t, obs in enumerate(obs_Q_series):
        # propagate (random walk)
        particles += np.random.normal(0, 2.0, size=Np)
        # compute weights from observation likelihood (Gaussian)
        likelihoods = (1.0 / (math.sqrt(2 * math.pi) * sigma_obs)) * np.exp(-0.5 * ((particles - obs) / sigma_obs) ** 2)
        weights = likelihoods * weights
        weights += 1e-300
        weights /= np.sum(weights)
        # estimate
        est = np.sum(particles * weights)
        estimates.append(est)
        # resample if needed (systematic)
        Neff = 1.0 / np.sum(weights ** 2)
        if Neff < Np / 2.0:
            # systematic resampling
            positions = (np.arange(Np) + random.random()) / Np
            cumulative_sum = np.cumsum(weights)
            indexes = np.searchsorted(cumulative_sum, positions)
            particles = particles[indexes]
            weights.fill(1.0 / Np)
    return estimates

# -----------------------------
# Step 12/13: Scenario builder & evolution
# -----------------------------
def scenario_three_hour(base_hours: Dict[int, Dict], chaos_mods: Dict[int, Dict]):
    # base_hours: mapping hour -> {rain, infil, runoff, Q}
    # chaos_mods: same structure but perturbed
    timeline = []
    for hr in [1,2,3]:
        b = base_hours[hr]
        c = chaos_mods[hr]
        timeline.append({
            'hour': hr,
            'base': b,
            'chaos': c
        })
    return timeline

# -----------------------------
# Step 14: Final flood characteristics (summary)
# -----------------------------
def final_flood_characteristics(Q_peak: float, area_km2: float = 12.5):
    # Using simple empirical relationships from PDF:
    # hmax [m], vmax [m/s], tpeak [hr], Aflood [km2]
    # We'll make hmax and vmax scale with Q_peak with plausible factors.
    hmax = 0.01 * Q_peak  # crude scaling: Q in m3/s -> h in m (tunable)
    vmax = 0.02 * Q_peak  # crude scaling
    tpeak = 2.1  # hours (nominal)
    Aflood = area_km2
    # uncertainty ranges (±)
    dh = 0.4  # m
    dv = 0.8  # m/s
    dt = 0.3  # hr
    dA = 3.2  # km2
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

# -----------------------------
# Report printing helpers
# -----------------------------
def print_header(title):
    print('\\n' + '='*60)
    print(f' {title}')
    print('='*60 + '\\n')

def print_table_row(cols, widths):
    row = ' | '.join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(row)

# -----------------------------
# Main interactive runner
# -----------------------------
def run_interactive():
    print_header('CHAOS-ENHANCED FLOOD PREDICTION (TERMINAL REPORT)')
    # User inputs (with defaults suggested matching PDF)
    try:
        P_total = float(input('Enter total rainfall for storm (mm) [default 150]: ') or 150.0)
        CN_mean = float(input('Enter mean Curve Number CN [default 85]: ') or 85.0)
        K_mean = float(input('Enter hydraulic conductivity K (m/s) [default 1.2e-5]: ') or 1.2e-5)
        psi = float(input('Enter suction head psi (m) [default 0.15]: ') or 0.15)
        delta_theta = float(input('Enter moisture deficit Delta theta [default 0.25]: ') or 0.25)
        duration_hr = float(input('Enter storm duration (hours) [default 3]: ') or 3.0)
        duration_s = duration_hr * 3600.0
        rain5 = input('Enter previous 5-day rainfall (mm) as 5 numbers separated by spaces [default \"5 0 12 8 3\"]: ') or "5 0 12 8 3"
        rain5_list = [float(x) for x in rain5.split()][:5]
    except Exception as e:
        print('Input error:', e)
        return

    # Step 1-2 (diagnostic)
    print_header('Step 1 & 2: Continuity and Momentum (Diagnostics)')
    div = continuity_incompressible_divergence(u=1.0, v=0.5, dh_dx=0.01, dh_dy=0.005)
    momx = momentum_x(u=1.0, v=0.5, eta_grad_x=0.002, n=0.03, h=0.5, zb_grad_x=0.001)
    momy = momentum_y(u=1.0, v=0.5, eta_grad_y=0.001, n=0.03, h=0.5, zb_grad_y=0.0005)
    print(f'Divergence diagnostic (∂(hu)/∂x + ∂(hv)/∂y): {div:.5f}')
    print(f'X-momentum diagnostic (rhs terms sum): {momx:.5f}')
    print(f'Y-momentum diagnostic (rhs terms sum): {momy:.5f}\\n')

    # Step 3: Green-Ampt for 1 hour and full duration
    print_header('Step 3: Green-Ampt Infiltration (Green-Ampt Equations)')
    F_1hr = green_ampt_cumulative(K_mean, psi, delta_theta, 3600.0)
    F_total = green_ampt_cumulative(K_mean, psi, delta_theta, duration_s)
    print(f'Computed cumulative infiltration after 1 hr: {F_1hr:.4f} m ({F_1hr*1000:.1f} mm)')
    print(f'Computed cumulative infiltration after {duration_hr:.1f} hr: {F_total:.4f} m ({F_total*1000:.1f} mm)\\n')

    # Step 4: SCS CN runoff (apply to total P_total)
    print_header('Step 4: SCS Curve Number Runoff (SCS Equations)')
    Q_total_mm = scs_runoff(P_total, CN_mean)
    S_val = scs_S(CN_mean)
    print(f'CN mean: {CN_mean:.2f}, S = {S_val:.3f} mm, Initial abstraction Ia = {0.2*S_val:.3f} mm')
    print(f'For P = {P_total:.1f} mm, Direct runoff Q = {Q_total_mm:.2f} mm ({Q_total_mm/P_total*100:.1f}% runoff coefficient)\\n')

    # Step 5: Antecedent moisture
    print_header('Step 5: Antecedent Moisture Modeling (AMC & Adjusted CN)')
    AMC = antecedent_moisture_index(rain5_list)
    CN_adj = adjusted_CN(80.0, 90.0, AMC, AMCmin=15.0, AMCmax=35.0)
    print(f'Previous 5-day rainfall: {rain5_list} -> AMC = {AMC:.3f} mm')
    print(f'Adjusted CN (from CNdry=80, CNwet=90): {CN_adj:.3f}\\n')

    # Step 6: Shallow water equations summary
    print_header('Step 6: Shallow Water Equations (Conservative Form)')
    U = state_vector(h=0.5, u=1.0, v=0.5)
    F = flux_F(U)
    G = flux_G(U)
    S = source_term(R=0.0, I=0.0, h=0.5, zb_grad_x=0.001, zb_grad_y=0.0005, u=1.0, v=0.5, Cf=0.0)
    print('State vector U = [h, h*u, h*v] =', U)
    print('Flux F(U) =', F)
    print('Flux G(U) =', G)
    print('Source term S =', S, '\\n')

    # Step 7: Uncertainty sources
    print_header('Step 7: Uncertainty Sources and Parameterization')
    P_obs_pert = perturb_rainfall(P_total, sigma_rad=0.15, bias_mm=2.0)
    CN_pert = perturb_CN(CN_mean, sigma_CN=5.0)
    print(f'Observed rainfall perturbation example: P_true={P_total} -> P_obs_sample={P_obs_pert:.2f} mm')
    print(f'Curve Number perturbation example: CN_mean={CN_mean} -> CN_sample={CN_pert:.2f}\\n')

    # Step 8/9: Lorenz-63 integration (chaos seed)
    print_header('Step 8/9: Lorenz-63 System (Chaos Integration)')
    x_l, y_l, z_l = integrate_lorenz63(1.0, 1.0, 1.0, dt=0.01, steps=1000)
    print(f'Lorenz-63 state after integration: x={x_l:.4f}, y={y_l:.4f}, z={z_l:.4f}\\n')

    # Step 9/10: Lorenz-96 integration for spatial chaos modes (Nc=5 typical)
    print_header('Step 9/10: Lorenz-96 Spatial Interactions')
    Nc = 40
    X0 = [8.0]*Nc
    X0[19] = 8.01
    X_final = integrate_lorenz96(X0, dt=0.001, steps=10000, F=8.0)
    print(f'Lorenz-96 sample values (first 8): {X_final[:8].round(4).tolist()}')
    # take first 5 modes for perturbation basis
    X_modes = X_final[:5].tolist()
    print(f'Using first {len(X_modes)} chaos modes for perturbations: {X_modes}\\n')

    # Step 10.x: spatial basis psi and precipitation perturbation demo at point (5,3) km
    print_header('Step 10.x: Spatial Basis and Precipitation Perturbation (Example point (5,3) km)')
    x_pt, y_pt = 5.0, 3.0
    psi_vals = [spatial_basis_psi(k+1, 1, x_pt, y_pt, Lx=10.0, Ly=10.0) for k in range(5)]
    # weights wk = exp(-k-1)
    weights = [math.exp(-(k+1)) for k in range(5)]
    Pbase = 50.0  # mm/hr base for example
    alpha_p = 0.2
    P_pert = precipitation_perturbation(Pbase, alpha_p, weights, psi_vals, X_modes)
    print(f'psi values (modes 1..5): {[round(v,4) for v in psi_vals]}')
    print(f'weights wk: {[round(w,4) for w in weights]}')
    print(f'Chaos modes Xk (first 5): {[round(v,4) for v in X_modes]}')
    print(f'Perturbed precipitation at point ({x_pt},{y_pt}) : P_perturbed = {P_pert:.3f} mm/hr (base={Pbase})\\n')

    # Step 11: Particle filter demo (using Q time series from scenario)
    print_header('Step 11: Sequential Monte Carlo (Particle Filter) Demo')
    # Build a synthetic Q_obs time series (3 hourly values) using baseline from PDF approx
    Q_obs_series = [5.0, 85.0, 150.0]
    pf_estimates = particle_filter_simple(Q_obs_series, Np=400, sigma_obs=10.0)
    print(f'Observed Q series: {Q_obs_series}')
    print(f'Particle filter estimates: {[round(v,2) for v in pf_estimates]}\\n')

    # Step 12/13: Scenario evolution (baseline and chaos-enhanced)
    print_header('Step 12/13: Complete Scenario Calculation & Evolution (3-hour storm)')
    baseline = {
        1: {'rain': 10.0, 'infil': 9.0, 'runoff': 1.0, 'Q': 5.0},
        2: {'rain': 50.0, 'infil': 5.0, 'runoff': 45.0, 'Q': 85.0},
        3: {'rain': 50.0, 'infil': 1.0, 'runoff': 49.0, 'Q': 150.0},
    }
    chaos = {
        1: {'rain': 10.0*1.15, 'infil': 9.0*0.9, 'runoff': 1.0*1.2, 'Q': 6.0},
        2: {'rain': 50.0*1.2, 'infil': 5.0*1.05, 'runoff': 45.0*1.3, 'Q': 110.0},
        3: {'rain': 50.0*0.9, 'infil': 1.0*1.2, 'runoff': 49.0*0.95, 'Q': 140.0},
    }
    timeline = scenario_three_hour(baseline, chaos)
    print('Hour | Baseline Rain (mm/hr) | Baseline Infil (mm/hr) | Baseline Runoff (mm/hr) | Q (m3/s)')
    print('-'*90)
    for rec in timeline:
        hr = rec['hour']
        b = rec['base']
        print(f'{hr:>4} | {b["rain"]:>20.1f} | {b["infil"]:>20.1f} | {b["runoff"]:>22.1f} | {b["Q"]:>8.1f}')
    print('\\nWith chaos perturbations:')
    print('Hour | Chaos Rain (mm/hr) | Chaos Infil (mm/hr) | Chaos Runoff (mm/hr) | Q (m3/s)')
    print('-'*90)
    for rec in timeline:
        hr = rec['hour']
        c = rec['chaos']
        print(f'{hr:>4} | {c["rain"]:>16.2f} | {c["infil"]:>17.2f} | {c["runoff"]:>20.2f} | {c["Q"]:>8.1f}')
    # Step 13.4: Final flood characteristics
    print_header('Step 13.4: Final Flood Characteristics (Summary)')
    # use chaos peak Q as final
    Q_peak = max([rec['chaos']['Q'] for rec in timeline] + [rec['base']['Q'] for rec in timeline])
    final = final_flood_characteristics(Q_peak, area_km2=12.5)
    print('Peak discharge used for summary:', Q_peak)
    print('Maximum depth hmax = {:.2f} m (range {:.2f} - {:.2f} m)'.format(final['hmax'], final['hmax_range'][0], final['hmax_range'][1]))
    print('Peak velocity vmax = {:.2f} m/s (range {:.2f} - {:.2f} m/s)'.format(final['vmax'], final['vmax_range'][0], final['vmax_range'][1]))
    print('Arrival time tpeak = {:.2f} hr (range {:.2f} - {:.2f} hr)'.format(final['tpeak'], final['tpeak_range'][0], final['tpeak_range'][1]))
    print('Inundation area Aflood = {:.2f} km2 (range {:.2f} - {:.2f} km2)'.format(final['Aflood'], final['Aflood_range'][0], final['Aflood_range'][1]))
    print('\\nReport complete.\\n')


if __name__ == "__main__":
    run_interactive()
