"""
Optimized flood backend (numba + multiprocessing)

Notes:
- Requires: numpy, numba
- To run ensembles in parallel on Windows, ensure the usual
  `if __name__ == "__main__":` guard around top-level calls that create pools.
- This preserves the physics and numerics in your original file
  (HLLC + hydrostatic reconstruction + SSP-RK3 + SCS + Lorenz).
- For reference and algorithmic choices see the uploaded PDF. :contentReference[oaicite:2]{index=2}
"""

import numpy as np
from numba import njit, prange
from multiprocessing import Pool, cpu_count
import multiprocessing
import os

from hec_ras_integration import run_hecras_and_get_results


g = 9.80665
tiny = 1e-12

# --------------------------
# Utility helpers (numba)
# --------------------------
@njit
def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

# --------------------------
# Lorenz (njit)
# --------------------------
@njit
def lorenz_rhs(X, Y, Z, sigma, rho, beta):
    return sigma * (Y - X), X * (rho - Z) - Y, X * Y - beta * Z

@njit
def lorenz_step_rk4(X, Y, Z, dt, sigma, rho, beta):
    k1x, k1y, k1z = lorenz_rhs(X, Y, Z, sigma, rho, beta)
    x2 = X + 0.5*dt*k1x; y2 = Y + 0.5*dt*k1y; z2 = Z + 0.5*dt*k1z
    k2x, k2y, k2z = lorenz_rhs(x2, y2, z2, sigma, rho, beta)
    x3 = X + 0.5*dt*k2x; y3 = Y + 0.5*dt*k2y; z3 = Z + 0.5*dt*k2z
    k3x, k3y, k3z = lorenz_rhs(x3, y3, z3, sigma, rho, beta)
    x4 = X + dt*k3x; y4 = Y + dt*k3y; z4 = Z + dt*k3z
    k4x, k4y, k4z = lorenz_rhs(x4, y4, z4, sigma, rho, beta)
    X += dt/6.0*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    Y += dt/6.0*(k1y + 2.0*k2y + 2.0*k3y + k4y)
    Z += dt/6.0*(k1z + 2.0*k2z + 2.0*k3z + k4z)
    return X, Y, Z

@njit
def lorenz_modulation(X, alpha_P, sigma_X):
    phi = (X / sigma_X) * np.exp(-0.5 * (X / (3.0 * sigma_X))**2)
    return 1.0 + alpha_P * phi

# --------------------------
# SCS infiltration (njit)
# --------------------------
@njit
def scs_precompute_S(CN):
    # CN scalar or array element
    return 25.4 / CN - 0.254

@njit
def scs_cumulative_runoff(P_cum, Ia, S):
    # P_cum: cumulative precipitation scalar or array element
    if P_cum <= Ia:
        return 0.0
    num = (P_cum - Ia) * (P_cum - Ia)
    den = P_cum - Ia + S
    return num / den

@njit
def scs_instantaneous_infiltration(P_inst, P_cum, dPdt, Ia, S):
    # scalar variant; used elementwise
    if P_cum <= Ia:
        return P_inst
    Pm = P_cum
    # derivative-based formula
    num = (2.0*(Pm - Ia)*(Pm - Ia + S) - (Pm - Ia)**2)
    den = (Pm - Ia + S)**2
    dQdP = num / den
    dQdt = dQdP * dPdt
    I_inst = dPdt - dQdt
    if I_inst < 0.0:
        I_inst = 0.0
    return I_inst

# --------------------------
# HLLC flux (njit, scalar)
# --------------------------
@njit
def hllc_flux_1d_scalar(hL, uL, vL, hR, uR, vR, g_local):
    # ensure non-negative
    if hL < 0.0: hL = 0.0
    if hR < 0.0: hR = 0.0
    huL = hL * uL
    huR = hR * uR
    hvL = hL * vL
    hvR = hR * vR
    cL = 0.0
    cR = 0.0
    if hL > 1e-12:
        cL = np.sqrt(g_local * hL)
    if hR > 1e-12:
        cR = np.sqrt(g_local * hR)
    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)
    # physical fluxes
    FhL = huL
    FhuL = hL * uL * uL + 0.5 * g_local * hL * hL
    FhvL = hL * uL * vL
    FhR = huR
    FhuR = hR * uR * uR + 0.5 * g_local * hR * hR
    FhvR = hR * uR * vR
    if SR - SL < 1e-14:
        return 0.5*(FhL + FhR), 0.5*(FhuL + FhuR), 0.5*(FhvL + FhvR)
    S_star = (SR * uR - SL * uL + 0.5 * g_local * (hL*hL - hR*hR)) / (SR - SL)
    # star depths
    if abs(SL - S_star) > 1e-15:
        hstarL = hL * (SL - uL) / (SL - S_star)
    else:
        hstarL = hL
    if abs(SR - S_star) > 1e-15:
        hstarR = hR * (SR - uR) / (SR - S_star)
    else:
        hstarR = hR
    # select flux
    if 0.0 <= SL:
        return FhL, FhuL, FhvL
    elif SL < 0.0 <= S_star:
        hu_starL = hstarL * S_star
        hv_starL = hstarL * vL
        Fh_starL = FhL + SL * (hstarL - hL)
        Fhu_starL = FhuL + SL * (hu_starL - huL)
        Fhv_starL = FhvL + SL * (hv_starL - hvL)
        return Fh_starL, Fhu_starL, Fhv_starL
    elif S_star < 0.0 <= SR:
        hu_starR = hstarR * S_star
        hv_starR = hstarR * vR
        Fh_starR = FhR + SR * (hstarR - hR)
        Fhu_starR = FhuR + SR * (hu_starR - huR)
        Fhv_starR = FhvR + SR * (hv_starR - hvR)
        return Fh_starR, Fhu_starR, Fhv_starR
    else:
        return FhR, FhuR, FhvR

# --------------------------
# Spatial operator (numba parallel)
# --------------------------
@njit(parallel=True, fastmath=True)
def spatial_operator_numba(h, u, v, zb, dx, dy, g_local):
    Ny, Nx = h.shape
    rhs_h = np.zeros_like(h)
    rhs_hu = np.zeros_like(h)
    rhs_hv = np.zeros_like(h)

    # flux arrays (face-centered)
    Fx_h = np.zeros((Ny, Nx+1))
    Fx_hu = np.zeros((Ny, Nx+1))
    Fx_hv = np.zeros((Ny, Nx+1))

    # x-direction faces
    for j in prange(Ny):
        for i_face in range(Nx+1):
            iL = i_face - 1
            iR = i_face
            if iL < 0:
                hL = h[j, 0]; uL = -u[j, 0]; vL = v[j, 0]; zbL = zb[j, 0]
            else:
                hL = h[j, iL]; uL = u[j, iL]; vL = v[j, iL]; zbL = zb[j, iL]
            if iR >= Nx:
                hR = h[j, -1]; uR = -u[j, -1]; vR = v[j, -1]; zbR = zb[j, -1]
            else:
                hR = h[j, iR]; uR = u[j, iR]; vR = v[j, iR]; zbR = zb[j, iR]

            # hydrostatic reconstruction
            etaL = hL + zbL
            etaR = hR + zbR
            zbmax = zbL if zbL > zbR else zbR
            hLr = etaL - zbmax
            if hLr < 0.0: hLr = 0.0
            hRr = etaR - zbmax
            if hRr < 0.0: hRr = 0.0

            uLr = uL if hLr > 1e-12 else 0.0
            vLr = vL if hLr > 1e-12 else 0.0
            uRr = uR if hRr > 1e-12 else 0.0
            vRr = vR if hRr > 1e-12 else 0.0

            Fh, Fhu, Fhv = hllc_flux_1d_scalar(hLr, uLr, vLr, hRr, uRr, vRr, g_local)
            Fx_h[j, i_face] = Fh
            Fx_hu[j, i_face] = Fhu
            Fx_hv[j, i_face] = Fhv

    # y-direction faces (rotate variables)
    Gy_h = np.zeros((Ny+1, Nx))
    Gy_hu = np.zeros((Ny+1, Nx))
    Gy_hv = np.zeros((Ny+1, Nx))

    for i in prange(Nx):
        for j_face in range(Ny+1):
            jL = j_face - 1
            jR = j_face
            if jL < 0:
                hL = h[0, i]; uL = u[0, i]; vL = -v[0, i]; zbL = zb[0, i]
            else:
                hL = h[jL, i]; uL = u[jL, i]; vL = v[jL, i]; zbL = zb[jL, i]
            if jR >= Ny:
                hR = h[-1, i]; uR = u[-1, i]; vR = -v[-1, i]; zbR = zb[-1, i]
            else:
                hR = h[jR, i]; uR = u[jR, i]; vR = v[jR, i]; zbR = zb[jR, i]

            etaL = hL + zbL
            etaR = hR + zbR
            zbmax = zbL if zbL > zbR else zbR
            hLr = etaL - zbmax
            if hLr < 0.0: hLr = 0.0
            hRr = etaR - zbmax
            if hRr < 0.0: hRr = 0.0

            # rotated velocities
            uLr = vL if hLr > 1e-12 else 0.0
            vLr = uL if hLr > 1e-12 else 0.0
            uRr = vR if hRr > 1e-12 else 0.0
            vRr = uR if hRr > 1e-12 else 0.0

            Fh, Fhu, Fhv = hllc_flux_1d_scalar(hLr, uLr, vLr, hRr, uRr, vRr, g_local)
            Gy_h[j_face, i] = Fh
            Gy_hv[j_face, i] = Fhu
            Gy_hu[j_face, i] = Fhv

    # divergence
    for j in prange(Ny):
        for i in range(Nx):
            rhs_h[j, i] = - (Fx_h[j, i+1] - Fx_h[j, i]) / dx - (Gy_h[j+1, i] - Gy_h[j, i]) / dy
            rhs_hu[j, i] = - (Fx_hu[j, i+1] - Fx_hu[j, i]) / dx - (Gy_hu[j+1, i] - Gy_hu[j, i]) / dy
            rhs_hv[j, i] = - (Fx_hv[j, i+1] - Fx_hv[j, i]) / dx - (Gy_hv[j+1, i] - Gy_hv[j, i]) / dy

    # bed slope source
    dzbdx = np.zeros_like(zb)
    dzbdy = np.zeros_like(zb)
    # dx derivatives
    for j in prange(Ny):
        for i in range(1, Nx-1):
            dzbdx[j, i] = (zb[j, i+1] - zb[j, i-1]) / (2.0 * dx)
        dzbdx[j, 0] = (zb[j, 1] - zb[j, 0]) / dx
        dzbdx[j, -1] = (zb[j, -1] - zb[j, -2]) / dx
    # dy derivatives
    for i in prange(Nx):
        for j in range(1, Ny-1):
            dzbdy[j, i] = (zb[j+1, i] - zb[j-1, i]) / (2.0 * dy)
        dzbdy[0, i] = (zb[1, i] - zb[0, i]) / dy
        dzbdy[-1, i] = (zb[-1, i] - zb[-2, i]) / dy

    for j in prange(Ny):
        for i in range(Nx):
            rhs_hu[j, i] += - g_local * h[j, i] * dzbdx[j, i]
            rhs_hv[j, i] += - g_local * h[j, i] * dzbdy[j, i]

    return rhs_h, rhs_hu, rhs_hv

# --------------------------
# Friction (semi-implicit style, njit)
# --------------------------
@njit(parallel=True, fastmath=True)
def apply_manning_friction_numba(h, u, v, n_arr, dt, g_local):
    Ny, Nx = h.shape
    hu = h * u
    hv = h * v
    # denom computed elementwise
    for j in prange(Ny):
        for i in range(Nx):
            hi = h[j, i]
            if hi <= 1e-12:
                u[j, i] = 0.0
                v[j, i] = 0.0
                continue
            speed = np.sqrt(u[j, i]*u[j, i] + v[j, i]*v[j, i])
            denom = 1.0 + dt * g_local * n_arr[j, i] * n_arr[j, i] * speed / (hi**(4.0/3.0) + 1e-16)
            hu_new = hu[j, i] / denom
            hv_new = hv[j, i] / denom
            u[j, i] = hu_new / hi
            v[j, i] = hv_new / hi

# --------------------------
# Precipitation + infiltration (elementwise loops)
# --------------------------
@njit(parallel=True, fastmath=True)
def apply_precip_infiltration_numba(h, P_field, P_cum, Q_cum, CN_field, dt):
    Ny, Nx = h.shape
    # precompute S and Ia elementwise
    for j in prange(Ny):
        for i in range(Nx):
            S = scs_precompute_S(CN_field[j, i])
            Ia = 0.2 * S
            dP = P_field[j, i] * dt
            P_cum[j, i] += dP
            Qnew = scs_cumulative_runoff(P_cum[j, i], Ia, S)
            dQ = Qnew - Q_cum[j, i]
            Q_cum[j, i] = Qnew
            dF = dP - dQ
            # apply precipitation (mass)
            h[j, i] += dP
            # subtract infiltration loss (dF) -> effectively h += dQ
            h[j, i] -= dF
            if h[j, i] < 0.0:
                h[j, i] = 0.0

# --------------------------
# CFL timestep (njit)
# --------------------------
@njit
def compute_timestep_numba(h, u, v, dx, dy, alpha_CFL, dt_max, dt_min, g_local):
    c = np.sqrt(g_local * np.maximum(h, 0.0))
    lam_x = np.abs(u) + c
    lam_y = np.abs(v) + c
    lam_max = 0.0
    # compute global max
    nrows, ncols = lam_x.shape
    for j in range(nrows):
        for i in range(ncols):
            if lam_x[j, i] > lam_max:
                lam_max = lam_x[j, i]
            if lam_y[j, i] > lam_max:
                lam_max = lam_y[j, i]
    if lam_max <= 0.0:
        return dt_max
    dt1 = alpha_CFL * min(dx, dy) / lam_max
    if dt1 < dt_min:
        dt1 = dt_min
    if dt1 > dt_max:
        dt1 = dt_max
    return dt1

# --------------------------
# Pack / unpack state (simple wrappers)
# --------------------------
@njit
def pack_state_numba(h, u, v):
    Ny, Nx = h.shape
    U = np.zeros((3, Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            U[0, j, i] = h[j, i]
            U[1, j, i] = h[j, i] * u[j, i]
            U[2, j, i] = h[j, i] * v[j, i]
    return U

@njit
def unpack_state_numba(U, h, u, v):
    Ny, Nx = h.shape
    for j in range(Ny):
        for i in range(Nx):
            h[j, i] = U[0, j, i]
            hu = U[1, j, i]
            hv = U[2, j, i]
            if h[j, i] > 1e-12:
                u[j, i] = hu / h[j, i]
                v[j, i] = hv / h[j, i]
            else:
                u[j, i] = 0.0
                v[j, i] = 0.0

# --------------------------
# Main simulation (numba driver wrapper + python orchestration)
# --------------------------
def simulate_flood_member(params):
    """
    This function is the entry point for a single deterministic member.
    It's arranged as a pure-Python function that calls numba kernels repeatedly.
    It returns time series arrays.
    """
    Nx = int(params.get("Nx", 40))
    Ny = int(params.get("Ny", 20))
    Lx = float(params.get("Lx", 20000.0))
    Ly = float(params.get("Ly", 10000.0))
    CN = float(params.get("CN", 70.0))
    manning_n = float(params.get("manning_n", 0.03))
    t_end = float(params.get("t_end", 60.0))
    output_interval = float(params.get("output_interval", 10.0))
    seed_pond = bool(params.get("seed_pond", True))
    Pmax_mm_h = float(params.get("Pmax_mm_h", 100.0))
    Rmax_km = float(params.get("Rmax_km", 30.0))
    alphaP = float(params.get("alphaP", 0.15))
    sigmaX = float(params.get("sigmaX", 7.89))
    sigma = float(params.get("lorenz_sigma", 10.0))
    rho = float(params.get("lorenz_rho", 28.0))
    beta = float(params.get("lorenz_beta", 8.0/3.0))

    # create grid arrays (C-contiguous)
    dx = Lx / Nx
    dy = Ly / Ny
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    Xgrid, Ygrid = np.meshgrid(xc, yc)

    h = np.zeros((Ny, Nx), dtype=np.float64)
    u = np.zeros((Ny, Nx), dtype=np.float64)
    v = np.zeros((Ny, Nx), dtype=np.float64)
    zb = np.zeros((Ny, Nx), dtype=np.float64)
    n_arr = manning_n * np.ones((Ny, Nx), dtype=np.float64)
    CN_field = CN * np.ones((Ny, Nx), dtype=np.float64)
    P_cum = np.zeros((Ny, Nx), dtype=np.float64)
    Q_cum = np.zeros((Ny, Nx), dtype=np.float64)

    # set bed gaussian (match your original)
    peak = params.get("bed_peak", 300.0)
    x0 = params.get("bed_x0", Lx/2.0)
    y0 = params.get("bed_y0", Ly/2.0)
    sigma_bed = params.get("bed_sigma", 0.25*Lx)
    dxg = Xgrid - x0
    dyg = Ygrid - y0
    zb = peak * np.exp(-0.5 * (dxg*dxg + dyg*dyg) / (sigma_bed*sigma_bed))

    if seed_pond:
        cx = Nx // 2
        cy = Ny // 2
        x0i = max(cx-1, 0); x1i = min(cx+2, Nx)
        y0i = max(cy-1, 0); y1i = min(cy+2, Ny)
        h[y0i:y1i, x0i:x1i] = 0.5

    # Lorenz initial state
    Xl = 1.0; Yl = 1.0; Zl = 1.0
    dt_chaos = 0.01

    times = []
    mean_rain = []
    max_h_list = []
    total_volume = []
    lorenz_x = []
    lorenz_y = []
    lorenz_z = []

    t = 0.0
    step = 0
    next_output = output_interval

    eye_x_m = Lx / 2.0
    eye_y_m = Ly / 2.0
    vtrans = 5.556  # m/s

    # helper to compute Pbase vectorized (same as your function but vectorized)
    def compute_typhoon_precip_field_local(Pmax_mm_per_h, Rmax_km, Xg, Yg, eye_x_m_local, eye_y_m_local):
        dx_ = Xg - eye_x_m_local
        dy_ = Yg - eye_y_m_local
        r_km = np.sqrt(dx_*dx_ + dy_*dy_) / 1000.0
        r1 = 0.5 * Rmax_km
        r2 = 3.0 * Rmax_km
        Pbase_mm_h = Pmax_mm_per_h * (1.0 - np.exp(-r_km / r1)) * np.exp(-r_km / r2)
        Pbase_m_s = (Pbase_mm_h * 1e-3) / 3600.0
        return Pbase_m_s

    # main time loop
    while t < t_end - 1e-12:
        dt = compute_timestep_numba(h, u, v, dx, dy, 0.3, 1.0, 1e-6, g)
        # integrate Lorenz with substeps
        n_chaos = max(1, int(np.ceil(dt / dt_chaos)))
        for _ in range(n_chaos):
            Xl, Yl, Zl = lorenz_step_rk4(Xl, Yl, Zl, dt_chaos, sigma, rho, beta)

        lam = lorenz_modulation(Xl, alphaP, sigmaX)
        Pbase = compute_typhoon_precip_field_local(Pmax_mm_h, Rmax_km, Xgrid, Ygrid, eye_x_m, eye_y_m)
        Ptotal = Pbase * lam

        # record
        times.append(t)
        mean_rain.append(np.mean(Ptotal) * 3600.0 * 1000.0)
        max_h_list.append(np.max(h))
        total_volume.append(np.sum(h) * dx * dy)
        lorenz_x.append(Xl); lorenz_y.append(Yl); lorenz_z.append(Zl)

        # SSP-RK3 stages (call spatial operator and source terms)
        U0 = pack_state_numba(h, u, v)

        # Stage 1
        rhs_h, rhs_hu, rhs_hv = spatial_operator_numba(h, u, v, zb, dx, dy, g)
        U1 = U0.copy()
        NyS, NxS = h.shape
        for jj in range(NyS):
            for ii in range(NxS):
                U1[0, jj, ii] = U0[0, jj, ii] + dt * rhs_h[jj, ii]
                U1[1, jj, ii] = U0[1, jj, ii] + dt * rhs_hu[jj, ii]
                U1[2, jj, ii] = U0[2, jj, ii] + dt * rhs_hv[jj, ii]
        unpack_state_numba(U1, h, u, v)
        apply_precip_infiltration_numba(h, Ptotal, P_cum, Q_cum, CN_field, dt)
        rhs_h, rhs_hu, rhs_hv = spatial_operator_numba(h, u, v, zb, dx, dy, g)

        # Stage 2
        U2 = np.zeros_like(U0)
        for jj in range(NyS):
            for ii in range(NxS):
                U2[0, jj, ii] = 0.75*U0[0, jj, ii] + 0.25*(U1[0, jj, ii] + dt*rhs_h[jj, ii])
                U2[1, jj, ii] = 0.75*U0[1, jj, ii] + 0.25*(U1[1, jj, ii] + dt*rhs_hu[jj, ii])
                U2[2, jj, ii] = 0.75*U0[2, jj, ii] + 0.25*(U1[2, jj, ii] + dt*rhs_hv[jj, ii])
        unpack_state_numba(U2, h, u, v)
        apply_precip_infiltration_numba(h, Ptotal, P_cum, Q_cum, CN_field, dt)
        rhs_h, rhs_hu, rhs_hv = spatial_operator_numba(h, u, v, zb, dx, dy, g)

        # Stage 3 (final)
        Unew = np.zeros_like(U0)
        for jj in range(NyS):
            for ii in range(NxS):
                Unew[0, jj, ii] = (1.0/3.0)*U0[0, jj, ii] + (2.0/3.0)*(U2[0, jj, ii] + dt*rhs_h[jj, ii])
                Unew[1, jj, ii] = (1.0/3.0)*U0[1, jj, ii] + (2.0/3.0)*(U2[1, jj, ii] + dt*rhs_hu[jj, ii])
                Unew[2, jj, ii] = (1.0/3.0)*U0[2, jj, ii] + (2.0/3.0)*(U2[2, jj, ii] + dt*rhs_hv[jj, ii])
        unpack_state_numba(Unew, h, u, v)

        apply_precip_infiltration_numba(h, Ptotal, P_cum, Q_cum, CN_field, dt)
        apply_manning_friction_numba(h, u, v, n_arr, dt, g)

        # advect eye
        eye_x_m -= vtrans * dt

        t += dt
        step += 1

        if t >= next_output - 1e-9 or t >= t_end - 1e-9:
            # minimal print for monitoring
            print(f"[member] t={t:.2f}s step {step}, max h = {np.max(h):.4f} m")
            next_output += output_interval

    # finalize and pack timeseries
    times.append(t)
    mean_rain.append(np.mean(Ptotal) * 3600.0 * 1000.0)
    max_h_list.append(np.max(h))
    total_volume.append(np.sum(h) * dx * dy)
    lorenz_x.append(Xl); lorenz_y.append(Yl); lorenz_z.append(Zl)

    times = np.array(times)
    mean_rain = np.array(mean_rain)
    max_h_list = np.array(max_h_list)
    total_volume = np.array(total_volume)
    lorenz_x = np.array(lorenz_x); lorenz_y = np.array(lorenz_y); lorenz_z = np.array(lorenz_z)
    dV = np.diff(total_volume, prepend=total_volume[0])
    avg_dt = times[1] - times[0] if len(times) > 1 else 1.0
    discharge_proxy = dV / avg_dt

    results = {
        "time": times,
        "mean_raiin_mmhr": mean_rain,
        "max_h": max_h_list,
        "total_volume_m3": total_volume,
        "discharge_m3s": discharge_proxy,
        "lorenz_x": lorenz_x,
        "lorenzz_y": lorenz_y,
        "lorenz_z": lorenz_z,
        "flood_depth": h,
    }
    return results

# --------------------------
# High-level run that does ensembles with multiprocessing
# --------------------------
def run_flood_simulation(params):
    """
    Runs deterministic member and builds ensembles in parallel.
    Returns a dict of results with ensemble array.
    """
    # run single member (deterministic) to get baseline and times
    baseline = simulate_flood_member(params)
    times = baseline["time"]
    discharge = baseline["discharge_m3s"]
    n_ens = int(params.get("n_ens", 20))
    spread = max(1e-9, 0.05 * np.nanmax(np.abs(discharge)) if discharge.size>0 else 1e-6)

    # run ensemble members in parallel by perturbing parameters slightly
    # create param copies
    param_list = []
    for k in range(n_ens):
        p = params.copy()
        # small random seed perturbation in Lorenz initial condition / Pmax to create ensemble spread
        p["Pmax_mm_h"] = params.get("Pmax_mm_h", 100.0) * (1.0 + np.random.normal(0.0, 0.01))
        param_list.append(p)

    # choose number of processes
    nprocs = min(cpu_count(), n_ens)
    # Windows-friendly guard: user should call this function from main script guard.
    with Pool(processes=nprocs) as pool:
        ens_results = pool.map(simulate_flood_member, param_list)

    ensembles = np.array([res["discharge_m3s"] for res in ens_results])

    results = {
        "time": times,
        "rain": baseline["mean_rain_mmhr"],
        "discharge": discharge,
        "ensembles": ensembles,
        "flood_depth": baseline["flood_depth"],
        "lorenz_x": baseline["lorenz_x"],
        "lorenz_y": baseline["lorenz_y"],
        "lorenz_z": baseline["lorenz_z"],
        "max_h_ts": baseline["max_h"],
        "total_volume": baseline["total_volume_m3"],
    }
    return results

# Quick test when run as script
if __name__ == "__main__":
    # === 1. RUN HEC-RAS FIRST ===
    print("[SYS] Launching HEC-RAS…")

    bc = run_hecras_and_get_results(
        project=r"C:\Path\To\Your\HECRASProject.prj",   # <-- CHANGE THIS
        plan="Plan 1",                                  # <-- CHANGE THIS
        results_hint=None                               # optional
    )

    print("[SYS] HEC-RAS finished.")
    print("[SYS] Results file:", bc["hec_results_path"])
    print("[SYS] Parsed top-level:", bc["raw_parsed"])

    # TODO: Convert bc → your model’s boundary conditions here
    # Example:
    # inflow_bc = convert_hecras_to_model(bc)

    # === 2. RUN YOUR PYTHON FLOOD MODEL ===
    multiprocessing.set_start_method("spawn", force=False)

    params = {
        "Nx": 80,
        "Ny": 40,
        "Lx": 20000.0,
        "Ly": 10000.0,
        "t_end": 30.0,
        "output_interval": 10.0,
        "n_ens": 4
    }

    # You can pass HEC-RAS data into your simulation here, e.g.:
    # res = run_flood_simulation(params, boundary_conditions=bc)

    res = run_flood_simulation(params)

    print("Done. time samples:", res["time"].shape)
    print("Max final depth:", np.max(res["flood_depth"]))
