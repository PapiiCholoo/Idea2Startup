import numpy as np

# Physical constants
g = 9.80665

# --------------------------
# Utility functions/classes
# --------------------------
def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

class Lorenz63:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, X0=1.0, Y0=1.0, Z0=1.0):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self.X = float(X0)
        self.Y = float(Y0)
        self.Z = float(Z0)

    def rhs(self, X, Y, Z):
        dX = self.sigma * (Y - X)
        dY = X * (self.rho - Z) - Y
        dZ = X * Y - self.beta * Z
        return dX, dY, dZ

    def step_rk4(self, dt):
        k1x, k1y, k1z = self.rhs(self.X, self.Y, self.Z)
        x2 = self.X + 0.5*dt*k1x
        y2 = self.Y + 0.5*dt*k1y
        z2 = self.Z + 0.5*dt*k1z
        k2x, k2y, k2z = self.rhs(x2, y2, z2)

        x3 = self.X + 0.5*dt*k2x
        y3 = self.Y + 0.5*dt*k2y
        z3 = self.Z + 0.5*dt*k2z
        k3x, k3y, k3z = self.rhs(x3, y3, z3)

        x4 = self.X + dt*k3x
        y4 = self.Y + dt*k3y
        z4 = self.Z + dt*k3z
        k4x, k4y, k4z = self.rhs(x4, y4, z4)

        self.X += dt/6.0*(k1x + 2.0*k2x + 2.0*k3x + k4x)
        self.Y += dt/6.0*(k1y + 2.0*k2y + 2.0*k3y + k4y)
        self.Z += dt/6.0*(k1z + 2.0*k2z + 2.0*k3z + k4z)

    def modulation_factor(self, alpha_P=0.15, sigma_X=7.89):
        # Eq. (50) and (49) style
        X = self.X
        phi = (X / sigma_X) * np.exp(-0.5 * (X / (3.0 * sigma_X))**2)
        lam = 1.0 + alpha_P * phi
        return float(lam)

# --------------------------
# Precipitation model
# --------------------------
def willoughby_rahn_r(Pmax_mm_per_h, Rmax_km, r_km, r1_factor=0.5, r2_factor=3.0):
    r1 = r1_factor * Rmax_km
    r2 = r2_factor * Rmax_km
    Pbase_mm_h = Pmax_mm_per_h * (1.0 - np.exp(-r_km / r1)) * np.exp(-r_km / r2)
    Pbase_m_s = (Pbase_mm_h * 1e-3) / 3600.0
    return Pbase_m_s

def compute_typhoon_precip_field(Pmax_mm_h, Rmax_km, X_grid_m, Y_grid_m, eye_x_m, eye_y_m):
    dx = (X_grid_m - eye_x_m)
    dy = (Y_grid_m - eye_y_m)
    r_km = np.sqrt(dx*dx + dy*dy) / 1000.0
    return willoughby_rahn_r(Pmax_mm_h, Rmax_km, r_km)

# --------------------------
# SCS Infiltration (CN)
# --------------------------
class SCSInfiltration:
    def __init__(self, CN=70.0):
        self.CN = float(CN)
        S_in_inch = 1000.0 / self.CN - 10.0
        self.S = S_in_inch * 0.0254
        self.Ia = 0.2 * self.S

    def cumulative_runoff_from_P(self, P_cum_m):
        P = P_cum_m
        Q = np.zeros_like(P)
        mask = P > self.Ia
        if np.any(mask):
            num = (P[mask] - self.Ia)**2
            den = P[mask] - self.Ia + self.S
            Q[mask] = num / den
        return Q

    def instantaneous_infiltration_rate(self, P_instant_m_s, P_cum_m, dPdt=0.0):
        P = P_cum_m
        P_inst = P_instant_m_s
        I_inst = np.zeros_like(P)
        mask0 = (P <= self.Ia)
        I_inst[mask0] = P_inst[mask0]
        mask1 = (P > self.Ia)
        if np.any(mask1):
            Pm = P[mask1]
            dP_dt = dPdt if np.isscalar(dPdt) else dPdt[mask1]
            num = (2.0*(Pm - self.Ia)*(Pm - self.Ia + self.S) - (Pm - self.Ia)**2)
            den = (Pm - self.Ia + self.S)**2
            dQdP = num / den
            dQdt = dQdP * dP_dt
            I_inst[mask1] = dP_dt - dQdt
            I_inst[mask1] = np.maximum(I_inst[mask1], 0.0)
        return I_inst

# --------------------------
# Grid / Flood state
# --------------------------
class FloodGrid:
    def __init__(self, Nx, Ny, Lx, Ly, CN=70.0, manning_n=0.025):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.xc = (np.arange(self.Nx) + 0.5) * self.dx
        self.yc = (np.arange(self.Ny) + 0.5) * self.dy
        X, Y = np.meshgrid(self.xc, self.yc)
        self.X = X
        self.Y = Y
        self.h = np.zeros((self.Ny, self.Nx), dtype=float)
        self.u = np.zeros((self.Ny, self.Nx), dtype=float)
        self.v = np.zeros((self.Ny, self.Nx), dtype=float)
        self.zb = np.zeros((self.Ny, self.Nx), dtype=float)
        self.n = manning_n * np.ones((self.Ny, self.Nx), dtype=float)
        self.P_cum = np.zeros((self.Ny, self.Nx), dtype=float)
        self.Q_cum = np.zeros((self.Ny, self.Nx), dtype=float)
        self.CN = CN * np.ones((self.Ny, self.Nx), dtype=float)
        self.scs = SCSInfiltration(CN=np.mean(self.CN))
        self.mask = np.ones((self.Ny, self.Nx), dtype=float)

    def set_bed_gaussian(self, peak=1966.0, x0=None, y0=None, sigma=3000.0):
        if x0 is None:
            x0 = self.Lx / 2.0
        if y0 is None:
            y0 = self.Ly / 2.0
        dx = self.X - x0
        dy = self.Y - y0
        self.zb = peak * np.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))

# --------------------------
# HLLC Riemann solver
# --------------------------
def hllc_flux_1d(hL, uL, vL, hR, uR, vR, g=g):
    tiny = 1e-12
    hL = max(hL, 0.0)
    hR = max(hR, 0.0)
    huL = hL * uL
    hvL = hL * vL
    huR = hR * uR
    hvR = hR * vR
    cL = np.sqrt(g * hL) if hL > tiny else 0.0
    cR = np.sqrt(g * hR) if hR > tiny else 0.0
    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)
    FhL = huL
    FhuL = hL * uL * uL + 0.5 * g * hL * hL
    FhvL = hL * uL * vL
    FhR = huR
    FhuR = hR * uR * uR + 0.5 * g * hR * hR
    FhvR = hR * uR * vR
    if SR - SL < 1e-14:
        return 0.5*(FhL + FhR), 0.5*(FhuL + FhuR), 0.5*(FhvL + FhvR)
    S_star = (SR * uR - SL * uL + 0.5 * g * (hL*hL - hR*hR)) / (SR - SL)
    hstarL = hL * (SL - uL) / (SL - S_star) if abs(SL - S_star) > 1e-15 else hL
    hstarR = hR * (SR - uR) / (SR - S_star) if abs(SR - S_star) > 1e-15 else hR
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
# Hydrostatic reconstruction
# --------------------------
def hydrostatic_reconstruction(hL, zbL, hR, zbR):
    etaL = hL + zbL
    etaR = hR + zbR
    zbmax = np.maximum(zbL, zbR)
    hL_star = np.maximum(0.0, etaL - zbmax)
    hR_star = np.maximum(0.0, etaR - zbmax)
    return hL_star, hR_star, zbmax

# --------------------------
# Spatial operator (FV)
# --------------------------
def spatial_operator(grid: FloodGrid):
    Ny, Nx = grid.Ny, grid.Nx
    dx, dy = grid.dx, grid.dy
    rhs_h = np.zeros_like(grid.h)
    rhs_hu = np.zeros_like(grid.h)
    rhs_hv = np.zeros_like(grid.h)
    Fx_h = np.zeros((Ny, Nx+1))
    Fx_hu = np.zeros((Ny, Nx+1))
    Fx_hv = np.zeros((Ny, Nx+1))
    for j in range(Ny):
        for i_face in range(Nx+1):
            iL = i_face - 1
            iR = i_face
            if iL < 0:
                hL = grid.h[j, 0]
                uL = -grid.u[j, 0]
                vL = grid.v[j, 0]
                zbL = grid.zb[j, 0]
            else:
                hL = grid.h[j, iL]; uL = grid.u[j, iL]; vL = grid.v[j, iL]; zbL = grid.zb[j, iL]
            if iR >= Nx:
                hR = grid.h[j, -1]; uR = -grid.u[j, -1]; vR = grid.v[j, -1]; zbR = grid.zb[j, -1]
            else:
                hR = grid.h[j, iR]; uR = grid.u[j, iR]; vR = grid.v[j, iR]; zbR = grid.zb[j, iR]
            hLr, hRr, zbmax = hydrostatic_reconstruction(hL, zbL, hR, zbR)
            uLr = uL if hLr > 1e-12 else 0.0
            vLr = vL if hLr > 1e-12 else 0.0
            uRr = uR if hRr > 1e-12 else 0.0
            vRr = vR if hRr > 1e-12 else 0.0
            Fh, Fhu, Fhv = hllc_flux_1d(hLr, uLr, vLr, hRr, uRr, vRr, g=g)
            Fx_h[j, i_face] = Fh
            Fx_hu[j, i_face] = Fhu
            Fx_hv[j, i_face] = Fhv
    # y-direction (similar rotation)
    Gy_h = np.zeros((Ny+1, Nx))
    Gy_hu = np.zeros((Ny+1, Nx))
    Gy_hv = np.zeros((Ny+1, Nx))
    for i in range(Nx):
        for j_face in range(Ny+1):
            jL = j_face - 1
            jR = j_face
            if jL < 0:
                hL = grid.h[0, i]; uL = grid.u[0, i]; vL = -grid.v[0, i]; zbL = grid.zb[0, i]
            else:
                hL = grid.h[jL, i]; uL = grid.u[jL, i]; vL = grid.v[jL, i]; zbL = grid.zb[jL, i]
            if jR >= Ny:
                hR = grid.h[-1, i]; uR = grid.u[-1, i]; vR = -grid.v[-1, i]; zbR = grid.zb[-1, i]
            else:
                hR = grid.h[jR, i]; uR = grid.u[jR, i]; vR = grid.v[jR, i]; zbR = grid.zb[jR, i]
            hLr, hRr, zbmax = hydrostatic_reconstruction(hL, zbL, hR, zbR)
            uLr = vL if hLr > 1e-12 else 0.0
            vLr = uL if hLr > 1e-12 else 0.0
            uRr = vR if hRr > 1e-12 else 0.0
            vRr = uR if hRr > 1e-12 else 0.0
            Fh, Fhu, Fhv = hllc_flux_1d(hLr, uLr, vLr, hRr, uRr, vRr, g=g)
            Gy_h[j_face, i] = Fh
            Gy_hv[j_face, i] = Fhu
            Gy_hu[j_face, i] = Fhv
    rhs_h += - (Fx_h[:, 1:] - Fx_h[:, :-1]) / dx
    rhs_h += - (Gy_h[1:, :] - Gy_h[:-1, :]) / dy
    rhs_hu += - (Fx_hu[:, 1:] - Fx_hu[:, :-1]) / dx
    rhs_hu += - (Gy_hu[1:, :] - Gy_hu[:-1, :]) / dy
    rhs_hv += - (Fx_hv[:, 1:] - Fx_hv[:, :-1]) / dx
    rhs_hv += - (Gy_hv[1:, :] - Gy_hv[:-1, :]) / dy
    dzbdx = np.zeros_like(grid.zb)
    dzbdy = np.zeros_like(grid.zb)
    dzbdx[:, 1:-1] = (grid.zb[:, 2:] - grid.zb[:, :-2]) / (2.0 * dx)
    dzbdx[:, 0] = (grid.zb[:, 1] - grid.zb[:, 0]) / dx
    dzbdx[:, -1] = (grid.zb[:, -1] - grid.zb[:, -2]) / dx
    dzbdy[1:-1, :] = (grid.zb[2:, :] - grid.zb[:-2, :]) / (2.0 * dy)
    dzbdy[0, :] = (grid.zb[1, :] - grid.zb[0, :]) / dy
    dzbdy[-1, :] = (grid.zb[-1, :] - grid.zb[-2, :]) / dy
    rhs_hu += - g * grid.h * dzbdx
    rhs_hv += - g * grid.h * dzbdy
    return rhs_h, rhs_hu, rhs_hv

# --------------------------
# Friction
# --------------------------
def apply_manning_friction(grid: FloodGrid, dt):
    h = grid.h
    n = grid.n
    tiny = 1e-12
    hu = h * grid.u
    hv = h * grid.v
    speed = np.sqrt(grid.u**2 + grid.v**2)
    denom = 1.0 + dt * g * n**2 * speed / (np.maximum(h, tiny)**(4.0/3.0))
    hu_new = hu / denom
    hv_new = hv / denom
    mask = h > 1e-12
    grid.u[:] = 0.0
    grid.v[:] = 0.0
    grid.u[mask] = hu_new[mask] / h[mask]
    grid.v[mask] = hv_new[mask] / h[mask]

# --------------------------
# Precipitation + infiltration
# --------------------------
def apply_precip_infiltration(grid: FloodGrid, P_field_m_s, dt):
    scs = grid.scs
    dP = P_field_m_s * dt
    grid.P_cum += dP
    Q_cum_new = scs.cumulative_runoff_from_P(grid.P_cum)
    dQ = Q_cum_new - grid.Q_cum
    grid.Q_cum = Q_cum_new
    dF = dP - dQ
    grid.h += dP
    grid.h -= dF
    grid.h[:] = np.maximum(0.0, grid.h)

# --------------------------
# CFL timestep
# --------------------------
def compute_timestep(grid: FloodGrid, alpha_CFL=0.3, dt_max=1.0, dt_min=1e-6):
    c = np.sqrt(g * np.maximum(grid.h, 0.0))
    lam_x = np.abs(grid.u) + c
    lam_y = np.abs(grid.v) + c
    lam_max = max(np.max(lam_x) if lam_x.size>0 else 0.0, np.max(lam_y) if lam_y.size>0 else 0.0)
    if lam_max <= 0.0:
        return dt_max
    dt1 = alpha_CFL * min(grid.dx, grid.dy) / lam_max
    dt = float(np.clip(dt1, dt_min, dt_max))
    return dt

# --------------------------
# State helpers
# --------------------------
def pack_state(grid: FloodGrid):
    return np.stack([grid.h, grid.h * grid.u, grid.h * grid.v], axis=0)

def unpack_state(grid: FloodGrid, U):
    grid.h[:, :] = U[0]
    hu = U[1]
    hv = U[2]
    mask = grid.h > 1e-12
    grid.u[:, :] = 0.0
    grid.v[:, :] = 0.0
    grid.u[mask] = hu[mask] / grid.h[mask]
    grid.v[mask] = hv[mask] / grid.h[mask]

def spatial_operator_pack(grid: FloodGrid):
    rhs_h, rhs_hu, rhs_hv = spatial_operator(grid)
    return np.stack([rhs_h, rhs_hu, rhs_hv], axis=0)

# --------------------------
# Main simulation routine
# --------------------------
def simulate_flood(grid: FloodGrid, lorenz: Lorenz63, t_end=60.0, output_interval=10.0, record=True,
                   Pmax_mm_h=100.0, Rmax_km=30.0, alphaP=0.15, sigmaX=7.89):
    t = 0.0
    step = 0
    next_output = output_interval
    eye_x_m = grid.Lx / 2.0
    eye_y_m = grid.Ly / 2.0
    times = []
    mean_rain = []
    max_h_list = []
    total_volume = []
    lorenz_x = []
    lorenz_y = []
    lorenz_z = []
    dt_chaos = 0.01

    while t < t_end - 1e-12:
        dt = compute_timestep(grid, alpha_CFL=0.3, dt_max=1.0, dt_min=1e-6)
        n_chaos = max(1, int(np.ceil(dt / dt_chaos)))
        dtc = dt / float(n_chaos)
        for _ in range(n_chaos):
            lorenz.step_rk4(dtc)
        lam = lorenz.modulation_factor(alpha_P=alphaP, sigma_X=sigmaX)
        Pbase = compute_typhoon_precip_field(Pmax_mm_h, Rmax_km, grid.X, grid.Y, eye_x_m, eye_y_m)
        Ptotal = Pbase * lam
        if record:
            times.append(t)
            mean_rain.append(np.mean(Ptotal) * 3600.0 * 1000.0)
            max_h_list.append(np.max(grid.h))
            total_volume.append(np.sum(grid.h) * grid.dx * grid.dy)
            lorenz_x.append(lorenz.X)
            lorenz_y.append(lorenz.Y)
            lorenz_z.append(lorenz.Z)
        U0 = pack_state(grid)
        rhs0 = spatial_operator_pack(grid)
        U1 = U0 + dt * rhs0
        unpack_state(grid, U1)
        apply_precip_infiltration(grid, Ptotal, dt)
        rhs1 = spatial_operator_pack(grid)
        U2 = 0.75 * U0 + 0.25 * (U1 + dt * rhs1)
        unpack_state(grid, U2)
        apply_precip_infiltration(grid, Ptotal, dt)
        rhs2 = spatial_operator_pack(grid)
        Unew = (1.0/3.0) * U0 + (2.0/3.0) * (U2 + dt * rhs2)
        unpack_state(grid, Unew)
        apply_precip_infiltration(grid, Ptotal, dt)
        apply_manning_friction(grid, dt)
        t += dt
        step += 1
        vtrans = 5.556
        eye_x_m -= vtrans * dt
        if t >= next_output - 1e-9 or t >= t_end - 1e-9:
            print(f"[t={t:.2f}s] step {step}, max h = {grid.h.max():.4f} m")
            next_output += output_interval

    if record:
        times.append(t)
        mean_rain.append(np.mean(Ptotal) * 3600.0 * 1000.0)
        max_h_list.append(np.max(grid.h))
        total_volume.append(np.sum(grid.h) * grid.dx * grid.dy)
        lorenz_x.append(lorenz.X)
        lorenz_y.append(lorenz.Y)
        lorenz_z.append(lorenz.Z)
        times = np.array(times)
        mean_rain = np.array(mean_rain)
        max_h_list = np.array(max_h_list)
        total_volume = np.array(total_volume)
        lorenz_x = np.array(lorenz_x)
        lorenz_y = np.array(lorenz_y)
        lorenz_z = np.array(lorenz_z)
        dV = np.diff(total_volume, prepend=total_volume[0])
        avg_dt = times[1] - times[0] if len(times) > 1 else 1.0
        discharge_proxy = dV / avg_dt
        ts = {
            "time": times,
            "mean_rain_mmhr": mean_rain,
            "max_h": max_h_list,
            "total_volume_m3": total_volume,
            "discharge_m3s": discharge_proxy,
            "lorenz_x": lorenz_x,
            "lorenz_y": lorenz_y,
            "lorenz_z": lorenz_z,
        }
        return grid, ts

    return grid

# --------------------------
# GUI wrapper
# --------------------------
def run_flood_simulation(params):
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
    grid = FloodGrid(Nx, Ny, Lx, Ly, CN=CN, manning_n=manning_n)
    grid.set_bed_gaussian(peak=300.0, x0=Lx/2.0, y0=Ly/2.0, sigma=0.25*Lx)
    if seed_pond:
        cx = Nx // 2
        cy = Ny // 2
        x0 = max(cx-1, 0); x1 = min(cx+2, Nx)
        y0 = max(cy-1, 0); y1 = min(cy+2, Ny)
        grid.h[y0:y1, x0:x1] = 0.5
    lorenz = Lorenz63()
    grid_res, ts = simulate_flood(grid, lorenz, t_end=t_end, output_interval=output_interval, record=True,
                                  Pmax_mm_h=Pmax_mm_h, Rmax_km=Rmax_km, alphaP=alphaP, sigmaX=sigmaX)
    discharge = ts["discharge_m3s"]
    n_ens = int(params.get("n_ens", 20))
    spread = max(1e-6, 0.05 * np.nanmax(np.abs(discharge)) if discharge.size>0 else 1e-3)
    ensembles = np.array([discharge + np.random.normal(0.0, spread, size=discharge.shape) for _ in range(n_ens)])
    results = {
        "time": ts["time"],
        "rain": ts["mean_rain_mmhr"],
        "discharge": discharge,
        "ensembles": ensembles,
        "flood_depth": grid_res.h,
        "lorenz_x": ts["lorenz_x"],
        "lorenz_y": ts["lorenz_y"],
        "lorenz_z": ts["lorenz_z"],
        "max_h_ts": ts["max_h"],
        "total_volume": ts["total_volume_m3"]
    }
    return results

# --------------------------
# Quick test
# --------------------------
if __name__ == "__main__":
    params = {"Nx":40, "Ny":20, "Lx":20000.0, "Ly":10000.0, "t_end":30.0, "output_interval":10.0}
    res = run_flood_simulation(params)
    print("Done. time samples:", res["time"].shape)
    print("Max final depth:", np.max(res["flood_depth"]))
