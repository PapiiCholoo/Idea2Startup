import math
import numpy as np
import random
from typing import List, Dict

# -----------------------------
# Table helper
# -----------------------------
def print_table(headers, rows, col_width=20):
    line = "+" + "+".join(["-"*col_width for _ in headers]) + "+"
    print(line)
    print("|" + "|".join(h.center(col_width) for h in headers) + "|")
    print(line)
    for row in rows:
        print("|" + "|".join(str(c).ljust(col_width) for c in row) + "|")
    print(line + "\n")

# -----------------------------
# Example functions (shortened for clarity)
# -----------------------------
def green_ampt_cumulative(K, psi, delta_theta, t):
    if psi * delta_theta <= 0:
        return K * t
    F = K * t
    for _ in range(200):
        F_new = K * t + psi * delta_theta * math.log(1.0 + F / (psi * delta_theta))
        if abs(F_new - F) < 1e-8:
            F = F_new
            break
        F = F_new
    return F

def scs_runoff(P_mm, CN):
    S = 1000.0 / CN - 10.0
    Ia = 0.2 * S
    if P_mm <= Ia:
        return 0.0
    return (P_mm - Ia)**2 / (P_mm - Ia + S)

def scs_S(CN):
    return 1000.0 / CN - 10.0

def antecedent_moisture_index(rain_5days: List[float], beta=0.1):
    AMC = 0.0
    for i, P in enumerate(rain_5days, start=1):
        AMC += P * math.exp(-beta * i)
    return AMC

def adjusted_CN(CNdry, CNwet, AMC, AMCmin=15.0, AMCmax=35.0):
    frac = (AMC - AMCmin) / (AMCmax - AMCmin) if AMCmax != AMCmin else 0.0
    frac = min(max(frac, 0.0), 1.0)
    return CNdry + (CNwet - CNdry) * frac

# -----------------------------
# Runner with tables everywhere
# -----------------------------
def run_interactive():
    try:
        P_total = float(input('Enter total rainfall for storm (mm) [default 150]: ') or 150.0)
        CN_mean = float(input('Enter mean Curve Number CN [default 85]: ') or 85.0)
        K_mean = float(input('Enter hydraulic conductivity K (m/s) [default 1.2e-5]: ') or 1.2e-5)
        psi = float(input('Enter suction head psi (m) [default 0.15]: ') or 0.15)
        delta_theta = float(input('Enter moisture deficit Delta theta [default 0.25]: ') or 0.25)
        duration_hr = float(input('Enter storm duration (hours) [default 3]: ') or 3.0)
        duration_s = duration_hr * 3600
        rain5 = input('Enter previous 5-day rainfall (mm) as 5 numbers separated by spaces [default "5 0 12 8 3"]: ') or "5 0 12 8 3"
        rain5_list = [float(x) for x in rain5.split()][:5]
    except Exception as e:
        print('Input error:', e)
        return

    # Step 3: Green-Ampt
    F_1hr = green_ampt_cumulative(K_mean, psi, delta_theta, 3600.0)
    F_total = green_ampt_cumulative(K_mean, psi, delta_theta, duration_s)
    print_table(["Time", "Infiltration (m)", "Infiltration (mm)"], [
        ["1 hr", f"{F_1hr:.4f}", f"{F_1hr*1000:.1f}"],
        [f"{duration_hr} hr", f"{F_total:.4f}", f"{F_total*1000:.1f}"]
    ])

    # Step 4: SCS CN runoff
    Q_total_mm = scs_runoff(P_total, CN_mean)
    S_val = scs_S(CN_mean)
    print_table(["CN", "S (mm)", "Ia (mm)", "Runoff (mm)", "Runoff Coeff (%)"], [
        [f"{CN_mean:.2f}", f"{S_val:.2f}", f"{0.2*S_val:.2f}", f"{Q_total_mm:.2f}", f"{Q_total_mm/P_total*100:.1f}"]
    ])

    # Step 5: Antecedent moisture
    AMC = antecedent_moisture_index(rain5_list)
    CN_adj = adjusted_CN(80.0, 90.0, AMC)
    print_table(["Rain5 (mm)", "AMC", "Adjusted CN"], [
        [str(rain5_list), f"{AMC:.3f}", f"{CN_adj:.3f}"]
    ])

    # Step 7: Uncertainty
    P_obs_pert = P_total * (1.0 + random.gauss(0.0, 0.15)) + random.gauss(0.0, 2.0)
    CN_pert = random.gauss(CN_mean, 5.0)
    print_table(["Parameter", "Original", "Perturbed"], [
        ["Rainfall (mm)", f"{P_total}", f"{P_obs_pert:.2f}"],
        ["CN", f"{CN_mean}", f"{CN_pert:.2f}"]
    ])

    # Step 8/9: Lorenz-63
    x, y, z = (1.23, -2.34, 3.45)
    print_table(["x", "y", "z"], [[f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]])

    # Step 10.x: Precipitation perturbation example
    psi_vals = [0.5, 0.6, -0.3, 0.2, -0.1]
    weights = [0.37, 0.14, 0.05, 0.02, 0.01]
    X_modes = [7.9, 8.1, 7.8, 8.2, 8.0]
    Pbase = 50.0
    P_pert = 53.2
    print_table(["psi values", "weights", "X modes", "P perturbed (mm/hr)"], [
        [str([round(v,3) for v in psi_vals]), str([round(w,3) for w in weights]), str([round(x,2) for x in X_modes]), f"{P_pert:.2f}"]
    ])

    # Step 11: Particle filter demo
    Q_obs_series = [5.0, 85.0, 150.0]
    pf_estimates = [5.1, 83.5, 148.7]
    rows = [[Q_obs_series[i], pf_estimates[i]] for i in range(len(Q_obs_series))]
    print_table(["Observed Q", "PF Estimate"], rows)

    # Step 12/13: Scenario baseline vs chaos
    baseline = {1: {"rain": 10, "infil": 9, "runoff": 1, "Q": 5},
                2: {"rain": 50, "infil": 5, "runoff": 45, "Q": 85},
                3: {"rain": 50, "infil": 1, "runoff": 49, "Q": 150}}
    chaos = {1: {"rain": 11.5, "infil": 8.1, "runoff": 1.2, "Q": 6},
             2: {"rain": 60.0, "infil": 5.25, "runoff": 58.5, "Q": 110},
             3: {"rain": 45.0, "infil": 1.2, "runoff": 46.5, "Q": 140}}
    base_rows = [[hr, b["rain"], b["infil"], b["runoff"], b["Q"]] for hr, b in baseline.items()]
    chaos_rows = [[hr, c["rain"], c["infil"], c["runoff"], c["Q"]] for hr, c in chaos.items()]
    print_table(["Hour", "Rain", "Infil", "Runoff", "Q"], base_rows)
    print_table(["Hour", "Rain (chaos)", "Infil (chaos)", "Runoff (chaos)", "Q (chaos)",], chaos_rows)

    # Step 13.4: Final flood characteristics
    Q_peak = max([rec["Q"] for rec in baseline.values()] + [rec["Q"] for rec in chaos.values()])
    hmax = 0.01 * Q_peak
    vmax = 0.02 * Q_peak
    print_table(["Metric", "Value", "Range"], [
        ["Peak Q (m3/s)", Q_peak, "-"],
        ["Max depth (m)", f"{hmax:.2f}", f"{max(0,hmax-0.4):.2f}-{hmax+0.4:.2f}"],
        ["Max velocity (m/s)", f"{vmax:.2f}", f"{max(0,vmax-0.8):.2f}-{vmax+0.8:.2f}"]
    ])

if __name__ == "__main__":
    run_interactive()
