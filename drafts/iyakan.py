import math
import numpy as np
import random

# ===============================
# Step 3: Green-Ampt Infiltration
# ===============================
def green_ampt_infiltration(K, psi, delta_theta, t):
    F = K * t  # initial guess
    for _ in range(50):  # iterative solution
        F_new = K * t + psi * delta_theta * math.log(1 + F / (psi * delta_theta))
        if abs(F_new - F) < 1e-8:
            break
        F = F_new
    return F

# ===============================
# Step 4: SCS Curve Number Runoff
# ===============================
def scs_runoff(P, CN):
    S = 1000.0 / CN - 10.0
    Ia = 0.2 * S
    if P > Ia:
        Q = (P - Ia) ** 2 / (P - Ia + S)
    else:
        Q = 0.0
    return Q

# ===============================
# Step 5: Antecedent Moisture
# ===============================
def antecedent_moisture(rain_5days, CNdry=80, CNwet=90, AMCmin=15, AMCmax=35, beta=0.1):
    AMC = 0
    for i, P in enumerate(rain_5days, 1):
        AMC += P * math.exp(-beta * i)
    CNadjusted = CNdry + (CNwet - CNdry) * (AMC - AMCmin) / (AMCmax - AMCmin)
    return AMC, CNadjusted

# ===============================
# Step 9: Lorenz-63 Chaos System
# ===============================
def lorenz63(x, y, z, dt=0.01, steps=100, sigma=10, rho=28, beta=8/3):
    for _ in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
    return x, y, z

# ===============================
# Step 10: Lorenz-96 Chaos System
# ===============================
def lorenz96(X, F=8, dt=0.01, steps=100):
    N = len(X)
    X = np.array(X)
    for _ in range(steps):
        dX = np.zeros(N)
        for k in range(N):
            dX[k] = (X[(k+1)%N] - X[(k-2)%N]) * X[(k-1)%N] - X[k] + F
        X += dX * dt
    return X

# ===============================
# Step 11: Simple Particle Filter
# ===============================
def particle_filter(obs_Q, Np=1000, sigma=5):
    particles = [random.gauss(obs_Q, sigma) for _ in range(Np)]
    weights = [math.exp(-0.5 * ((p - obs_Q) / sigma) ** 2) for p in particles]
    weights = np.array(weights) / sum(weights)
    estimate = np.dot(particles, weights)
    return estimate, np.std(particles)

# ===============================
# Step 13: Scenario Evolution
# ===============================
def scenario_evolution():
    baseline = {
        1: {"rain": 10, "infil": 9, "runoff": 1, "Q": 5},
        2: {"rain": 50, "infil": 5, "runoff": 45, "Q": 85},
        3: {"rain": 50, "infil": 1, "runoff": 49, "Q": 150},
    }

    chaos = {
        1: {"rain": 10*1.15, "infil": 9*0.9, "runoff": 1*1.2, "Q": 6},
        2: {"rain": 50*1.2, "infil": 5*1.05, "runoff": 45*1.3, "Q": 110},
        3: {"rain": 50*0.9, "infil": 1*1.2, "runoff": 49*0.95, "Q": 140},
    }
    
    return baseline, chaos

# ===============================
# MAIN PROGRAM
# ===============================
def main():
    print("\n===========================================")
    print(" CHAOS-ENHANCED FLOOD PREDICTION REPORT")
    print("===========================================\n")

    # User input
    rainfall = float(input("Enter total rainfall (mm): "))
    CN = float(input("Enter Curve Number (CN): "))
    K = float(input("Enter Hydraulic Conductivity K (m/s): "))
    psi = float(input("Enter Suction Head ψ (m): "))
    delta_theta = float(input("Enter Moisture Deficit Δθ: "))
    t = float(input("Enter duration (s): "))
    rain_5days = [float(x) for x in input("Enter last 5 days rainfall (mm, space-separated): ").split()]

    print("\n--- INPUT SCENARIO ---")
    print(f"Rainfall: {rainfall:.2f} mm")
    print(f"Curve Number: {CN}")
    print(f"Soil Parameters: K={K:.2e} m/s, ψ={psi} m, Δθ={delta_theta}")
    print(f"Duration: {t/3600:.2f} hours")
    print(f"Previous 5-day Rainfall: {rain_5days}\n")

    # Step 3
    F = green_ampt_infiltration(K, psi, delta_theta, t) * 1000
    print("Step 3: Infiltration (Green-Ampt Model)")
    print("---------------------------------------")
    print(f"Cumulative Infiltration after {t/3600:.1f} hrs: {F:.2f} mm\n")

    # Step 4
    Q = scs_runoff(rainfall, CN)
    print("Step 4: Runoff Generation (SCS CN Method)")
    print("------------------------------------------")
    print(f"Direct Runoff: {Q:.2f} mm ({Q/rainfall*100:.1f}% of rainfall)\n")

    # Step 5
    AMC, CNadj = antecedent_moisture(rain_5days)
    print("Step 5: Antecedent Moisture Conditions")
    print("---------------------------------------")
    print(f"Antecedent Moisture Content (AMC): {AMC:.2f} mm")
    print(f"Adjusted Curve Number: {CNadj:.2f}\n")

    # Step 9
    x, y, z = lorenz63(1.0, 1.0, 1.0)
    print("Step 9: Chaos Dynamics (Lorenz-63 System)")
    print("------------------------------------------")
    print(f"Chaotic State after 100 steps: x={x:.3f}, y={y:.3f}, z={z:.3f}\n")

    # Step 10
    X = [8.0]*40
    X[19] = 8.01
    X = lorenz96(X)
    print("Step 10: Spatial Chaos (Lorenz-96 System)")
    print("------------------------------------------")
    print(f"Sample Variable X5 after evolution: {X[4]:.3f}\n")

    # Step 11
    est, spread = particle_filter(obs_Q=50.0)
    print("Step 11: Sequential Monte Carlo (Particle Filter)")
    print("-------------------------------------------------")
    print(f"Estimated Streamflow: {est:.2f} m³/s")
    print(f"Ensemble Spread (Uncertainty): {spread:.2f} m³/s\n")

    # Step 13
    baseline, chaos = scenario_evolution()
    print("Step 13: Scenario Evolution (3-hour Storm)")
    print("------------------------------------------")
    print("Hour | Rainfall (mm/hr) | Infiltration | Runoff | Discharge Q (m³/s)")
    print("-------------------------------------------------------------------")
    for hr in range(1, 4):
        b = baseline[hr]
        print(f" {hr:<3} | {b['rain']:<15.1f} | {b['infil']:<11.1f} | {b['runoff']:<6.1f} | {b['Q']:<6.1f}")

    print("\nWith Chaos Perturbations:")
    print("Hour | Rainfall (mm/hr) | Infiltration | Runoff | Discharge Q (m³/s)")
    print("-------------------------------------------------------------------")
    for hr in range(1, 4):
        c = chaos[hr]
        print(f" {hr:<3} | {c['rain']:<15.1f} | {c['infil']:<11.1f} | {c['runoff']:<6.1f} | {c['Q']:<6.1f}")
    print()

    print("===========================================")
    print(" END OF REPORT")
    print("===========================================\n")

if __name__ == "__main__":
    main()
