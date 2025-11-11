# 🌊 Chaos-Enhanced Flood Prediction System

A **Tkinter-based desktop application** for simulating and visualizing flood dynamics enhanced by **chaotic Lorenz attractor modeling**.
It combines 2D shallow-water hydrodynamics with rainfall-runoff processes and ensemble uncertainty visualization — wrapped in a modern blue-white UI.

---

## 🚀 Features

* **Interactive UI** built with `Tkinter` and `matplotlib`
* **2D Flood simulation** using finite-volume shallow-water equations
* **Chaotic modulation** via Lorenz-63 system for rainfall variability
* **SCS Curve Number** infiltration and runoff model
* **Real-time plots** for:

  * Flood depth map
  * 3D Lorenz attractor (with start & end markers)
  * Rainfall vs. discharge
  * Ensemble spread (±1σ)
  * Diagnostics & time-series statistics
* **Data export options:**

  * Save any panel as PNG
  * Export all plots to a multi-page PDF

---

## 🧩 Project Structure

```
├── flood_ui.py              # Main UI application
├── chaos_flood_model.py     # Numerical flood & chaos model
└── README.md                # Documentation (this file)
```

---

## ⚙️ Requirements

Python ≥ 3.8
Required libraries:

```bash
pip install numpy matplotlib
```

*(Tkinter comes with most Python distributions)*

Optional (for PDF export):

```bash
pip install reportlab
```

---

## ▶️ How to Run

1. Clone or download the repository.
2. Make sure both files are in the same directory:

   ```
   flood_ui.py
   chaos_flood_model.py
   ```
3. Run the UI:

   ```bash
   python flood_ui.py
   ```
4. Adjust simulation parameters in the interface, then click **“Run Simulation”**.

---

## 🧮 Simulation Parameters

| Parameter         | Description                   | Default         |
| ----------------- | ----------------------------- | --------------- |
| Nx, Ny            | Grid resolution (x & y cells) | 40 × 20         |
| Lx, Ly            | Domain size (meters)          | 20,000 × 10,000 |
| Curve Number (CN) | SCS infiltration coefficient  | 70              |
| Manning n         | Surface roughness             | 0.03            |
| Simulation Time   | Duration (s)                  | 120             |
| Output Interval   | Output frequency (s)          | 10              |
| Ensemble Members  | Number of ensemble runs       | 3               |

---

## 📊 Output Tabs

1. **Flood Map** – 2D color map of final water depth
2. **3D Lorenz Attractor** – chaotic trajectory visualization
3. **Rainfall vs. Runoff** – rainfall and discharge over time
4. **Ensemble Spread** – mean ± standard deviation of discharge
5. **Diagnostics** – peak depth and total volume evolution
6. **Summary** – key performance metrics

---

## 💾 Export Options

* **💾 Save Panel:** Saves the current tab as `.png`
* **🧾 Export All (PDF):** Exports all panels into a single multi-page PDF

---

## 🧠 Technical Overview

* **Flood physics:**

  * Shallow-water equations solved via **HLLC Riemann solver**
  * **Hydrostatic reconstruction** ensures well-balanced states
  * **Manning’s friction** applied semi-implicitly
  * **Precipitation/Infiltration** modeled using SCS Curve Number

* **Chaotic forcing:**

  * **Lorenz-63 system** drives rainfall modulation
  * Represents dynamic atmospheric variability affecting precipitation

---

## 🧑‍💻 Authors

Developed by **PapiiCholoo(2025)**
Lead Developer: *Theomel De Guzman*

---

## 📜 License

This project is released for academic and research use only.
© 2025 PapiiCholoo — All rights reserved.
