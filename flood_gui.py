#Author: Ako


import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import requests
from io import BytesIO
from pyproj import Transformer

# === Import your flood model with diagnostics ===
from agay_fixed3 import (
    HighAccuracyNagaCityConfig,
    UltraAccurateFloodPredictor,
    create_particle_filter_diagnostics,
)

# 🔑 Mapbox setup
MAPBOX_TOKEN = "pk.eyJ1IjoiY2hvLWQtbXlzdGVyaW91cy0wNDIwIiwiYSI6ImNtZnBzMXFlazBlbm0ybHNjdWc1bXg1bGcifQ.sEaQ5Q2qu3Tm7WlYTTXsog"

# ✅ Barangays of Naga City
BARANGAYS = [
    "Abella", "Bagumbayan Norte", "Bagumbayan Sur", "Balatas", "Calauag",
    "Cararayan", "Carolina", "Concepcion Grande", "Concepcion Pequeña",
    "Dayangdang", "Dinaga", "Igualdad Interior", "Lerma", "Liboton", "Mabolo",
    "Pacol", "Panicuason", "Peñafrancia", "Sabang", "San Felipe", "San Francisco",
    "San Isidro", "Santa Cruz", "Tabuco", "Tinago", "Triangulo", "Villa Grande"
]

# ✅ Placeholder barangay bounding boxes
BARANGAY_BOUNDS = {
    "Abella": (123.17, 123.18, 13.62, 13.63),
    "Bagumbayan Norte": (123.17, 123.19, 13.62, 13.635),
    "Triangulo": (123.19, 123.20, 13.62, 13.63),
    # ... si iba pa kulang, pa othe barangay
}


class FloodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Naga City Flood — Static Map")
        self.results = None
        self.dt_hours = 0.25
        self.show_grid = tk.BooleanVar(value=False)
        self.mapbox_style = tk.StringVar(value="satellite-v9")
        self.selected_bounds = None
        self.gauge_xy = None

        # Full Naga City extent
        self.extent_ll = [
            HighAccuracyNagaCityConfig.lon_min,
            HighAccuracyNagaCityConfig.lon_max,
            HighAccuracyNagaCityConfig.lat_min,
            HighAccuracyNagaCityConfig.lat_max
        ]
        self.to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # --- Control panel ---
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Grid Resolution (m):").pack()
        self.dx_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.dx_var).pack()

        ttk.Label(control_frame, text="Ensemble Size:").pack()
        self.n_particles_var = tk.IntVar(value=10)
        ttk.Entry(control_frame, textvariable=self.n_particles_var).pack()

        ttk.Label(control_frame, text="Forecast Hours:").pack()
        self.hours_var = tk.IntVar(value=24)
        ttk.Entry(control_frame, textvariable=self.hours_var).pack()
        
        ttk.Label(control_frame, text="Rainfall (mm/hr):").pack()
        self.rainfall_var = tk.DoubleVar(value=10.0)
        ttk.Entry(control_frame, textvariable=self.rainfall_var).pack()

        ttk.Label(control_frame, text="Water Level (m):").pack()
        self.water_level_var = tk.DoubleVar(value=0.5)
        ttk.Entry(control_frame, textvariable=self.water_level_var).pack()

        ttk.Label(control_frame, text="Flow Velocity (m/s):").pack()
        self.flow_velocity_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.flow_velocity_var).pack()

        ttk.Label(control_frame, text="Wind Speed (m/s):").pack()
        self.wind_speed_var = tk.DoubleVar(value=5.0)
        ttk.Entry(control_frame, textvariable=self.wind_speed_var).pack()

        ttk.Label(control_frame, text="Wind Direction (°):").pack()
        self.wind_dir_var = tk.DoubleVar(value=90.0)
        ttk.Entry(control_frame, textvariable=self.wind_dir_var).pack()

        ttk.Label(control_frame, text="Temperature (°C):").pack()
        self.temp_var = tk.DoubleVar(value=28.0)
        ttk.Entry(control_frame, textvariable=self.temp_var).pack()

        ttk.Label(control_frame, text="Humidity (%):").pack()
        self.humidity_var = tk.DoubleVar(value=85.0)
        ttk.Entry(control_frame, textvariable=self.humidity_var).pack()

        ttk.Label(control_frame, text="Timestep (hours):").pack()
        self.dt_var = tk.DoubleVar(value=0.25)
        ttk.Entry(control_frame, textvariable=self.dt_var).pack()

        ttk.Label(control_frame, text="Precipitation Scenario:").pack(pady=(10, 0))
        self.scenario_var = tk.StringVar(value="Normal Rain")
        ttk.Combobox(
            control_frame,
            textvariable=self.scenario_var,
            values=["Normal Rain", "Typhoon", "Extreme Storm"],
            state="readonly",
        ).pack()

        # ✅ Barangay dropdown
        ttk.Label(control_frame, text="Select Barangay:").pack(pady=(10, 0))
        self.barangay_var = tk.StringVar(value=BARANGAYS[0])
        barangay_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.barangay_var,
            values=BARANGAYS,
            state="readonly",
            width=22
        )
        barangay_dropdown.pack()
        barangay_dropdown.bind("<<ComboboxSelected>>", self.set_barangay_bounds)

        # ✅ Mapbox style dropdown
        ttk.Label(control_frame, text="Mapbox Style:").pack(pady=(10, 0))
        ttk.Combobox(
            control_frame,
            textvariable=self.mapbox_style,
            values=["satellite-v9", "streets-v11", "outdoors-v11", "light-v10", "dark-v10"],
            state="readonly",
            width=18
        ).pack()
        ttk.Button(control_frame, text="Apply Style",
                   command=lambda: self.update_frame(self.slider.get())).pack(pady=5)

        ttk.Button(control_frame, text="🚀 Run Simulation", command=self.run_simulation).pack(pady=12)
        ttk.Button(control_frame, text="📊 Show Diagnostics", command=self.show_diagnostics).pack(pady=12)

        ttk.Label(control_frame, text="Animation:").pack(pady=(20, 0))
        self.slider = tk.Scale(
            control_frame, from_=0, to=0, orient="horizontal", length=220,
            command=self.update_frame, state="disabled",
        )
        self.slider.pack()

        # --- Plot area ---
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib figure: static map + line graph side by side
        self.fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_line = self.fig.add_subplot(gs[0, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.canvas.mpl_connect("button_press_event", self.on_map_click)

    # --- Simulation ---
    def run_simulation(self):
        proceed = messagebox.askokcancel(
            "Run Simulation",
            "Start the flood simulation?\n\nThis may take some time."
        )
        if not proceed:
            return

        thread = threading.Thread(target=self._run_simulation_worker)
        thread.daemon = True
        thread.start()

    def _run_simulation_worker(self):
        try:
            dx_m = self.dx_var.get()
            n_particles = self.n_particles_var.get()
            hours = self.hours_var.get()
            self.dt_hours = self.dt_var.get()
            scenario_type = self.scenario_var.get()
            rainfall = self.rainfall_var.get()
            water_level = self.water_level_var.get()
            flow_velocity = self.flow_velocity_var.get()
            wind_speed = self.wind_speed_var.get()
            wind_dir = self.wind_dir_var.get()
            temperature = self.temp_var.get()
            humidity = self.humidity_var.get()

            config = HighAccuracyNagaCityConfig(dx_m=dx_m, dy_m=dx_m)
            predictor = UltraAccurateFloodPredictor(config, n_particles=n_particles)

            precip_field = np.full((config.ny, config.nx), rainfall, dtype=float)
            scenario = [precip_field for _ in range(int(hours / self.dt_hours))]
            if scenario_type == "Typhoon":
                scenario = [r * 2.0 for r in scenario]
            elif scenario_type == "Extreme Storm":
                scenario = [r * 3.0 for r in scenario]

            results = predictor.run_ultra_accurate_simulation(
                scenario,
                n_steps=int(hours / self.dt_hours),
                dt_hours=self.dt_hours,
                water_level=water_level,
                flow_velocity=flow_velocity,
                wind_speed=wind_speed,
                wind_dir=wind_dir,
                temperature=temperature,
                humidity=humidity
            )

            self.root.after(0, self._finish_simulation, results)

        except Exception as e:
            print("⚠️ Simulation failed:", e)
            messagebox.showerror("Simulation Error", str(e))

    def _finish_simulation(self, results):
        self.results = results
        self.slider.config(to=len(self.results) - 1, state="normal")
        self.slider.set(len(self.results) - 1)
        self.update_frame(len(self.results) - 1)
        messagebox.showinfo("Simulation", "✅ Simulation finished!")

    # --- Barangay zoom ---
    def set_barangay_bounds(self, event=None):
        brgy = self.barangay_var.get()
        self.selected_bounds = BARANGAY_BOUNDS.get(brgy, None)
        self.update_frame(self.slider.get())

    # --- Fetch Mapbox static ---
    def _fetch_mapbox_image(self, lon_min, lon_max, lat_min, lat_max, width=800, height=800):
        style = self.mapbox_style.get()
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
            f"[{lon_min},{lat_min},{lon_max},{lat_max}]/{width}x{height}"
            f"?access_token={MAPBOX_TOKEN}"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return Image.open(BytesIO(r.content))
            else:
                print("⚠️ Mapbox fetch failed:", r.text)
        except Exception as e:
            print("⚠️ Mapbox fetch error:", e)
        return None

    def _draw_basemap(self, ax, wm_extent, lonlat_extent):
        x0, x1, y0, y1 = wm_extent
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])

        lon_min, lon_max, lat_min, lat_max = lonlat_extent
        map_img = self._fetch_mapbox_image(lon_min, lon_max, lat_min, lat_max)
        if map_img is not None:
            ax.imshow(map_img, extent=[x0, x1, y0, y1], origin="upper")

        if self.show_grid.get():
            ax.set_xticks(np.linspace(x0, x1, 10))
            ax.set_yticks(np.linspace(y0, y1, 10))
            ax.grid(True, alpha=0.2, color="white", linewidth=0.5)
        else:
            ax.grid(False)

    # --- Update static map + line graph ---
    def update_frame(self, frame_idx):
        if self.results is None:
            return

        frame_idx = int(float(frame_idx))
        flood_depth = np.array(self.results[frame_idx])
        ny, nx = flood_depth.shape

        if self.selected_bounds:
            lon_min, lon_max, lat_min, lat_max = self.selected_bounds
        else:
            lon_min, lon_max, lat_min, lat_max = self.extent_ll

        x0, y0 = self.to_merc.transform(lon_min, lat_min)
        x1, y1 = self.to_merc.transform(lon_max, lat_max)
        wm_extent = [x0, x1, y0, y1]

        self.ax_map.clear()
        self._draw_basemap(self.ax_map, wm_extent, (lon_min, lon_max, lat_min, lat_max))

        mask = np.ma.masked_less_equal(flood_depth, 0.15)
        self.ax_map.imshow(
            mask, cmap="Blues", alpha=0.6,
            origin="upper", extent=wm_extent, interpolation="bilinear"
        )
        self.ax_map.set_title(f"Flood Overlay (t = {frame_idx * self.dt_hours:.2f} h)")

        if self.gauge_xy:
            self.ax_map.plot(*self.gauge_xy, "ro", markersize=7, markeredgecolor="black")

        gx, gy = self.gauge_xy or ((x0 + x1) / 2, (y0 + y1) / 2)
        frac_x = (gx - x0) / (x1 - x0) if (x1 - x0) != 0 else 0.5
        frac_y = 1 - (gy - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
        col = np.clip(int(frac_x * nx), 0, nx - 1)
        row = np.clip(int(frac_y * ny), 0, ny - 1)

        depths = [float(np.array(self.results[i])[row, col]) for i in range(len(self.results))]
        times = [i * self.dt_hours for i in range(len(self.results))]

        self.ax_line.clear()
        self.ax_line.plot(times, depths, color="blue", linewidth=1.5)
        self.ax_line.set_title("Flood Depth vs Time")
        self.ax_line.set_xlabel("Time (hours)")
        self.ax_line.set_ylabel("Depth (m)")
        self.ax_line.grid(True)

        self.canvas.draw()

    def on_map_click(self, event):
        if event.inaxes != self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.gauge_xy = (event.xdata, event.ydata)
        self.update_frame(self.slider.get())

    # --- Diagnostics ---
    def show_diagnostics(self):
        if self.results is None:
            messagebox.showwarning("Warning", "Run a simulation first!")
            return
        results_array = np.array(self.results)
        if results_array.ndim == 3:
            ensemble_median = np.median(results_array, axis=(1, 2))
        else:
            ensemble_median = results_array

        observations = ensemble_median + np.random.normal(0, 5, size=len(ensemble_median))
        truth = ensemble_median * 0.95
        params = {"roughness": np.random.normal(0.05, 0.01, 50),
                  "infiltration": np.random.normal(1.0, 0.2, 50)}
        residuals = {"predicted": list(ensemble_median),
                     "residuals": list(observations - ensemble_median)}
        ess_history = np.linspace(10, 100, len(ensemble_median))
        resampling_threshold = 50

        results_dict = {
            "rainfall": np.random.rand(len(ensemble_median)) * 10,
            "runoff": np.random.rand(len(ensemble_median)) * 50,
            "ci": (ensemble_median * 0.8, ensemble_median * 1.2),
            "phase_space": (ensemble_median, np.gradient(ensemble_median))
        }

        create_particle_filter_diagnostics(
            results_dict, observations, ensemble_median, truth,
            params, residuals, ess_history, resampling_threshold
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = FloodGUI(root)
    root.mainloop()
