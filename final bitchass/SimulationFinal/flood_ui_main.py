import threading
import importlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any
import numpy as np
import win32com.client  # NEW (HEC-RAS Controller)
import os

# ---------------- HEC-RAS INTEGRATION ----------------
def run_hecras_and_get_results(project_path: str, plan_name: str = ""):
    """
    Runs HEC-RAS using the official Controller API.
    Returns a dictionary formatted exactly like your simulation output.
    """
    try:
        ras = win32com.client.Dispatch("RAS507.HECRASController")

        ras.ShowRas()
        ras.Project_Open(project_path)

        if plan_name:
            ras.Plan_SetCurrent(plan_name)

        ras.Compute_CurrentPlan()

        # ------------ READ RESULTS ------------
        n_nodes = ras.Output_NodeCount()
        xs_names = [ras.Output_NodeName(i + 1) for i in range(n_nodes)]
        q = []
        ws = []

        for i in range(n_nodes):
            vals_q = ras.Output_GetFlow(i + 1, 1)
            vals_ws = ras.Output_GetWSE(i + 1, 1)

            q.append(vals_q)
            ws.append(vals_ws)

        discharge = np.array(q).flatten()
        wse = np.array(ws).flatten()
        depth = wse - np.min(wse)  # simple depth (placeholder)

        Ny = 40
        Nx = int(len(depth) / Ny) if len(depth) >= Ny else 10
        depth_grid = depth[:Nx * Ny].reshape(Ny, Nx)

        time = np.linspace(0, 3600, len(discharge))

        ras.Project_Close()

        return {
            "flood_depth": depth_grid,
            "flood_frames": [depth_grid for _ in range(20)],
            "time": time,
            "rain": np.zeros_like(time),
            "discharge": discharge,
            "ensembles": [discharge for _ in range(3)],
            "max_h_ts": np.max(depth_grid) * np.ones_like(time),
            "total_volume": np.sum(depth_grid) * np.ones_like(time),
            "lorenz_x": time * 0,
            "lorenz_y": time * 0,
            "lorenz_z": time * 0,
        }

    except Exception as e:
        raise RuntimeError(f"HEC-RAS error: {str(e)}")


# ---------------- Optional imports ----------------
GPUStatCollection = None
try:
    mod = importlib.import_module("gpustat")
    GPUStatCollection = getattr(mod, "GPUStatCollection", None)
except Exception:
    GPUStatCollection = None

try:
    import psutil
except Exception:
    psutil = None

# ---------------- GPU / CPU Info ----------------
def gpu_info():
    if GPUStatCollection is None:
        return None
    try:
        stats = GPUStatCollection.new_query()
        if len(stats) == 0:
            return None
        g = stats[0]
        return (
            getattr(g, "name", str(g)),
            getattr(g, "memory_total", 0),
            getattr(g, "memory_used", 0),
            getattr(g, "utilization", 0),
        )
    except Exception:
        return None

def cpu_info():
    if psutil is None:
        return (0.0,)
    try:
        return (psutil.cpu_percent(interval=None),)
    except Exception:
        return (0.0,)

# ---------------- FALLBACK MODEL ----------------
try:
    cfm = importlib.import_module("chaos_flood_model")
except Exception:
    class _CFMStub:
        @staticmethod
        def run_flood_simulation(params):
            import numpy as np
            Nx = params.get("Nx", 40)
            Ny = params.get("Ny", 20)
            depth = np.random.random((Ny, Nx)) * 0.5
            frames = [depth * (i / 10) for i in range(10)]
            t = np.linspace(0, params.get("t_end", 60), 50)
            discharge = 20 + 10 * np.sin(t / 12)
            return {
                "flood_depth": depth,
                "flood_frames": frames,
                "time": t,
                "rain": np.ones_like(t) * params.get("rain_intensity", 50),
                "discharge": discharge,
                "ensembles": [discharge for _ in range(params.get("n_ens", 3))],
                "max_h_ts": np.max(depth) * np.ones_like(t),
                "total_volume": np.sum(depth) * np.ones_like(t),
                "lorenz_x": t * 0.01,
                "lorenz_y": t * 0.005,
                "lorenz_z": t * 0.02,
            }
    cfm = _CFMStub()


# ---------------- PLOTTING UTILS ----------------
def plot_flood_map(fig, data, title=None):
    fig.clear()
    ax = fig.add_subplot(111)
    try:
        im = ax.imshow(data, cmap="Blues", origin="lower")
        fig.colorbar(im, ax=ax)
    except:
        ax.text(0.5, 0.5, "No Data", ha='center')
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def build_frames_from_results(res, n=30):
    for key in ("flood_frames", "frames"):
        if key in res:
            return res[key][:n]
    return [res["flood_depth"] for _ in range(n)]


# ---------------- UI CLASS ----------------
class FloodApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Chaos + HEC-RAS Flood Prediction System")
        self.root.geometry("1500x950")
        self._set_palette()
        self.results = None
        self.hec_mode = tk.BooleanVar(value=False)  # NEW FLAG

        self.header_frame = ttk.Frame(root, style="Header.TFrame", padding=5)
        self.header_frame.pack(fill="x")
        ttk.Label(self.header_frame, text="Flood Prediction", style="Header.TLabel").pack(side="left", padx=10)
        ttk.Checkbutton(self.header_frame, text="Use HEC-RAS", variable=self.hec_mode).pack(side="left", padx=10)

        self._build_parameter_controls()
        self._build_tabs()

        self.info_frame = ttk.LabelFrame(root, text="Simulation Info", padding=10)
        self.info_frame.pack(fill="x", padx=10, pady=5)

        self.peak_depth_label = ttk.Label(self.info_frame, text="Peak Depth: -- m", font=("Arial", 10, "bold"))
        self.peak_depth_label.pack(side="left", padx=10)
        self.peak_discharge_label = ttk.Label(self.info_frame, text="Peak Discharge: -- m³/s", font=("Arial", 10, "bold"))
        self.peak_discharge_label.pack(side="left", padx=10)
        self.total_volume_label = ttk.Label(self.info_frame, text="Total Volume: --", font=("Arial", 10, "bold"))
        self.total_volume_label.pack(side="left", padx=10)

    # ---------------- UI STYLE ----------------
    def _set_palette(self):
        style = ttk.Style(self.root)
        self.root.configure(bg="#e6f2ff")

        style.configure("Header.TFrame", background="#cce6ff")
        style.configure("Header.TLabel", background="#cce6ff", foreground="#003366", font=("Arial", 12, "bold"))

    # ---------------- PARAMETER PANEL ----------------
    def _build_parameter_controls(self):
        self.param_frame = ttk.LabelFrame(self.root, text="Simulation Parameters", padding=10)
        self.param_frame.pack(fill="x", padx=10, pady=5)

        self.inputs = {}

        fields = [
            ("Nx", 40), ("Ny", 20),
            ("Lx (m)", 20000), ("Ly (m)", 10000),
            ("Simulation Time (s)", 600),
            ("Output Interval (s)", 30),
        ]

        for i, (lbl, default) in enumerate(fields):
            ttk.Label(self.param_frame, text=lbl).grid(row=0, column=i, padx=5)
            e = ttk.Entry(self.param_frame, width=8)
            e.insert(0, str(default))
            e.grid(row=1, column=i, padx=5)
            self.inputs[lbl] = e

        self.hec_file_btn = ttk.Button(self.param_frame, text="📂 Select HEC-RAS Project", command=self.pick_hec_file)
        self.hec_file_btn.grid(row=0, column=len(fields), rowspan=2, padx=10)

        self.hec_project_path = None

        self.control_frame = ttk.Frame(self.root, padding=5)
        self.control_frame.pack(fill="x", padx=10)

        self.run_btn = ttk.Button(self.control_frame, text="▶ Run Simulation", command=self.run_pressed)
        self.run_btn.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(self.control_frame, length=300)
        self.progress.pack(side="right", padx=10)

    def pick_hec_file(self):
        path = filedialog.askopenfilename(filetypes=[("HEC-RAS Project", "*.prj")])
        if path:
            self.hec_project_path = path
            messagebox.showinfo("Selected", os.path.basename(path))

    # ---------------- TABS ----------------
    def _build_tabs(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.figures = []
        self.canvases = []

        titles = ["Flood Map", "Animation", "3D", "Hydrograph", "Ensemble", "Diagnostics"]
        for t in titles:
            frm = ttk.Frame(self.tabs)
            self.tabs.add(frm, text=t)
            fig = Figure(figsize=(6, 4))
            canvas = FigureCanvasTkAgg(fig, master=frm)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures.append(fig)
            self.canvases.append(canvas)

    # ---------------- RUN ----------------
    def run_pressed(self):
        self.progress["value"] = 20
        self.run_btn.config(state="disabled")

        threading.Thread(target=self._run_backend, daemon=True).start()

    def _run_backend(self):
        try:
            if self.hec_mode.get():
                if not self.hec_project_path:
                    raise RuntimeError("Select a HEC-RAS project first!")
                res = run_hecras_and_get_results(self.hec_project_path)
            else:
                params = {k.replace(" (m)", ""): float(e.get()) for k, e in self.inputs.items()}
                res = cfm.run_flood_simulation(params)

            self.results = res
            self.root.after(0, self._update_panels)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            self.root.after(0, lambda: self.progress.configure(value=100))

    # ---------------- UPDATE GUI ----------------
    def _update_panels(self):
        res = self.results
        if not res:
            return

        # Flood map
        plot_flood_map(self.figures[0], res["flood_depth"], "Flood Depth")
        self.canvases[0].draw()

        # Animation (frame 0)
        frames = build_frames_from_results(res)
        plot_flood_map(self.figures[1], frames[0], "Animation")
        self.canvases[1].draw()

        # Hydrograph
        fig = self.figures[3]; fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(res["time"], res["discharge"])
        ax.set_title("Discharge")
        self.canvases[3].draw()

        # Ensemble
        ens = np.array(res["ensembles"])
        fig = self.figures[4]; fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(res["time"], ens.mean(axis=0))
        self.canvases[4].draw()

        # Diagnostics
        fig = self.figures[5]; fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(res["time"], res["max_h_ts"])
        ax.set_title("Max Depth")
        self.canvases[5].draw()

        self.peak_depth_label.config(text=f"Peak Depth: {np.max(res['max_h_ts']):.2f} m")
        self.peak_discharge_label.config(text=f"Peak Discharge: {np.max(res['discharge']):.2f} m³/s")
        self.total_volume_label.config(text=f"Total Volume: {np.sum(res['flood_depth']):.2f}")


# ---------------- MAIN ----------------
def main():
    root = tk.Tk()
    app = FloodApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
