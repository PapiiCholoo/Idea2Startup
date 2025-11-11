# ===========================================================
# flood_ui_pro.py  (with 💨 Quick Test Mode)
# ===========================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # use TkAgg for GUI rendering
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import chaos_flood_model as cfm

# ===== COLORS & STYLE =====
PRIMARY = "#0078D7"
LIGHT_BLUE = "#E6F0FA"
BG = "#F8FBFF"
DARK_TEXT = "#0C2340"
FONT = ("Segoe UI", 10)
TITLE_FONT = ("Segoe UI Semibold", 15)


class FloodUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Chaos-Enhanced Flood Prediction System")
        self.root.geometry("1500x900")
        self.root.configure(bg=BG)

        # ---- ttk style ----
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=DARK_TEXT, font=FONT)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[("active", PRIMARY)],
                  foreground=[("active", "white")])
        style.configure("TNotebook", background=LIGHT_BLUE)
        style.configure("TNotebook.Tab", padding=[12, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", PRIMARY)],
                  foreground=[("selected", "white")])

        # ---- header ----
        title = ttk.Label(root, text="Chaos-Enhanced Flood Prediction System",
                          font=("Segoe UI Semibold", 18), foreground=PRIMARY)
        title.pack(pady=10)

        # ---- input frame ----
        input_frame = ttk.LabelFrame(root, text="Simulation Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.entries = {}
        fields = [
            ("Nx", 40), ("Ny", 20),
            ("Lx (m)", 20000.0), ("Ly (m)", 10000.0),
            ("Curve Number (CN)", 70.0), ("Manning n", 0.03),
            ("Simulation Time (s)", 120.0), ("Output Interval (s)", 10.0),
            ("Ensemble Members", 3)
        ]
        for i, (lbl, val) in enumerate(fields):
            ttk.Label(input_frame, text=lbl).grid(row=0, column=i, padx=6, pady=5)
            e = ttk.Entry(input_frame, width=8)
            e.insert(0, str(val))
            e.grid(row=1, column=i, padx=6, pady=5)
            self.entries[lbl] = e

        # ---- buttons ----
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=20, pady=5)

        self.btn_run = ttk.Button(btn_frame, text="▶ Run Simulation", command=self.run_thread)
        self.btn_run.pack(side=tk.LEFT, padx=6)

        # 💨 Quick Test Mode button
        self.quick_mode = False
        self.btn_quick = ttk.Button(btn_frame, text="💨 Quick Test Mode: OFF", command=self.toggle_quick_mode)
        self.btn_quick.pack(side=tk.LEFT, padx=6)

        ttk.Button(btn_frame, text="💾 Save Panel", command=self.save_panel).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="🧾 Export All (PDF)", command=self.export_pdf).pack(side=tk.LEFT, padx=6)

        # ---- progress bar ----
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)
        self.status_label = ttk.Label(root, text="", font=("Segoe UI", 9))
        self.status_label.pack()

        # ---- notebook ----
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tab_titles = [
            "Flood Map", "3D Lorenz Attractor",
            "Rainfall vs Runoff", "Ensemble Spread",
            "Diagnostics", "Summary"
        ]
        self.figures, self.canvases = [], []
        for t in tab_titles:
            frame = ttk.Frame(self.tabs, padding=5)
            self.tabs.add(frame, text=t)
            fig = Figure(figsize=(7, 5), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figures.append(fig)
            self.canvases.append(canvas)

        # ---- footer ----
        ttk.Label(root, text="© 2025 Papi Choloo | Chaos-Enhanced Flood Prediction System",
                  foreground="#555", font=("Segoe UI", 9)).pack(pady=5)
        self.results = None

    # ===========================================================
    # 💨 Quick Mode Toggle
    def toggle_quick_mode(self):
        self.quick_mode = not self.quick_mode
        if self.quick_mode:
            self.btn_quick.config(text="💨 Quick Test Mode: ON", style="Quick.TButton")
            self.status_label.config(text="Quick Test Mode active (small grid, fast test).")
        else:
            self.btn_quick.config(text="💨 Quick Test Mode: OFF")
            self.status_label.config(text="Quick Test Mode deactivated.")

    # ===========================================================
    def run_thread(self):
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def run_simulation(self):
        try:
            self.btn_run.config(state=tk.DISABLED)
            self.status_label.config(text="Running simulation...")
            self.progress['value'] = 0

            # read parameters
            p = {
                "Nx": int(self.entries["Nx"].get()),
                "Ny": int(self.entries["Ny"].get()),
                "Lx": float(self.entries["Lx (m)"].get()),
                "Ly": float(self.entries["Ly (m)"].get()),
                "CN": float(self.entries["Curve Number (CN)"].get()),
                "manning_n": float(self.entries["Manning n"].get()),
                "t_end": float(self.entries["Simulation Time (s)"].get()),
                "output_interval": float(self.entries["Output Interval (s)"].get()),
            }
            n_ens = int(self.entries["Ensemble Members"].get())

            # 💨 Override for Quick Test
            if self.quick_mode:
                p["Nx"], p["Ny"] = 20, 10
                p["t_end"] = 60
                p["output_interval"] = 5
                n_ens = 1

            self.results = []
            for i in range(n_ens):
                self.status_label.config(text=f"Running ensemble {i+1}/{n_ens}...")
                self.progress['value'] = ((i + 1) / n_ens) * 100
                self.root.update_idletasks()
                res = cfm.run_flood_simulation(p)
                self.results.append(res)
                time.sleep(0.2)

            self.status_label.config(text="Simulation completed successfully.")
            self.update_all()
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed:\n{e}")
        finally:
            self.btn_run.config(state=tk.NORMAL)
            self.progress['value'] = 100

    # ===========================================================
    def update_all(self):
        if not self.results:
            return
        res = self.results[0]
        t = res["time"]

        # --- Flood Map ---
        fig = self.figures[0]; fig.clf()
        ax = fig.add_subplot(111)
        ax.set_title("Final Flood Depth Map")
        im = ax.imshow(res["flood_depth"], cmap="Blues", origin="lower")
        fig.colorbar(im, ax=ax, label="Depth (m)")
        self.canvases[0].draw()

        # --- 3D Lorenz ---
        fig = self.figures[1]; fig.clf()
        ax = fig.add_subplot(111, projection="3d")

        for e in self.results:
            ax.plot(e["lorenz_x"], e["lorenz_y"], e["lorenz_z"], alpha=0.4, color="#9CCAF6")

        ax.plot(res["lorenz_x"], res["lorenz_y"], res["lorenz_z"], color=PRIMARY, linewidth=2, label="Main Trajectory")
        ax.scatter(res["lorenz_x"][0], res["lorenz_y"][0], res["lorenz_z"][0], color="green", s=60, label="Start (t=0)")
        ax.scatter(res["lorenz_x"][-1], res["lorenz_y"][-1], res["lorenz_z"][-1], color="red", s=60, label="End (t=end)")

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Lorenz Attractor (Chaotic Component)")
        ax.legend(loc="upper left")
        self.canvases[1].draw()

        # --- Rainfall vs Runoff ---
        fig = self.figures[2]; fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(t, res["rain"], color="#33A1FF", label="Rainfall (mm/hr)")
        ax.plot(t, res["discharge"], color=PRIMARY, label="Discharge (m³/s)")
        ax.legend(); ax.set_title("Rainfall vs Discharge")
        self.canvases[2].draw()

        # --- Ensemble Spread ---
        fig = self.figures[3]; fig.clf()
        ax = fig.add_subplot(111)
        ens = np.array([e["discharge"] for e in self.results])
        mean, std = np.mean(ens, axis=0), np.std(ens, axis=0)
        ax.plot(t, mean, color="black", label="Mean")
        ax.fill_between(t, mean - std, mean + std, color="#A0C4FF", alpha=0.6)
        ax.legend(); ax.set_title("Ensemble Spread (±1σ)")
        self.canvases[3].draw()

        # --- Diagnostics ---
        fig = self.figures[4]; fig.clf()
        ax1 = fig.add_subplot(211)
        ax1.plot(t, res["max_h_ts"], color="#0056A4")
        ax1.set_ylabel("Depth (m)"); ax1.set_title("Maximum Flood Depth")
        ax2 = fig.add_subplot(212)
        ax2.plot(t, res["total_volume"], color="#0078D7")
        ax2.set_ylabel("Volume (m³)"); ax2.set_xlabel("Time (s)")
        self.canvases[4].draw()

        # --- Summary ---
        fig = self.figures[5]; fig.clf()
        ax = fig.add_subplot(111)
        ax.axis("off")
        txt = (
            f"Peak Depth: {np.max(res['max_h_ts']):.2f} m\n"
            f"Peak Discharge: {np.max(res['discharge']):.2f} m³/s\n"
            f"Total Volume: {np.max(res['total_volume']):.2f} m³\n"
            f"Ensemble Members: {len(self.results)}"
        )
        ax.text(0.05, 0.6, txt, fontsize=13, color=PRIMARY)
        self.canvases[5].draw()

    # ===========================================================
    def save_panel(self):
        idx = self.tabs.index(self.tabs.select())
        fig = self.figures[idx]
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            fig.savefig(path, dpi=300)
            messagebox.showinfo("Saved", f"Panel saved to {path}")

    def export_pdf(self):
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            with PdfPages(path) as pdf:
                for fig in self.figures:
                    pdf.savefig(fig)
            messagebox.showinfo("Exported", f"All panels saved to {path}")


# ===========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FloodUI(root)
    root.mainloop()
    