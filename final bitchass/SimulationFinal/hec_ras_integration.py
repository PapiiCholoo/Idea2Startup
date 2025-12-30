
import os
import time
import glob
import numpy as np
import pandas as pd

try:
    import win32com.client
except Exception as e:
    raise RuntimeError("pywin32 (win32com) is required. Install with: pip install pywin32") from e

# -----------------------
# USER CONFIG — EDIT THESE
# -----------------------
# 1) Full path to your HEC-RAS project file (.prj)
HECRAS_PROJECT = r"C:\Users\Theomel\Desktop\Idea2Startup\DSM_Naga\Naga River"    

# 2) Name of the plan inside the project to run (exact name as appears in HEC-RAS plan list)
HECRAS_PLAN_NAME = "Naga River"                             # <-- CHANGE THIS

# 3) Optional: path (or pattern) to the HEC-RAS results HDF5 file that the project creates.
#    If left None, script will try to find new .hdf or .h5 files inside project folder after run.
HECRAS_RESULTS_FILE = r"C:\Users\Theomel\Desktop\Idea2Startup\DSM_Naga\Naga River\Results\Naga City Flood Inundation Plan.h5"

# 4) Show computation window = True/False
SHOW_HECRAS_GUI = False

# 5) Whether to hide computation progress dialog during compute
HIDE_COMPUTE_WINDOW = True

# -----------------------
# Implementation details
# -----------------------
PROGIDS_TO_TRY = [
    "RAS610.HECRASController",
    "RAS600.HECRASController",
    "RAS5X.HECRASController",
    "RAS505.HECRASControllr",
    "RAS503.HECRASController",
    "RAS502.HECRASController",
]

def get_hecras_controller():
    """Try several ProgIDs and return the first working HECRAS controller COM object."""
    for progid in PROGIDS_TO_TRY:
        try:
            ras = win32com.client.Dispatch(progid)
            print(f"[hec] Connected to HEC-RAS controller with ProgID: {progid}")
            return ras
        except Exception:
            continue
    raise RuntimeError("Could not find a HEC-RAS controller COM object. Ensure HEC-RAS is installed and pywin32 is installed.")

def open_and_run_project(project_path, plan_name, show_gui=False, hide_compute=True, results_file_hint=None, timeout_minutes=30):
    """
    Open the HEC-RAS project, set the plan, run it, and wait for completion.
    Returns path to results file (if found) or None.
    """
    ras = get_hecras_controller()

    # Optionally show the HEC-RAS application window
    try:
        if show_gui:
            if hasattr(ras, "ShowRAS"):
                ras.ShowRAS()
    except Exception:
        pass

    # Optionally hide computation window if supported
    try:
        if hide_compute and hasattr(ras, "Compute_HideComputationWindow"):
            ras.Compute_HideComputationWindow(1)
    except Exception:
        # not critical
        pass

    print(f"[hec] Opening project: {project_path}")
    ras.Project_Open(project_path)

    # Set the plan to run
    print(f"[hec] Setting current plan: {plan_name}")
    try:
        ras.Plan_SetCurrent(plan_name)
    except Exception as e:
        print("[hec] Warning: Plan_SetCurrent failed. Ensure plan name is exact. Error:", e)

    # Start compute of current plan (args: saveResults?, showComputeWindow? - many examples use (0,0) or (1,0))
    print("[hec] Starting compute of current plan...")
    try:
        # Many examples use Compute_CurrentPlan(0,0) — common pattern from HEC-RAS controller docs.
        ras.Compute_CurrentPlan(0,0)
    except Exception as e:
        raise RuntimeError("Compute_CurrentPlan call failed. Check HEC-RAS version and controller availability.") from e

    # Wait for compute completion
    print("[hec] Waiting for HEC-RAS compute to finish...")
    start = time.time()
    timeout = timeout_minutes * 60
    while True:
        try:
            # The controller exposes a Compute_Complete property (boolean)
            complete = bool(ras.Compute_Complete)
        except Exception:
            # If property not available, attempt a safe sleep with a crude timeout
            complete = False
        if complete:
            print("[hec] Compute complete.")
            break
        if (time.time() - start) > timeout:
            raise TimeoutError("HEC-RAS compute did not complete within timeout.")
        time.sleep(1)

    # Try to close the project (optional)
    try:
        ras.Project_Close()
    except Exception:
        pass

    # Try to quit RAS
    try:
        if hasattr(ras, "QuitRas"):
            ras.QuitRas()
        elif hasattr(ras, "QuitRAS"):
            ras.QuitRAS()
    except Exception:
        pass

    # Find results file: either user-specified or search for newest .hdf/.h5/.csv in project folder
    if results_file_hint and os.path.exists(results_file_hint):
        print(f"[hec] Results found at provided path: {results_file_hint}")
        return results_file_hint

    project_dir = os.path.dirname(project_path)
    # common result extensions
    patterns = ["**/*.hdf", "**/*.h5", "**/*_results.hdf", "**/*.csv"]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(project_dir, pat), recursive=True))
    if candidates:
        # pick newest file
        newest = max(candidates, key=os.path.getmtime)
        print(f"[hec] Found candidate results file: {newest}")
        return newest

    print("[hec] No results file found automatically. You may need to set HECRAS_RESULTS_FILE to the exact results HDF path.")
    return None

def read_results_simple(results_path):
    """
    Very simple results reader:
     - if HDF5 (.hdf/.h5): attempt to read using h5py (structure varies by HEC-RAS version)
     - if CSV: load with pandas
    This function returns a dict with raw datasets or DataFrames suitable for feeding your model.
    You will likely adapt parsing to the exact tables you need (e.g., profile elevations, velocities).
    """
    import os
    if results_path is None:
        return {}

    ext = os.path.splitext(results_path)[1].lower()
    if ext in (".csv",):
        df = pd.read_csv(results_path)
        return {"csv": df}
    elif ext in (".hdf", ".h5"):
        try:
            import h5py
        except Exception as e:
            raise RuntimeError("h5py required to read HEC-RAS HDF5 results. Install: pip install h5py") from e
        data = {}
        with h5py.File(results_path, "r") as f:
            # HEC-RAS HDF structures vary by version. We'll try to list top-level groups
            for k in f.keys():
                data[k] = {}  # do not auto-load giant arrays here
            # Example: you can later inspect groups/datasets and pick the ones you need
        return {"h5_groups": list(data.keys())}
    else:
        raise RuntimeError(f"Unsupported results extension: {ext}")

# High-level function to call from chaos_flood_model
def run_hecras_and_get_results(project=HECRAS_PROJECT, plan=HECRAS_PLAN_NAME, results_hint=HECRAS_RESULTS_FILE):
    results_path = open_and_run_project(project, plan, show_gui=SHOW_HECRAS_GUI,
                                       hide_compute=HIDE_COMPUTE_WINDOW, results_file_hint=results_hint)
    parsed = read_results_simple(results_path)
    # TODO: convert parsed results to the boundary data structure your chaos_flood_model expects.
    # Example placeholder (modify to match your model):
    bc = {
        "hec_results_path": results_path,
        "raw_parsed": parsed
    }
    return bc

# If run as script, demonstrate behavior (prints)
if __name__ == "__main__":
    print("Running HEC-RAS integration demo...")
    bc = run_hecras_and_get_results()
    print("Boundary conditions (summary):", {k: (v if isinstance(v, str) else type(v)) for k,v in bc.items()})
