# flood_presets.py
# Some example presets (Typhoon Uwan / Naga River)


import json

from matplotlib import path


UWAN = {
"name": "Uwan_Naga_Default",
"Nx": 120,
"Ny": 60,
"Lx": 40000.0,
"Ly": 20000.0,
"CN": 75.0,
"manning_n": 0.035,
"t_end": 7200.0,
"output_interval": 60.0,
"Pmax_mm_h": 200.0,
"Rmax_km": 60.0,
"alphaP": 0.18,
"sigmaX": 8.0,
"n_ens": 12
}


QUICK = {
"name": "Quick_Test",
"Nx": 20,
"Ny": 10,
"Lx": 20000.0,
"Ly": 10000.0,
"CN": 70.0,
"manning_n": 0.03,
"t_end": 120.0,
"output_interval": 5.0,
"Pmax_mm_h": 100.0,
"Rmax_km": 30.0,
"alphaP": 0.15,
"sigmaX": 7.89,
"n_ens": 3
}




def save_preset(path, preset):
    with open(path, 'w') as f:
        json.dump(preset, f, indent=2)