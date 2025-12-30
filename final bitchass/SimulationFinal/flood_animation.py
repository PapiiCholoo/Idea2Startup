# flood_animation.py
# lightweight animation builder. If backend supports frame capture, it will be used.
# Otherwise it falls back to interpolating between initial & final depth.


import numpy as np

try:
    from tqdm import tqdm
except Exception:
    # fallback tqdm that is a no-op iterator wrapper
    def tqdm(it):
        return it


def build_frames_from_results(results, nframes=30):
    # results expected to have 'flood_depth' final 2D array, and optionally 'depth_frames' key
    if 'depth_frames' in results and results['depth_frames'] is not None:
        return list(results['depth_frames'])

    # fallback: interpolate from zeros to final
    final = results['flood_depth']
    frames = []
    for k in tqdm(range(nframes)):
        alpha = (k + 1) / float(nframes)
        frames.append(final * alpha)
    return frames