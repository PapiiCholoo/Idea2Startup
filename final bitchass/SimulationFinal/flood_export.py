# flood_export.py
# Export utilities: CSV, PNGs, GIF, PDF report


import os
import numpy as np
import imageio
from matplotlib.backends.backend_pdf import PdfPages
import json




def export_csv_timeseries(path, results):
# results: dict with keys time, rain, discharge, total_volume
	# results: dict with keys time, rain, discharge, total_volume
	arr = np.vstack([results['time'], results['mean_rain_mmhr'], results['discharge'], results['total_volume']]).T
	header = 'time,mean_rain_mmhr,discharge_m3s,total_volume_m3'
	np.savetxt(path, arr, delimiter=',', header=header, comments='')


def export_pngs(path_dir, figs):
	os.makedirs(path_dir, exist_ok=True)
	paths = []
	for i, fig in enumerate(figs):
		p = os.path.join(path_dir, f'panel_{i+1}.png')
		fig.savefig(p, dpi=200)
		paths.append(p)
	return paths


def export_pdf_report(path_pdf, figs, summary_text=None):
	with PdfPages(path_pdf) as pdf:
		for fig in figs:
			pdf.savefig(fig)
		if summary_text:
			# create a simple page
			import matplotlib.pyplot as plt
			fig = plt.figure(figsize=(8.27, 11.69))
			fig.clf()
			ax = fig.add_subplot(111)
			ax.axis('off')
			ax.text(0.05, 0.95, summary_text, va='top')
			pdf.savefig(fig)


def export_gif(path_gif, frames, fps=6):
	# frames: list of 2D arrays or image file paths
	imgs = []
	for f in frames:
		if isinstance(f, str):
			imgs.append(imageio.imread(f))
		else:
			# normalize array safely to 0-255 uint8
			maxv = np.nanmax(f)
			if maxv == 0 or np.isnan(maxv):
				arr = (255 * np.clip(f, 0, 1)).astype('uint8')
			else:
				arr = (255 * np.clip(f / maxv, 0, 1)).astype('uint8')
			# convert grayscale 2D to RGB
			if arr.ndim == 2:
				arr = np.stack([arr] * 3, axis=-1)
			imgs.append(arr)
	imageio.mimsave(path_gif, imgs, fps=fps)


def save_json(path, obj):
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2)