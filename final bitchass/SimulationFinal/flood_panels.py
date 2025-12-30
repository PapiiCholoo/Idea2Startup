# flood_panels.py
# UI panel helper functions: create charts and widgets to insert into main app


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def plot_flood_map(fig, depth_arr, title="Flood Depth"):
	fig.clf()
	ax = fig.add_subplot(111)
	im = ax.imshow(depth_arr, origin='lower')
	fig.colorbar(im, ax=ax, label='Depth (m)')
	ax.set_title(title)
	return fig




def plot_lorenz_3d(fig, x, y, z, color='#1A73E8'):
	fig.clf()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x, y, z, color=color)
	ax.set_title('Lorenz Attractor')
	return fig




def plot_rain_vs_discharge(fig, t, rain, discharge):
	fig.clf()
	ax = fig.add_subplot(111)
	ax.plot(t, rain, label='Rain (mm/hr)')
	ax.plot(t, discharge, label='Discharge (m3/s)')
	ax.legend()
	ax.set_title('Rainfall vs Discharge')
	return fig




def plot_ensemble_spread(fig, t, ensembles):
	fig.clf()
	ax = fig.add_subplot(111)
	ens = np.array(ensembles)
	mean = np.mean(ens, axis=0)
	std = np.std(ens, axis=0)
	ax.plot(t, mean, color='black')
	ax.fill_between(t, mean - std, mean + std, alpha=0.5)
	ax.set_title('Ensemble Spread')
	return fig
