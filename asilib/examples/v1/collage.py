"""
A collage to sell asilib to new users
"""
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy as np

import asilib.asi
import asilib.map


location_code = 'RANK'
time = datetime(2017, 9, 15, 2, 34, 0)
time_range = (datetime(2017, 9, 15, 2, 0, 0), datetime(2017, 9, 15, 4, 0, 0))
map_alt_km = 110
fontsize = 17

lon_bounds = (-102, -82)
lat_bounds = (56, 68)

# Create plots.
# ax is a 2x2 subplot grid with the fisheye and map images from THEMIS and REGO.
# bx and cx subplots are the corresponding keograms.
fig = plt.figure(figsize=(4.5, 7))
gs = matplotlib.gridspec.GridSpec(3, 2, fig)
ax = np.zeros((2, 2), dtype=object)
ax[0, 0] = plt.subplot(gs[0, 0])
ax[1, 0] = plt.subplot(gs[1, 0])
ax[0, 1] = asilib.map.create_map(
    lon_bounds=lon_bounds, lat_bounds=lat_bounds, fig_ax=(fig, gs[0, 1])
)
ax[1, 1] = asilib.map.create_map(
    lon_bounds=lon_bounds, lat_bounds=lat_bounds, fig_ax=(fig, gs[1, 1])
)
bx = plt.subplot(gs[2, :])
# cx = plt.subplot(gs[3,:])

themis1 = asilib.asi.themis(location_code, time=time)
themis1.plot_fisheye(ax=ax[0, 0], label=False)
themis1.plot_map(ax=ax[0, 1], asi_label=False)
rego1 = asilib.asi.rego(location_code, time=time)
rego1.plot_fisheye(ax=ax[1, 0], label=False)
rego1.plot_map(ax=ax[1, 1], asi_label=False)

themis2 = asilib.asi.themis(location_code, time_range=time_range)
themis2.plot_keogram(ax=bx, title=False)
# rego2 = asilib.asi.rego(location_code, time_range=time_range)
# rego2.plot_keogram(ax=cx, title=False)

ax[0, 0].axis('off')
ax[1, 0].axis('off')
bx.axis('off')

plt.suptitle(f'asilib', fontsize=20)
plt.subplots_adjust(top=0.941, bottom=0.01, left=0.023, right=0.977, hspace=0.1, wspace=0.11)
plt.show()
