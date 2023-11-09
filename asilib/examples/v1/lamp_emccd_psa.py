from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import asilib.map
from asilib.asi.psa_emccd import psa_emccd

asi = psa_emccd(
    'vee',
    time=datetime(2022, 3, 5, 11, 0),
    redownload=False,
)
fig = plt.figure(figsize=(12, 4))
gs = matplotlib.gridspec.GridSpec(1, 3, fig)
ax = fig.add_subplot(gs[0,0])
bx = asilib.map.create_simple_map(
    lon_bounds=(asi.meta['lon']-8, asi.meta['lon']+8),
    lat_bounds=(asi.meta['lat']-4, asi.meta['lat']+4),
    fig_ax=(fig, gs[0,1])
    )
cx = fig.add_subplot(gs[0,2])
ax.axis('off')
asi.plot_fisheye(ax=ax)
asi.plot_map(ax=bx)

time, image = asi.data
cx.hist(image.flatten(), bins=np.linspace(1E3, 1E4))
cx.set(yscale='log')
plt.tight_layout()
plt.show()