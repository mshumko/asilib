from datetime import datetime

import matplotlib.pyplot as plt
import asilib.map
import asilib
from asilib.asi.trex import trex_rgb

time = datetime(2022, 12, 19, 14, 4)
asi = trex_rgb('YKNF', time=time)
ax = asilib.map.create_map(
    lon_bounds=(asi.meta['lon']-10, asi.meta['lon']+10),
    lat_bounds=(asi.meta['lat']-5, asi.meta['lat']+5)
)
asi.plot_map(ax=ax, color_bounds=(50, 100))
plt.tight_layout()
plt.show()