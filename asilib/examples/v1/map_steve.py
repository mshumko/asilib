"""
Maps an image of STEVE (the thin band). Reproduced from 
http://themis.igpp.ucla.edu/nuggets/nuggets_2018/Gallardo-Lacourt/fig2.jpg
"""

from datetime import datetime

import matplotlib.pyplot as plt

import asilib.asi
import asilib.map

ax = asilib.map.create_map(lon_bounds=(-127, -100), lat_bounds=(45, 65))

asi = asilib.asi.themis('ATHA', time=datetime(2010, 4, 5, 6, 7, 0), alt=110)
asi.plot_map(ax=ax)
plt.tight_layout()
plt.show()
