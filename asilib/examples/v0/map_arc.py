"""
Recreate the auroral arc that is shown in Fig. 2b in:
Donovan, E., Liu, W., Liang, J., Spanswick, E., Voronkov, I., Connors, M., ... & Rae,
I. J. (2008). Simultaneous THEMIS in situ and auroral observations of a small substorm.
Geophysical Research Letters, 35(17).
"""

from datetime import datetime

import matplotlib.pyplot as plt

import asilib
import asilib.map
import asilib.asi.themis

time = datetime(2007, 3, 13, 5, 8, 45)
asi_array_code = 'THEMIS'
location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
map_alt = 110
min_elevation = 2

ax = asilib.map.create_map(lon_bounds=(-140, -60), lat_bounds=(40, 82))

for location_code in location_codes:
    img = asilib.asi.themis.themis(location_code, time=time, alt=map_alt)
    img.plot_map(ax=ax, min_elevation=min_elevation)

ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
plt.show()
