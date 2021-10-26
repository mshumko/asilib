""" 
Recreate the auroral arc that is shown in Fig. 2b in:
Donovan, E., Liu, W., Liang, J., Spanswick, E., Voronkov, I., Connors, M., ... & Rae, 
I. J. (2008). Simultaneous THEMIS in situ and auroral observations of a small substorm. 
Geophysical Research Letters, 35(17).
"""

from datetime import datetime

import matplotlib.pyplot as plt

import asilib

time = datetime(2007, 3, 13, 5, 8, 45)
asi_array_code = 'THEMIS'
location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
map_alt = 110
min_elevation = 2

ax = asilib.create_cartopy_map(map_style='white', lon_bounds=(-160, -52), lat_bounds=(40, 82))

for location_code in location_codes:
    asilib.plot_map(
        asi_array_code, location_code, time, map_alt, ax=ax, min_elevation=min_elevation
    )

ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
plt.show()
