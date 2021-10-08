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

# At this time asilib doesn't have an intuitive way to map multiple ASI images, so you need
# to plot the first imager, and reuse the retuned subplot map to plot the other images.
image_time, image, skymap, ax, pcolormesh_obj = asilib.plot_map(
    asi_array_code, location_codes[0], time, map_alt, 
    map_style='green', min_elevation=min_elevation)

for location_code in location_codes[1:]:
    asilib.plot_map(asi_array_code, location_code, time, map_alt, ax=ax, min_elevation=min_elevation)

ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
plt.show()