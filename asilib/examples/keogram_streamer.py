import matplotlib.pyplot as plt

import asilib

asi_array_code = 'THEMIS'
location_code = 'RANK'

asilib.plot_keogram(
    ['2008-02-05/08:00:00', '2008-02-05/09:00:00'], asi_array_code, location_code, map_alt=110
)
plt.show()
