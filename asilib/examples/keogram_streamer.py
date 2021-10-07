import matplotlib.pyplot as plt

import asilib

asi_array_code = 'THEMIS'
location_code = 'RANK'
time_range = ['2008-02-05/08:00:00', '2008-02-05/09:00:00']

asilib.plot_keogram(asi_array_code, location_code, time_range, map_alt=110)
plt.show()
