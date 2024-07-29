from datetime import datetime

import matplotlib.pyplot as plt

import asilib.asi

location_code = 'RANK'
time = datetime(2017, 9, 15, 2, 34, 0)

asi = asilib.asi.themis(location_code, time=time)
ax, im = asi.plot_fisheye()
plt.colorbar(im)
ax.axis('off')
plt.show()
