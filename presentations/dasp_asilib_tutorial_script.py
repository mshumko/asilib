"""
A script to download all of the necessary TREx data and animate the mosaic
in preparation for the student workshop held on February 19th, 2024 at the 
Division of Atmospheric and Space Physics (DASP) meeting at the University 
of Alberta. 
"""

import asilib
import asilib.asi
import asilib.map

import matplotlib.pyplot as plt
import matplotlib.dates

time_range = ('2023-02-24T05:00', '2023-02-24T07:00')

# Usually we will need to specify locations ourselves, but for this time range 
# all of the TREx-RGB imagers were operating.
trex_metadata = asilib.asi.trex_rgb_info()

asi = asilib.asi.trex_rgb('GILL', time_range=time_range) 
ax, p = asi.plot_keogram(aacgm=True)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax.set_ylabel(f'$\lambda_{{AACGM}}$ [deg]')
plt.show()