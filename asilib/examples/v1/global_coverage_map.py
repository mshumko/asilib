"""
Make a 
"""

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature

import asilib.asi

asi_arrays = {
    'THEMIS':(asilib.asi.themis, asilib.asi.themis_info),
    'REGO':(asilib.asi.rego, asilib.asi.rego_info),
    'TREx-NIR':(asilib.asi.trex_nir, asilib.asi.trex_nir_info),
    'TREx-RGB':(asilib.asi.trex_rgb, asilib.asi.trex_rgb_info),
    }

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection=ccrs.Orthographic(-100, 55)))

ax.add_feature(cartopy.feature .LAND, color='g')
ax.add_feature(cartopy.feature .OCEAN, color='w')
ax.add_feature(cartopy.feature .COASTLINE, edgecolor='k')

ax.gridlines()

for array, (loader, info_df) in asi_arrays.items():
    pass

plt.tight_layout()
plt.show()