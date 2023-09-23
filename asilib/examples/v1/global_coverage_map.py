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
ax = fig.add_subplot(projection=ccrs.Orthographic(-100, 55))
# fig.subplots_adjust(bottom=0.05, top=0.95,
#                     left=0.04, right=0.95, wspace=0.02)

# Limit the map to -60 degrees latitude and below.
# ax1.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

ax.add_feature(cartopy.feature .LAND, color='g')
ax.add_feature(cartopy.feature .OCEAN, color='w')
ax.add_feature(cartopy.feature .COASTLINE, edgecolor='k')

ax.gridlines()

# # Compute a circle in axes coordinates, which we can use as a boundary
# # for the map. We can pan/zoom as much as we like - the boundary will be
# # permanently circular.
# theta = np.linspace(0, 2*np.pi, 100)
# center, radius = [0.5, 0.5], 0.5
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * radius + center)

# ax2.set_boundary(circle, transform=ax2.transAxes)



for array, (loader, info_df) in asi_arrays.items():
    pass

plt.tight_layout()
plt.show()