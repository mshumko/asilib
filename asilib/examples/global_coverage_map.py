"""
Create a world map showcasing the ASI arrays that asilib supports.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import numpy as np
import requests
import cartopy.crs as ccrs
import cartopy.feature

import asilib.asi

asi_arrays = {
    'THEMIS':(asilib.asi.themis, asilib.asi.themis_info, 'r', (10, 11)),
    'REGO':(asilib.asi.rego, asilib.asi.rego_info, 'k', (8, 9)),
    'TREx-NIR':(asilib.asi.trex_nir, asilib.asi.trex_nir_info, 'c', (12, 13)),
    'TREx-RGB':(asilib.asi.trex_rgb, asilib.asi.trex_rgb_info, 'purple', (14, 15)),
    }

time = datetime(2022, 1, 1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection=ccrs.Orthographic(-100, 55))

ax.add_feature(cartopy.feature .LAND, color='g')
ax.add_feature(cartopy.feature .OCEAN, color='w')
ax.add_feature(cartopy.feature .COASTLINE, edgecolor='k')

ax.gridlines()

legend_handles = []
for i, (array, (loader, info_df, color, elevation_range)) in enumerate(asi_arrays.items()):
    asi_array_info = info_df()
    for location in asi_array_info['location_code']:
        try:
            asi = loader(location, time=time, load_images=False)
        except requests.exceptions.HTTPError as err:
            if 'Not Found for url' in str(err):
                print(err)
                continue
            else:
                raise
        c = np.ones_like(asi.skymap['lat'][:-1, :-1])
        c[
            (asi.skymap['el'] < elevation_range[0]) |
            (asi.skymap['el'] > elevation_range[1]) | 
            np.isnan(asi.skymap['el'])
            ] = np.nan
        asi._pcolormesh_nan(
            asi.skymap['lon'],
            asi.skymap['lat'],
            c,
            ax,
            cmap=matplotlib.colors.ListedColormap([color]),
            pcolormesh_kwargs={
                'transform':ccrs.PlateCarree(),
            }
        )
        ax.text(0, i/30, array, va='bottom', color=color, transform=ax.transAxes, fontsize=20)
ax.text(0.99, 0, f'Generated on {datetime.now().date()}', va='top', ha='right', transform=ax.transAxes, fontsize=15)
ax.set_title('Spatial Coverage of Imagers Supported by asilib', fontsize=25)        
plt.tight_layout()
plt.show()