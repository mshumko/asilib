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
import asilib.asi.mango

asi_arrays = {
    'THEMIS':(asilib.asi.themis, asilib.asi.themis_info, 'k', (10, 11)),
    'REGO':(asilib.asi.rego, asilib.asi.rego_info, 'pink', (8, 9)),
    'TREx-NIR':(asilib.asi.trex_nir, asilib.asi.trex_nir_info, 'c', (12, 13)),
    'TREx-RGB':(asilib.asi.trex_rgb, asilib.asi.trex_rgb_info, 'purple', (14, 15)),
    }

mango_sations = [
    ['CVO', datetime(2024, 10, 25, 5, 28), 'r', (15, 16)],
    ['CFS', datetime(2024, 10, 25, 5, 28), 'r', (15, 16)],
    ['MTO', datetime(2024, 10, 25, 5, 28), 'r', (15, 16)],
    ['EIO', datetime(2024, 10, 25, 5, 28), 'r', (15, 16)],
    ['PAR', datetime(2024, 10, 25, 5, 28), 'r', (15, 16)]
]

time = datetime(2022, 1, 1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection=ccrs.Orthographic(-100, 55))

ax.add_feature(cartopy.feature .LAND, color='g')
ax.add_feature(cartopy.feature .OCEAN, color='w')
ax.add_feature(cartopy.feature .COASTLINE, edgecolor='k')

ax.gridlines()

for mango_station in mango_sations:
    asi = asilib.asi.mango.mango(mango_station[0], 'redline', time=mango_station[1])

    c = np.ones_like(asi.skymap['lat'])
    c[
        (asi.skymap['el'] < mango_station[3][0]) |
        (asi.skymap['el'] > mango_station[3][1]) | 
        np.isnan(asi.skymap['el'])
        ] = np.nan
    asi._pcolormesh_nan(
        asi.skymap['lon'],
        asi.skymap['lat'],
        c,
        ax,
        cmap=matplotlib.colors.ListedColormap([mango_station[2]]),
        pcolormesh_kwargs={
            'transform':ccrs.PlateCarree(),
        }
    )
ax.text(0, 4/30, 'MANGO', va='bottom', color=mango_station[2], transform=ax.transAxes, fontsize=20)

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