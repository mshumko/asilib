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
from asilib.imager import Skymap_Cleaner

asi_arrays = {
    'THEMIS':(asilib.asi.themis, asilib.asi.themis_info, 'k', 10),
    'REGO':(asilib.asi.rego, asilib.asi.rego_info, 'blue', 10),
    'TREx-NIR':(asilib.asi.trex_nir, asilib.asi.trex_nir_info, 'c', 12),
    'TREx-RGB':(asilib.asi.trex_rgb, asilib.asi.trex_rgb_info, 'purple', 15),
    }

mango_sations = [
    ['CVO', datetime(2024, 10, 25, 5, 28), 'r'],
    ['CFS', datetime(2024, 10, 25, 5, 28), 'r'],
    ['MTO', datetime(2024, 10, 25, 5, 28), 'r'],
    ['EIO', datetime(2024, 10, 25, 5, 28), 'r'],
    ['PAR', datetime(2024, 10, 25, 5, 28), 'r'],
    ['MDK', datetime(2023, 10, 14, 6, 44), 'r']
]

time = datetime(2022, 1, 1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection=ccrs.Orthographic(-100, 50))

ax.add_feature(cartopy.feature .LAND, color='g')
ax.add_feature(cartopy.feature .OCEAN, color='w')
ax.add_feature(cartopy.feature .COASTLINE, edgecolor='k')
ax.gridlines()

plotted_locations = []

for mango_station in mango_sations:
    asi = asilib.asi.mango(mango_station[0], 'redline', time=mango_station[1])
    ax.contour(
        asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'], levels=[16],
        transform=ccrs.PlateCarree(), colors=mango_station[2]
        )
    if mango_station[0] not in plotted_locations:
        ax.text(
            asi.meta['lon'],
            asi.meta['lat'],
            asi.meta['location'].upper(),
            color='k',
            transform = ccrs.PlateCarree(),
            va='center',
            ha='center',
        )
        plotted_locations.append(mango_station[0])

ax.text(
    0, 4/30, 'MANGO-redline', va='bottom', color=mango_station[2], 
    transform=ax.transAxes, fontsize=18
    )

for i, (array, (loader, info_df, color, elevation)) in enumerate(asi_arrays.items()):
    asi_array_info = info_df()
    for location in asi_array_info['location_code']:
        try:
            asi = loader(location, time=time, load_images=False)
        except requests.exceptions.HTTPError as err:
            if ('Not Found for url' in str(err)) or ('Precondition Failed for url' in str(err)):
                print(err)
                continue
            else:
                raise
        _skymap_cleaner = Skymap_Cleaner(
                asi.skymap['lon'], 
                asi.skymap['lat'], 
                asi.skymap['el'],
            )
        _lon, _lat = _skymap_cleaner.remove_nans()
        ax.contour(
            _lon[:-1, :-1], _lat[:-1, :-1], asi.skymap['el'], levels=[elevation],
            transform=ccrs.PlateCarree(), colors=color
        )
        if location not in plotted_locations:
            ax.text(
                asi.meta['lon'],
                asi.meta['lat'],
                asi.meta['location'].upper(),
                color='k',
                transform = ccrs.PlateCarree(),
                va='center',
                ha='center',
            )
            plotted_locations.append(location)
    ax.text(0, i/30, array, va='bottom', color=color, transform=ax.transAxes, fontsize=18)

ax.set_title(f'Imagers Supported by asilib as of {datetime.now().date()}', fontsize=22)        
plt.tight_layout()
plt.show()