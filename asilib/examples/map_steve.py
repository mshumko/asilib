"""
Maps an image of STEVE (the thin band). Reproduced from 
http://themis.igpp.ucla.edu/nuggets/nuggets_2018/Gallardo-Lacourt/fig2.jpg

Note that cartopy seems to hang sometimes the first time it runs this script. 
Give it time and it should load.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import asilib

# Create a custom map subplot from a satellite's perspective.
sat_lat = 54
sat_lon = -100
sat_altitude_km = 500

fig = plt.figure(figsize=(10, 7))
projection = ccrs.NearsidePerspective(sat_lon, sat_lat, satellite_height=1000*sat_altitude_km)
ax = fig.add_subplot(1, 1, 1, projection=projection)
ax.add_feature(cfeature.LAND, color='green')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(linestyle=':')

image_time, image, skymap, ax, p = asilib.plot_map(
        'THEMIS', 'ATHA', datetime(2010, 4, 5, 6, 7, 0), 110, ax=ax
    )
plt.show()