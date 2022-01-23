from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import asilib

# time_range = [datetime(2007, 1, 20, 0, 0), datetime(2007, 1, 20, 1, 0)]
asi_array_code = 'REGO'
location_code = 'LUCK'
map_alt = 110

# time_range = (datetime(2017, 9, 15, 2, 37, 0), datetime(2017, 9, 15, 3, 0, 0))
time_range = ['2017-09-27T07', '2017-09-27T09']

# Set up the custom path along the meridian.
skymap = asilib.load_skymap(asi_array_code, location_code, time_range[0])
alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]

image_resolution = skymap['FULL_MAP_LATITUDE'].shape[1:]
latlon = np.column_stack((
                skymap['FULL_MAP_LATITUDE'][alt_index, :, image_resolution[0]//2],
                skymap['FULL_MAP_LONGITUDE'][alt_index, :, image_resolution[0]//2]
                ))
latlon = latlon[np.where(~np.isnan(latlon[:,0]))[0], :]
dl = latlon[0, 1:] - latlon[0, :-1]
latlon[:, 0] += dl

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
# asilib.plot_keogram(asi_array_code, location_code, time_range, map_alt, ax=ax[0])
asilib.plot_keogram(asi_array_code, location_code, time_range, map_alt, 
                                path=latlon, ax=ax[1])
plt.show()