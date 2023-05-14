# Make a keogram of a proton aurora observed by The Pas along a custom path
# (the Gillam meridional scanning photometer).
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates

import asilib

time = datetime(2007, 1, 20, 0, 40)
time_range = [datetime(2007, 1, 20, 0, 0), datetime(2007, 1, 20, 2, 0)]
asi_array_code = 'THEMIS'
location_code = 'TPAS'
map_alt = 110
lon_bounds = (-110, -90)
lat_bounds = (62, 47)

msp_url = (
    'https://github.com/mshumko/aurora-asi-lib/raw/'
    '73ef9bd5220b781436aea3281c70da0f5b08ac05/asilib/data/GILL_MSP_coords.csv'
)
msp_df = pd.read_csv(msp_url)
msp_df.columns = [column.split('_')[1] for column in msp_df.columns]
# Convert longitudes (0 -> 360) to (-180 -> 180)
msp_df['glon'] = np.mod(msp_df['glon'] + 180, 360) - 180
print(msp_df.head())

# Create the map and keogram subplots
fig = plt.figure(figsize=(14, 4))
ax = asilib.make_map(lon_bounds=lon_bounds, lat_bounds=lat_bounds, ax=fig.add_subplot(131))
bx = fig.add_subplot(132)
cx = fig.add_subplot(133, sharey=bx, sharex=bx)

# Plot the mapped image from one time stamp.
asilib.plot_map(asi_array_code, location_code, time, map_alt, ax=ax)
ax.set_title(f'{asi_array_code}-{location_code} {time}')

# Plot the MSP field of view.
s = ax.plot(msp_df.loc[:, 'glon'], msp_df.loc[:, 'glat'], c='r')

# Plot the THEMIS-TPAS meridian
skymap = asilib.load_skymap(asi_array_code, location_code, time)
alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
keogram_latitude = skymap['FULL_MAP_LATITUDE'][
    alt_index, :, skymap['FULL_MAP_LATITUDE'].shape[1] // 2
]
keogram_longitude = skymap['FULL_MAP_LONGITUDE'][
    alt_index, :, skymap['FULL_MAP_LATITUDE'].shape[1] // 2
]
ax.plot(keogram_longitude, keogram_latitude, c='c')

# Plot the keogram along the meridian
asilib.plot_keogram(asi_array_code, location_code, time_range, map_alt, ax=bx)
bx.text(
    0,
    1,
    f'Keogram along zenith (cyan line in left panel).',
    transform=bx.transAxes,
    va='top',
    c='k',
)
# Plot the keogram along the MSP field of view
asilib.plot_keogram(
    asi_array_code,
    location_code,
    time_range,
    map_alt,
    ax=cx,
    path=msp_df.loc[:, ['glat', 'glon']].to_numpy(),
)

cx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

n_mlat_labels = 7
mlat_indices = np.arange(0, msp_df.shape[0] + 1, msp_df.shape[0] // n_mlat_labels).astype(int)
mlat_indices[-1] -= 1

for mlat_index in mlat_indices:
    if (msp_df.loc[mlat_index, 'glat'] < max(lat_bounds)) and (
        msp_df.loc[mlat_index, 'glat'] > min(lat_bounds)
    ):
        ax.text(
            msp_df.loc[mlat_index, 'glon'],
            msp_df.loc[mlat_index, 'glat'],
            f'<- $\lambda$={round(msp_df.loc[mlat_index, "mlat"], 1)}',
            color='red',
            va='center',
        )
cx.text(
    0,
    1,
    f'Keogram along GILL MSP (red line in left panel).',
    transform=cx.transAxes,
    va='top',
    c='w',
)
plt.tight_layout()
plt.show()
