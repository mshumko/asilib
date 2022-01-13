# Make a keogram of a proton aurora observed by The Pas along a custom path 
# (the Gillam meridional scanning photometer).
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import asilib

time = datetime(2007, 1, 20, 0, 40)
asi_array_code = 'THEMIS'
location_code = 'TPAS'
map_alt = 110
lon_bounds = (-110, -90)
lat_bounds = (60, 49)

msp_url = ('https://github.com/mshumko/aurora-asi-lib/raw/'
    '73ef9bd5220b781436aea3281c70da0f5b08ac05/asilib/data/GILL_MSP_coords.csv')
msp_df = pd.read_csv(msp_url)
msp_df.columns = [column.split('_')[1] for column in msp_df.columns]
print(msp_df.head())

fig = plt.figure(figsize=(10, 6))
ax = asilib.create_cartopy_map(lon_bounds=lon_bounds, lat_bounds=lat_bounds,
    fig_ax={'fig':fig, 'ax':121}
    ) 
asilib.plot_map(asi_array_code, location_code, time, map_alt, ax=ax)
s = ax.plot(msp_df.loc[:, 'glon'], msp_df.loc[:, 'glat'], c='r',
    transform=ccrs.PlateCarree())

# n_mlat_labels = 10
# mlat_indices = np.arange(0, msp_df.shape[0]+1, msp_df.shape[0]//n_mlat_labels).astype(int)
# mlat_indices[-1] -= 1

# for mlat_index in mlat_indices:
#     if ((msp_df.loc[mlat_index, 'glat'] < max(lat_bounds)) and
#         (msp_df.loc[mlat_index, 'glat'] > min(lat_bounds))):
        
#         ax.text(msp_df.loc[mlat_index, 'glon'], msp_df.loc[mlat_index, 'glat'], 
#             f'<- $\lambda$={round(msp_df.loc[mlat_index, "glat"], 1)}', 
#             transform=ccrs.PlateCarree(), color='red', va='center')

plt.tight_layout()
plt.show()