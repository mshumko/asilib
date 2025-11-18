"""
The clowncar module combines satellite, ground-based magnetometer, and auroral all-sky imager 
data into a visualization library. The main class is Clowncar which conducts the hetereogeneous
data orchistra and makes plots & animations. 
"""

import pathlib
# import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import aacgmv2
import cartopy.crs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates
import asilib
import IRBEM
import cdasws
import fontawesome as fa

import clowncar

R_E = 6378.137  # km

class Clowncar:
    def __init__(self, asi, observatories, ax=None, ax_kwargs={}):
        self.asi = asi
        self.observatories = observatories
        self.ax = ax
        if self.ax is None:
            self._init_map(ax_kwargs)
        return
    
    def _init_map(self, fig, ax_kwargs, aacgm_grid=True):

        projection = cartopy.crs.NearsidePerspective(
            central_longitude=center[0], 
            central_latitude=center[1], 
            satellite_height=ax_kwargs.get('sat_height_m', 10_000_000) 
        )
        fig = plt.figure(figsize=(9, 10))
        ax = fig.add_subplot(ax_kwargs.get('position', 111), projection=projection)
        ax.add_feature(cfeature.LAND, color='grey')
        ax.add_feature(cfeature.OCEAN, color='w')
        ax.add_feature(cfeature.COASTLINE, edgecolor='k')
        ax.set_extent(
            (center[0]-30, center[0]+30, center[1]-20, center[1]+20), 
            crs=cartopy.crs.PlateCarree()
            )
        if hasattr(self.asi, 'file_info'):
            if 'time_range' in self.asi.file_info:
                _time = self.asi.file_info['time_range'][0]
            else:
                _time = self.asi.file_info['time']
        else:
            if 'time_range' in self.asi.imagers[0].file_info:
                _time = self.asi.imagers[0].file_info['time_range'][0]
            else:
                _time = self.asi.imagers[0].file_info['time']
        alt = ax_kwargs.get('alt_km', 110)
        if aacgm_grid:
            self._plot_aacgm_grid(ax, time=_time, alt=alt)
        return

    def _plot_aacgm_grid(self, ax, time, lat_bounds=(0, 90), lon_bounds=(-150, 0), grid_res=51, alt=110):
        lat_grid, lon_grid = np.meshgrid(np.linspace(*lat_bounds, num=grid_res), np.linspace(*lon_bounds, num=grid_res))
        # Need to pass flattened arrays since aacgmv2 does not work with n-D arrays.
        aacgm_lat_grid, aacgm_lon_grid, _ = aacgmv2.wrapper.convert_latlon_arr(
            lat_grid.flatten(), lon_grid.flatten(), alt, time, method_code='G2A'
            )
        aacgm_lat_grid = aacgm_lat_grid.reshape(lat_grid.shape)
        aacgm_lon_grid = aacgm_lon_grid.reshape(lon_grid.shape)

        cs = ax.contour(
            lon_grid, 
            lat_grid, 
            aacgm_lat_grid,
            levels=np.arange(40, 91, 5), 
            colors='k',
            linestyles='dashed',
            linewidths=2,
            alpha=0.5, 
            transform=cartopy.crs.PlateCarree()
            )
        ax.contour(
            lon_grid, 
            lat_grid, 
            aacgm_lon_grid,
            # levels=np.arange(-50, -150, 15), 
            colors='k',
            linestyles='dashed',
            alpha=0.5, 
            transform=cartopy.crs.PlateCarree()
            )
        ax.clabel(cs, inline=True, fontsize=20, fmt=lambda x: f'$\\lambda = {{{round(x)}}}^{{\\circ}}$')
        return lat_grid, lon_grid, aacgm_lat_grid, aacgm_lon_grid, cs
    
    def _getmarker(self, marker_nam='satellite'):
        symbol = fa.icons[marker_nam]
        fp = matplotlib.font_manager.FontProperties(
            fname=pathlib.Path(asilib.__file__).parents[1] / 'asilib' / 'data' / "Font Awesome 7 Free-Solid-900.otf"
            )

        v, codes = matplotlib.textpath.TextToPath().get_text_path(fp, symbol)
        v = np.array(v)
        mean = np.mean([np.max(v,axis=0), np.min(v, axis=0)], axis=0)
        return matplotlib.path.Path(v-mean, codes, closed=False)
    

if __name__ == "__main__":
    from datetime import datetime
    from asilib.asi import trex_rgb

    time_range = (datetime(2021, 11, 4, 6, 30), datetime(2021, 11, 4, 7, 30))
    location_codes = ['LUCK', 'RABB', 'PINA', 'GILL']
    center=(-100, 54)
    alt=110

    asi_list = []
    for location_code in location_codes:
        asi_list.append(trex_rgb(location_code, time_range=time_range, colors='rgb', acknowledge=False, alt=alt))
    asis = asilib.Imagers(asi_list)

    cc = Clowncar(asis, None)
    print(cc.getmarker())
    pass