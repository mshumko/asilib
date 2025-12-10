"""
The clowncar module combines satellite, ground-based magnetometer, and auroral all-sky imager 
data into a visualization library. The main class is Clowncar which conducts the hetereogeneous
data orchistra and makes plots & animations. 
"""

import pathlib
import pprint
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import aacgmv2
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cartopy_imported = True
except ImportError as err:
    # You can also get a ModuleNotFoundError if cartopy is not installed
    # (as compared to failed to import), but it is a subclass of ImportError.
    cartopy_imported = False
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates
import asilib
import asilib.map
import IRBEM
import cdasws
import fontawesome as fa

R_E = 6378.137  # km

class Clowncar:
    def __init__(self, asi, observatories, ax=None, ax_kwargs={}):
        self.asi = asi
        self.observatories = observatories
        if not isinstance(self.observatories, (list, tuple)):
            self.observatories = (self.observatories,)
        self.ax = ax
        self._init_map(ax_kwargs)

        self.default_cmap = plt.cm.viridis
        self.default_cmap.set_bad(color='white', alpha=0.5)
        return
    
    def _init_map(self, ax_kwargs):

        center = ax_kwargs.get('center', (-100, 54))
        lon_bounds = ax_kwargs.get('lon_bounds', (center[0]-30, center[0]+30))
        lat_bounds = ax_kwargs.get('lat_bounds', (center[1]-20, center[1]+20))
        
        if cartopy_imported and (self.ax is None):
            projection = ccrs.NearsidePerspective(
                central_longitude=center[0], 
                central_latitude=center[1], 
                satellite_height=ax_kwargs.get('sat_height_m', 10_000_000) 
            )
            fig = plt.figure(figsize=(9, 10))
            self.ax = fig.add_subplot(ax_kwargs.get('position', 111), projection=projection)
            self.ax.add_feature(cfeature.LAND, color='grey')
            self.ax.add_feature(cfeature.OCEAN, color='w')
            self.ax.add_feature(cfeature.COASTLINE, edgecolor='k')
            self.ax.set_extent(
                (*lon_bounds, *lat_bounds), 
                crs=ccrs.PlateCarree()
                )
        elif (not cartopy_imported) and (self.ax is None):
            fig = plt.figure(figsize=(9, 10))
            self.ax = asilib.map.create_map(lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        else:
            pass
        
        if cartopy_imported:
            self._transform = ccrs.PlateCarree()
        else:
            self._transform = self.ax.transData

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
        if ax_kwargs.get('aacgm_grid', True):
            self._plot_aacgm_grid(self.ax, time=_time, alt=alt)
        return
    
    def animate_map(self, framerate=100):
        gen = self.asi.animate_map_gen(
            ax=self.ax, 
            asi_label=False, 
            ffmpeg_params={'framerate':framerate},
            overwrite=True,
            timestamp=False
            )
        for i, (_guide_time, _, _, _)  in enumerate(gen):
            if i == 0:
                for observatory in self.observatories:
                    if hasattr(observatory, '_cc_footprint_params'):
                        for _footprint in observatory.footprints.values():
                            self.ax.plot(
                                _footprint['lon'], 
                                _footprint['lat'], 
                                transform=self._transform,
                                **observatory._cc_footprint_params
                                )
            
            # if '_plot_time' in locals():
            #     _plot_time.remove()  # noqa: F821
            if ('observatory_markers' in locals()) and len(observatory_markers) > 0:  # noqa: F821
                for observatory_marker in observatory_markers:  # noqa: F821
                    observatory_marker.remove()
                for obs_label in obs_labels:  # noqa: F821
                    obs_label.remove()
            observatory_markers = []
            obs_labels = []
            for observatory in self.observatories:
                observatory_data = observatory(_guide_time, ax=self.ax)
                if observatory_data == {}:
                    Warning(f"No {observatory.__class__.__name__} data returned for time {_guide_time}.")
                    continue
                for sc_id, _lon, _lat, _flux in zip(
                    observatory_data['sc_id'],
                    observatory_data['footprint_lon'], 
                    observatory_data['footprint_lat'], 
                    observatory_data['flux']
                    ):
                    marker = observatory._cc_marker_params.get('marker', 'o')
                    if (isinstance(marker, str)) and (marker.split('-')[0] == 'fontawesome'):
                        marker = self._get_fontawesome_marker(marker.split('-')[1])

                    if np.isfinite(_lat) and np.isfinite(_lon):
                        print(f"Plotting {observatory.__class__.__name__} {_lon:.2f}, {_lat:.2f}, flux={_flux:.2e}")
                        observatory_markers.append(
                            self.ax.scatter(
                                _lon,
                                _lat,
                                c=_flux,
                                s=observatory._cc_marker_params.get('s', 1_500),
                                cmap=observatory._cc_marker_params.get('cmap', self.default_cmap),
                                norm=observatory._cc_marker_params.get('norm', matplotlib.colors.Normalize),
                                marker=marker,
                                edgecolors=observatory._cc_marker_params.get('edgecolors', None),
                                transform=self._transform,
                            ))
                    
                        if hasattr(observatory, '_cc_marker_label_params'):
                            obs_labels.append(self.ax.text(
                                _lon+observatory._cc_marker_label_params.get('lon_offset', 1),
                                _lat+observatory._cc_marker_label_params.get('lat_offset', 0),
                                sc_id,
                                fontsize=observatory._cc_marker_label_params.get('fontsize', 20),
                                color=observatory._cc_marker_label_params.get('color', 'orange'),
                                transform=self._transform
                            ))
                    else:
                        print(f"Skipping {observatory.__class__.__name__}-{sc_id} footprint plot for invalid lat/lon: {_lat}, {_lon}")
            
            # if i == 0:
            #     if gps_data[key].attrs['electron_diff_flux']['UNITS'] == 'cm^-2sec^-1sr^-1MeV^-1':
            #         label=fr'{gps_energy_mev} MeV Electron flux [$(cm^{{2}} \ s \ sr \ MeV)^{{-1}}$]'
            #     else:
            #         label=f"{gps_energy_mev} MeV [{gps_data[key].attrs['electron_diff_flux']['UNITS']}]"
                # cbar = plt.colorbar(observatory_markers[0], ax=ax, orientation='horizontal', pad=0.01, label=label)            
                # cbar.set_label(label=label, size=20)

            # _plot_time = ax.text(
            #     0.01, 0.98, f'TREX-RGB\n{_guide_time.strftime("%Y-%m-%d %H:%M:%S")}', 
            #     fontsize=20, transform=ax.transAxes, ha='left', va='top'
            #     )
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
            transform=self._transform
            )
        ax.contour(
            lon_grid, 
            lat_grid, 
            aacgm_lon_grid,
            # levels=np.arange(-50, -150, 15), 
            colors='k',
            linestyles='dashed',
            alpha=0.5, 
            transform=self._transform
            )
        ax.clabel(cs, inline=True, fontsize=20, fmt=lambda x: f'$\\lambda = {{{round(x)}}}^{{\\circ}}$')
        return lat_grid, lon_grid, aacgm_lat_grid, aacgm_lon_grid, cs
    
    def _get_fontawesome_marker(self, marker_nam='satellite'):
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

    from gps import GPS

    time_range = (datetime(2021, 11, 4, 6, 30), datetime(2021, 11, 4, 7, 30))
    location_codes = ['LUCK', 'RABB', 'PINA', 'GILL']
    center=(-100, 54)
    alt=110

    asi_list = []
    for location_code in location_codes:
        asi_list.append(trex_rgb(location_code, time_range=time_range, colors='rgb', acknowledge=False, alt=alt))
    asis = asilib.Imagers(asi_list)

    L_range = [4.25, 4.75]
    dt_min=60*3
    # file_paths, spacecraft_ids = download_gps(date, version='1.10', redownload=False)
    _gps = GPS(time_range, version='1.10', redownload=False)
    _gps.interpolate_gps_loc(freq='3s')
    _gps.gps_footprint(alt=110, hemi_flag=1)
    _gps.cc_footprint_config()
    _gps.cc_marker_config()
    _gps.cc_marker_label_config()

    cc = Clowncar(asis, _gps)
    cc.animate_map(framerate=30)
    pass