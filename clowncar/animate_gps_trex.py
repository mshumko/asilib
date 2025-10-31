"""
Hi Kareem! Since the meat of my code is at the bottom, feel free to read bottom up! .·°՞(> ᗜ <)՞°·.
"""
import pathlib
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import aacgmv2
import cartopy.crs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates
import asilib
from asilib.asi.trex import trex_rgb
import IRBEM
import cdasws
import fontawesome as fa
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.path import Path

import clowncar

R_E = 6378.137  # km

def getmarker(mID):
	symbol = fa.icons[mID]
	fp = FontProperties(fname=pathlib.Path(__file__).parent / "fontawesome_solid.otf")

	v, codes = TextToPath().get_text_path(fp, symbol)
	v = np.array(v)
	mean = np.mean([np.max(v,axis=0), np.min(v, axis=0)], axis=0)
	return Path(v-mean, codes, closed=False)


def gps_footprint(gps_dict, alt=110, hemi_flag=0):
    """
    Map from 
    """
    if 'mag_model' not in globals():
        # Initialize the magnetic field model.
        # This is a global variable so that it can be reused in multiple calls.
        global mag_model
        mag_model = IRBEM.MagFields(kext='None')

    _all = np.zeros((len(gps_dict['interpolated_times']), 3), dtype=float)
    time_loc = pd.DataFrame(data={
        'time':gps_dict['interpolated_times'], 
        'x1':(gps_dict['Rad_Re']-1)*R_E,
        'x2':gps_dict['Geographic_Latitude'], 
        'x3':gps_dict['Geographic_Longitude'],
        })
    # kp = _get_kp(gps_dict['interpolated_times'])
    
    z = zip(
        time_loc['time'],
        time_loc['x1'],
        time_loc['x2'],
        time_loc['x3'],
    )
    for i, (time, x1, x2, x3) in enumerate(z):
        X = {'Time':time, 'x1':x1, 'x2':x2, 'x3':x3}
        _all[i, :] = mag_model.find_foot_point(X, {}, alt, hemi_flag)['XFOOT']
    _all[_all == -1E31] = np.nan
    # Convert from (alt, lat, lon) to Mike's fav (lat, lon, alt) LLA format.
    return np.roll(_all, shift=-1, axis=1)


def _get_kp(times):
    """
    Load (and optionally download) the Kp index and resample it to times.
    """
    cdas = cdasws.CdasWs()
    time_range = cdasws.TimeInterval(
        datetime.fromisoformat(str(times[0])).replace(tzinfo=timezone.utc), 
        datetime.fromisoformat(str(times[-1])).replace(tzinfo=timezone.utc)
        )
    # Try to get Kp a few times.
    for _ in range(5):
        try:
            _, data = cdas.get_data(
                'OMNI2_H0_MRG1HR', ['KP1800'], time_range
                )
        except UnboundLocalError:
            time.sleep(1)
            continue
        break
    if 'data' not in locals():
        raise ConnectionError
    
    kp = pd.DataFrame(index=data['KP'].Epoch.data, data={'Kp':data['KP']})
    state_times = pd.DataFrame(index=times)
    state_times = pd.merge_asof(
        state_times, kp, left_index=True, right_index=True, 
        tolerance=pd.Timedelta('1H'), direction='backward'
        )
    return state_times.Kp.to_numpy()

time_range = (datetime(2021, 11, 4, 6, 30), datetime(2021, 11, 4, 7, 30))
location_codes = ['LUCK', 'RABB', 'PINA', 'GILL']
center=(-100, 54)
alt=110
sat_height_m = 10_000_000
gps_energy_mev = 1  # [0.12, 0.21, 0.3, 0.425, 0.6, 0.8, 1.0, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
color_bounds = (1E4, 1E7)

# marker_cmap = plt.cm.viridis
# marker_cmap.set_bad('black')
marker_cmap = plt.cm.turbo
marker_cmap.set_bad(color='white', alpha=0.5)

asi_list = []
projection = cartopy.crs.NearsidePerspective(
    central_longitude=center[0], 
    central_latitude=center[1], 
    satellite_height=sat_height_m
)
fig = plt.figure(figsize=(9, 10))
ax = fig.add_subplot(111, projection=projection)
ax.add_feature(cfeature.LAND, color='grey')
ax.add_feature(cfeature.OCEAN, color='w')
ax.add_feature(cfeature.COASTLINE, edgecolor='k')
# ax.gridlines(linestyle=':')
# ax.set_global()
ax.set_extent(
    (center[0]-30, center[0]+30, center[1]-20, center[1]+20), 
    crs=cartopy.crs.PlateCarree()
    )


plt.tight_layout()

# AACGM Grid
lat_bounds=(0, 90)
lon_bounds=(-150, 0)
lat_grid, lon_grid = np.meshgrid(np.linspace(*lat_bounds), np.linspace(*lon_bounds, num=51))
# Need to pass flattened arrays since aacgmv2 does not work with n-D arrays.
aacgm_lat_grid, aacgm_lon_grid, _ = aacgmv2.wrapper.convert_latlon_arr(
    lat_grid.flatten(), lon_grid.flatten(), 110, time_range[0], method_code='G2A'
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
ax.clabel(cs, inline=True, fontsize=20, fmt=lambda x: f'$\lambda = {{{round(x)}}}^{{\circ}}$')

for location_code in location_codes:
    asi_list.append(trex_rgb(location_code, time_range=time_range, colors='rgb', acknowledge=False, alt=alt))

asis = asilib.Imagers(asi_list)
gen = asis.animate_map_gen(
    ax=ax, 
    asi_label=False, 
    ffmpeg_params={'framerate':100},
    overwrite=True
    )

interp_times = pd.date_range(*time_range, freq='3S')
interp_times_numeric = matplotlib.dates.date2num(interp_times)

gps_data = clowncar.GPS((time_range[0]-timedelta(hours=1), time_range[1]+timedelta(hours=1)))
for sc_key in gps_data:
    # Jumps across the anti-meridian or poles.
    lon_jumps = np.where(np.abs(np.diff(gps_data[sc_key]['Geographic_Longitude'])) > 5)[0]
    lon_jump_start_times = gps_data[sc_key]['time'][lon_jumps]
    lon_jump_end_times = gps_data[sc_key]['time'][lon_jumps+1]

    gps_data[sc_key]['interpolated_times'] = interp_times

    interpolated_jump_indices = np.array([])
    for start_time, end_time in zip(lon_jump_start_times, lon_jump_end_times):
        idt = np.where(
            (gps_data[sc_key]['interpolated_times'] >= start_time) &
            (gps_data[sc_key]['interpolated_times'] <= end_time)
            )[0]
        interpolated_jump_indices = np.concatenate((interpolated_jump_indices, idt))
    for footprint_key in ['Geographic_Latitude', 'Geographic_Longitude', 'Rad_Re']:
        gps_data[sc_key][footprint_key] = np.interp(
            interp_times_numeric,
            matplotlib.dates.date2num(gps_data[sc_key]['time']),
            gps_data[sc_key][footprint_key],
            left=np.nan,
            right=np.nan,
        )
        if interpolated_jump_indices.shape[0] > 0:
            gps_data[sc_key][footprint_key][interpolated_jump_indices] = np.nan
    
    footprint_lla = gps_footprint(
        gps_data[sc_key], 
        alt=alt, 
        # hemi_flag=1 for northern hemisphere footprint, 0 for same magnetic hemisphere footprint.
        hemi_flag=1
        )
    gps_data[sc_key]['footprint_lat'] = footprint_lla[:, 0]
    gps_data[sc_key]['footprint_lon'] = footprint_lla[:, 1]
    gps_data[sc_key]['footprint_alt'] = footprint_lla[:, 2]

    assert np.all(
            gps_data[sc_key]['electron_diff_flux_energy'][0, :]==\
            gps_data[sc_key]['electron_diff_flux_energy'][-1, :]
            ), 'Energy channels are not the same for all times.'
        
    
    energy_idx = np.where(gps_energy_mev==gps_data[sc_key]['electron_diff_flux_energy'][-1, :])[0]
    assert len(energy_idx) == 1, \
        (f'Energy channels {gps_data[sc_key]["electron_diff_flux_energy"][-1, :]} '
        f'do not match the expected {gps_energy_mev}.')
    gps_data[sc_key]['energy_channel_idx'] = energy_idx[0]

ax_extent = ax.get_extent(crs=cartopy.crs.PlateCarree())

gps_units_not_in_view = []
for sc_key, data in gps_data.items():
    idx = np.where(
        (ax_extent[0] < data['footprint_lon']) &
        (data['footprint_lon'] < ax_extent[1]) &
        (ax_extent[2] < data['footprint_lat']) & 
        (data['footprint_lat'] < ax_extent[3])
    )[0]
    if idx.shape[0] == 0:
        gps_units_not_in_view.append(sc_key)

for sc_key in gps_units_not_in_view:
    gps_data.pop(sc_key)

if color_bounds is None:
    # Calculate the min/max fluxes.
    for i, (sc_key, data) in enumerate(gps_data.items()):
        _channel_idx = gps_data[sc_key]['energy_channel_idx'] 
        _flux = gps_data[sc_key]['electron_diff_flux'][:, _channel_idx]
        _flux[_flux==-1] = np.nan
        if i == 0:
            color_bounds = [np.nanmin(_flux), np.nanmax(_flux)]
        else:
            if np.nanmin(_flux) < color_bounds[0]:
                color_bounds[0] = np.nanmin(_flux)
            if np.nanmax(_flux) > color_bounds[1]:
                color_bounds[1] = np.nanmax(_flux)

for i, (_guide_time, _, _, _)  in enumerate(gen):
    if i == 0:
        for sc_key, data in gps_data.items():
            ax.plot(
                data['footprint_lon'], 
                data['footprint_lat'], 
                c='r',
                ls=':',
                transform=cartopy.crs.PlateCarree()
                )
    
    if '_plot_time' in locals():
        _plot_time.remove()  # noqa: F821
    if ('gps_locs' in locals()) and len(gps_locs) > 0:  # noqa: F821
        for gps_loc in gps_locs:  # noqa: F821
            gps_loc.remove()
        for gps_label in gps_labels:
            gps_label.remove()
    gps_locs = []
    gps_labels = []
    for key, data in gps_data.items():
        dt_idt = np.argmin(np.abs((pd.to_datetime(data['interpolated_times'])-_guide_time).total_seconds()))
        dt_idt_flux = np.argmin(np.abs((pd.to_datetime(data['time'])-_guide_time).total_seconds()))

        if (
            (np.abs((data['interpolated_times'][dt_idt]-_guide_time).total_seconds()) < 5*60) and
            np.isfinite(data['footprint_lat'][dt_idt])
            ):
            _channel_idx = gps_data[key]['energy_channel_idx'] 
            _flux = gps_data[key]['electron_diff_flux'][dt_idt_flux, _channel_idx]
            scat = gps_locs.append(ax.scatter(
                data['footprint_lon'][dt_idt],
                data['footprint_lat'][dt_idt],
                1_500,
                c=_flux,
                cmap=marker_cmap,
                norm=matplotlib.colors.LogNorm(*color_bounds),
                marker=getmarker('satellite'),  # use snowflake for POES
                edgecolors="none",
                transform=cartopy.crs.PlateCarree(),
                ))
            gps_labels.append(ax.text(
                data['footprint_lon'][dt_idt]+1, 
                data['footprint_lat'][dt_idt],
                key,
                fontsize=30,
                color='orange',
                transform=cartopy.crs.PlateCarree()
            ))
    
    if i == 0:
        if gps_data[key].attrs['electron_diff_flux']['UNITS'] == 'cm^-2sec^-1sr^-1MeV^-1':
            label=f'{gps_energy_mev} MeV Electron flux [$(cm^{{2}} \ s \ sr \ MeV)^{{-1}}$]'
        else:
            label=f"{gps_energy_mev} MeV [{gps_data[key].attrs['electron_diff_flux']['UNITS']}]"
        cbar = plt.colorbar(gps_locs[0], ax=ax, orientation='horizontal', pad=0.01, label=label)            
        cbar.set_label(label=label, size=20)

    _plot_time = ax.text(
        0.01, 0.98, f'TREX-RGB\n{_guide_time.strftime("%Y-%m-%d %H:%M:%S")}', 
        fontsize=20, transform=ax.transAxes, ha='left', va='top'
        )