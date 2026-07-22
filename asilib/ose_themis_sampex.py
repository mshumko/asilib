"""
Observational System Experiment (OSE) animation using SAMPEX and THEMIS ASI datasets.
"""
import dataclasses
import copy
from datetime import datetime
import string
from typing import Tuple, List, Union
import dateutil.parser

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates
import matplotlib.patches
import matplotlib.gridspec as gridspec
import scipy.interpolate
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import asilib
import asilib.map
import asilib.asi
import sampex
import cartopy.crs as ccrs
import manylabels
import pymap3d.los
import IRBEM


R_e = 6378.137  # km


class SAMPEX_footprint:
    def __init__(self, time_range):
        """"
        Load SAMPEX attitude data and calculate its footprint.

        Parameters
        ----------
        day: datetime or str
        """
        if isinstance(time_range[0], str):
            day = dateutil.parser.parse(time_range[0])
        else:
            day = copy.copy(time_range[0])
        day = pd.Timestamp(day).replace(hour=0, minute=0, second=0, microsecond=0)
        self.attitude = sampex.Attitude(day).load()

        self.attitude = self.attitude.loc[
            (self.attitude.index >= time_range[0]) & 
            (self.attitude.index < time_range[1]),
            :]
        self.m = IRBEM.MagFields(kext='None')
        self.coords = IRBEM.Coords()
        return

    def map_down(self, alt=110, hemi_flag=0):
        """
        Map self.lla down along the magnetic field line to alt using IRBEM.MagFields.find_foot_print.

        Parameters
        ----------
        alt: float
            The mapping altitude in units of kilometers
        hemi_flag: int
            What direction to trace the field line: 
            0 = same magnetic hemisphere as starting point
            +1   = northern magnetic hemisphere
            -1   = southern magnetic hemisphere
            +2   = opposite magnetic hemisphere as starting point
        """
        _all = np.zeros_like(self.attitude.loc[:, ['Altitude', 'GEO_Lat', 'GEO_Long']])

        for i, (time, row) in enumerate(self.attitude.iterrows()):
            X = {'Time':time, 'x1':row['Altitude'], 'x2':row['GEO_Lat'], 'x3':row['GEO_Long']}
            _all[i, :] = self.m.find_foot_point(X, {}, alt, hemi_flag)['XFOOT']
        _all[_all == -1E31] = np.nan
        _attitude = self.attitude.copy()
        _attitude.loc[:, ['Altitude', 'GEO_Lat', 'GEO_Long']] = _all
        return _attitude
    
    def map_up(self, alt, sysout='GDZ'):
        """
        Map self.lla up along the magnetic field line to alt using IRBEM.MagFields.find_foot_print.

        Parameters
        ----------
        alt: float
            The mapping altitude in units of kilometers.
        sysout: str
            The output coordinate system. Default is 'GDZ' (Geodetic Zenith Direction).
            Other options include 'GEO', 'GSE', 'SM', etc.
        """
        lampsat_mapped_x = np.zeros((self.attitude.shape[0], 3))

        for i, (time, row) in enumerate(self.attitude.iterrows()):
            X = {'Time':time, 'x1':row['Altitude'], 'x2':row['GEO_Lat'], 'x3':row['GEO_Long']}
            _field_line = self.m.trace_field_line(X, {})
            _field_line_alt = R_e*(np.linalg.norm(_field_line['POSIT'], axis=1)-1)

            xGEO = _field_line['POSIT'][:_field_line['Nposit'], 0] 
            yGEO = _field_line['POSIT'][:_field_line['Nposit'], 1] 
            zGEO = _field_line['POSIT'][:_field_line['Nposit'], 2] 
            S = range(len(_field_line['blocal'][:_field_line['Nposit']]))

            # Interpolate the magnetic field, as well as GEO coordinates.
            f_alt_diff = scipy.interpolate.interp1d(S, _field_line_alt-alt, kind='cubic')
            fx = scipy.interpolate.interp1d(S, xGEO, kind='cubic')
            fy = scipy.interpolate.interp1d(S, yGEO, kind='cubic')
            fz = scipy.interpolate.interp1d(S, zGEO, kind='cubic')

            lampsat_loc_idx = scipy.optimize.brentq(f_alt_diff, 0, _field_line_alt.shape[0]/2)
            lampsat_mapped_x[i, :] = np.array([fx(lampsat_loc_idx), fy(lampsat_loc_idx), fz(lampsat_loc_idx)])
        if sysout != 'GEO':
            lampsat_mapped_x = self.coords.transform(self.attitude.index, lampsat_mapped_x, 'GEO', sysout)
        if sysout == 'GDZ':
            _attitude = self.attitude.copy()
            _attitude.loc[:, ['Altitude', 'GEO_Lat', 'GEO_Long']] = lampsat_mapped_x
            return _attitude
        return lampsat_mapped_x

  
class Ellipsoid_alt:
    """
    From https://geospace-code.github.io/pymap3d/ellipsoid.html

    as everywhere else in this program, distance units are METERS
    """

    def __init__(self, alt):
        """
        feel free to suggest additional ellipsoids

        Parameters
        ----------
        alt: float
            Altitude in meters. This is used to calculate the ellipsoid
            parameters.
        """
        self.semimajor_axis = 6378137.0+alt
        self.semiminor_axis = 6356752.31424518+alt

        self.flattening = (self.semimajor_axis - self.semiminor_axis) / self.semimajor_axis
        self.thirdflattening = (self.semimajor_axis - self.semiminor_axis) / (self.semimajor_axis + self.semiminor_axis)
        self.eccentricity = np.sqrt(2 * self.flattening - self.flattening ** 2)

@dataclasses.dataclass
class ASI_OSE_Animation:
    """
    Calculate the THEMIS ASI white-light intensity inside a space-based imager FOV.

    Parameters
    ----------
    time_range: Tuple[datetime]
        Defines the time range for the OSE plot.
    fov: Tuple[float]
        The field of view of the AIC in degrees.
    resolution: Tuple[int]
        The resolution of the AIC in pixels (width, height).
    ona: float
        The off-nadir angle between nadir and the imager's center FOV vectors. If 0, the center of
        the FOV is pointing towards the nadir and if 90 it points at the limb.
    azimuth: float
        The azimuth angle of the AIC FOV measured clockwise from north.
    lampsat_alt: float
        The altitude of the LAMPsat in kilometers.
    themis_location_code: str
        The THEMIS location code, e.g., 'WHIT' for THEMIS ASI.
    aurora_alt: float
        The altitude of the aurora in kilometers.
    """
    time_range:Tuple[datetime]
    fov:Tuple[float]=(45, 30)
    resolution:Tuple[int]=(64, 64)
    ona:float=0,
    azimuth:float=0,
    lampsat_alt:float=500
    themis_location_code:str='WHIT'
    aurora_alt:float=110
    checkerboard:bool=True
    lon_bounds:Tuple[float]=None
    lat_bounds:Tuple[float]=None
    color_bounds:Tuple[int]=None
    detrend_hilt:bool=False
    detrend_duration_s:float=5
    detrend_quantile:float=0.5
    hilt_logscale:bool=True
    hilt_ylim:tuple=(-20, 4*10**3)

    def __post_init__(self):
        self.xx, self.yy = np.meshgrid(
            np.linspace(-self.fov[1]/2, self.fov[1]/2, self.resolution[1]),
            np.linspace(-self.fov[0]/2, self.fov[0]/2, self.resolution[0]),
            )
        self.xx += np.sin(np.deg2rad(self.azimuth))*self.ona
        self.yy += np.cos(np.deg2rad(self.azimuth))*self.ona
        self.tilts = np.sqrt(self.xx**2 + self.yy**2)
        self.azs = np.rad2deg(np.arctan2(self.yy, self.xx))
        self._checkerboard = np.zeros((10, 10), dtype=bool)
        self._checkerboard[::2, ::2] = True
        self._checkerboard[1::2, 1::2] = True
        self._checkerboard_xx, self._checkerboard_yy = np.meshgrid(
            np.linspace(0, self.resolution[0], num=self._checkerboard.shape[0]+1),
            np.linspace(0, self.resolution[1], num=self._checkerboard.shape[1]+1)
            )

    
    def load_data(self):
        self.asi = asilib.asi.themis(self.themis_location_code, time_range=self.time_range, alt=self.aurora_alt)

        if self.color_bounds is None:
            self.color_bounds = self.asi.auto_color_bounds()

        footprint_obj = SAMPEX_footprint(self.time_range)
        low_alt_footprint = footprint_obj.map_down(alt=self.aurora_alt)
        high_alt_footprint = footprint_obj.map_up(alt=self.lampsat_alt, sysout='GDZ')

        self.hilt = sampex.HILT(self.time_range[0]).load()
        if self.detrend_hilt:
            N = int(self.detrend_duration_s/20E-3)
            trend = self.hilt.rolling(N, center=True).quantile(self.detrend_quantile)
            # Can't use "-" here due to memory allocation issues.
            self.hilt['counts'] = self.hilt.sub(trend)
        self.hilt = self.hilt.loc[self.time_range[0]:self.time_range[1], :]

        conjunction_obj = asilib.Conjunction(
            self.asi, 
            pd.DataFrame(
                index=low_alt_footprint.index, 
                data={
                    'Lat':low_alt_footprint['GEO_Lat'], 
                    'Lon':low_alt_footprint['GEO_Long'], 
                    'Alt':low_alt_footprint['Altitude']}
                ),
            )
        self.low_alt_footprint = conjunction_obj.interp_sat()
        self.asi_nearest_count_intensity, _ = conjunction_obj.intensity(box=None)

        # Apply the Gabrielse+2021 THEMIS ASI-> 557.7 nm intensity conversion.
        # https://doi.org/10.3389/fphy.2021.744298
        self.asi_557_intensity = np.nan*np.zeros_like(self.asi_nearest_count_intensity)
        self.asi_557_intensity[self.asi_nearest_count_intensity>600] = \
            10**(1.20+0.93*np.log10(self.asi_nearest_count_intensity[self.asi_nearest_count_intensity>600]))
        self.asi_557_intensity[self.asi_nearest_count_intensity<=600] = \
            10**(1.54+0.81*np.log10(self.asi_nearest_count_intensity[self.asi_nearest_count_intensity<=600]))

        conjunction_obj = asilib.Conjunction(
            self.asi, 
            pd.DataFrame(
                index=high_alt_footprint.index, 
                data={
                    'Lat':high_alt_footprint['GEO_Lat'], 
                    'Lon':high_alt_footprint['GEO_Long'], 
                    'Alt':high_alt_footprint['Altitude']}
                ),
            )
        self.high_alt_footprint = conjunction_obj.interp_sat()
        return
    
    def animate(self):
        fig = plt.figure(figsize=(9, 7))
        spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, height_ratios=(2, 1))
        self.ax = asilib.map.create_map(lon_bounds=lon_bounds, lat_bounds=lat_bounds, fig_ax=(fig, spec[0, 0]))
        self.bx = fig.add_subplot(spec[0, 1])
        self.cx = fig.add_subplot(spec[1, :])

        self.bx.xaxis.set_visible(False)
        self.bx.yaxis.set_visible(False)

        self.ax.plot(
            self.low_alt_footprint.loc[:, 'lon'], 
            self.low_alt_footprint.loc[:, 'lat'], 
            'r:', 
            transform=ccrs.PlateCarree()
            )

        self.cx.plot(self.hilt.index, self.hilt['counts'], c='r')
        self.cx.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=5))
        self.cx.set_xlim(*time_range)
        if self.hilt_logscale:
            self.cx.set_yscale('log')
        self.cx.set_ylim(*self.hilt_ylim)
        self.cx.set_ylabel(f'[counts/20 ms]')
        self.cx.xaxis.set_major_locator(matplotlib.dates.SecondLocator(interval=30))
        manylabels.ManyLabels(
            self.cx, 
            self.high_alt_footprint, 
            label_coord=(-0.08, -0.09)
            )
        
        plt.subplots_adjust(
            top=0.91,
            bottom=0.12,
            left=0.09,
            right=0.95,
            hspace=0.07,
            wspace=0.01
        )

        if self.detrend_hilt:
            _detrend = 'Detrended'
        else:
            _detrend = ''
        _aic_label = self.bx.text(
            0.01, 0.96, f'({string.ascii_lowercase[1]}) AIC FOV', va='center', 
            transform=self.bx.transAxes, weight='bold', fontsize=15
        )
        _aic_label.set_bbox(dict(facecolor='white', pad=0.25))

        self.cx.text(
            0, 0.99, f'({string.ascii_lowercase[2]}) {_detrend} SAMPEX-HILT >1 MeV electrons', va='top', 
            transform=self.cx.transAxes, weight='bold', fontsize=15
            )
        plt.suptitle(
                f'LAMPsat OSE | FOV={self.fov[0]}x{self.fov[1]} [$^{{\\circ}}$] | resolution={self.resolution[0]}x{self.resolution[1]} px\n'
                f'{time_range[0].strftime("%Y-%m-%d %H:%M:%S")} - '
                f'{time_range[1].strftime("%H:%M:%S")}', fontsize=15
                )

        gen = self.asi.animate_map_gen(
            ax=self.ax, 
            asi_label=False, 
            lon_bounds=lon_bounds, 
            lat_bounds=lat_bounds,
            color_bounds=color_bounds, 
            pcolormesh_kwargs={'rasterized':True},
            overwrite=True,
            ffmpeg_params={'framerate':5},
            timestamp=False,
            )
        
        legend_plotted = False
        for i, (time, image, _, im) in enumerate(gen):
            if '_footprint_dot' in locals():
                # This is one way I found to clean up an added plotting object.
                _footprint_dot.remove()
                _vertical_line.remove()
                _lampsat_dot.remove()
                _lampsat_fov_dot.remove()
                _asi_timestamp.remove()
                try:
                    _perimiter_plot.remove()
                except TypeError:
                    del(_perimiter_plot)
                p3.remove()

            _asi_timestamp = self.ax.text(
                0.01, 0.96, 
                f'(a) THEMIS ASI {time:%H:%M:%S}', 
                va='center',
                transform=self.ax.transAxes, 
                weight='bold', 
                fontsize=15
                )
            _asi_timestamp.set_bbox(dict(facecolor='white', pad=0.25))

            footprint_idx = self.high_alt_footprint.index.get_indexer(
                [time], method='nearest', tolerance=pd.Timedelta(seconds=2)
                )
            lampsat_lon_lat = self.high_alt_footprint.iloc[footprint_idx][['lon', 'lat']].values[0]
            footprint_lon_lat = self.low_alt_footprint.iloc[footprint_idx][['lon', 'lat']].values[0]
            if np.isnan(self.high_alt_footprint.iloc[footprint_idx]['lat']).values:
                continue

            _lampsat_dot = self.ax.scatter(
                lampsat_lon_lat[0], 
                lampsat_lon_lat[1], 
                c='blue', s=100, marker='x',
                transform=ccrs.PlateCarree(),
                label=f'Satellite at {lampsat_alt} km'
                )
            _footprint_dot = self.ax.scatter(
                footprint_lon_lat[0], 
                footprint_lon_lat[1], 
                c='red', s=150, marker='.',
                transform=ccrs.PlateCarree(),
                label=f'Footprint at {aurora_alt} km'
                )
            _vertical_line = self.cx.axvline(
                time, c='k', ls='--'
                )
            
            if not legend_plotted:
                self.ax.legend(loc='lower right', fontsize=12, framealpha=0.5)
                legend_plotted = True
            
            aic_lons, aic_lats = self._calc_aic_skymap(time)
            footprint_px = self.footprint_fov(
                aic_lons, aic_lats, footprint_lon_lat[0], footprint_lon_lat[1]
                )

            asis = asilib.Imagers(self.asi[time])
            lat_lon_points, intensities = asis.get_points()
            interp_grid = scipy.interpolate.griddata(lat_lon_points, intensities, (aic_lats, aic_lons), method='cubic')

            interp_grid_copy = interp_grid.copy()
            # We need to mask out the gridded points that are far away from the original points as NaNs and this
            # is the most efficient way (source: https://stackoverflow.com/a/31189177)
            tree = cKDTree(lat_lon_points)
            xi = np.stack((aic_lats, aic_lons), axis=-1)
            dists, _ = tree.query(xi, distance_upper_bound=0.15)
            interp_grid_copy[~np.isfinite(dists)] = np.nan

            lon_perimeter = np.concatenate((
                aic_lons[0, :], 
                aic_lons[:, -1], 
                aic_lons[-1, ::-1], 
                aic_lons[::-1, 0]
                ))
            lat_perimeter = np.concatenate((
                aic_lats[0, :], 
                aic_lats[:, -1], 
                aic_lats[-1, ::-1], 
                aic_lats[::-1, 0]
                ))

            _perimiter_plot, = self.ax.plot(
                lon_perimeter, 
                lat_perimeter, 
                ls='--', 
                color='purple', 
                lw=2, 
                zorder=2.1, 
                transform=ccrs.PlateCarree(),
                )
            if self.checkerboard:
                self.bx.pcolormesh(self._checkerboard_xx, self._checkerboard_yy, self._checkerboard, cmap='Reds', vmin=0, vmax=1, rasterized=True)
            p3 = self.bx.pcolormesh(
                interp_grid_copy.T, 
                cmap='Greys_r', 
                vmin=self.color_bounds[0], 
                vmax=self.color_bounds[1], 
                rasterized=True
                )
            if i == 0:
                plt.colorbar(p3, ax=self.bx, label='THEMIS ASI intensity [counts]')
            _lampsat_fov_dot = self.bx.scatter(
                footprint_px[0], 
                footprint_px[1], 
                c='red', s=150, marker='.'
                )
        return
    
    def _calc_aic_skymap(self, time):
        """
        Calculat the AIC latitude and logitude skymaps for a given time.
        """
        footprint_idx = self.high_alt_footprint.index.get_indexer(
                [time], method='nearest', tolerance=pd.Timedelta(seconds=2)
                )
            
        aic_lons = np.zeros_like(self.azs).flatten()
        aic_lats = np.zeros_like(self.azs).flatten()
        for k, (azs_i, tilt_i) in enumerate(zip(self.azs.flatten(), self.tilts.flatten())):
            aic_lats[k], aic_lons[k], _ = pymap3d.los.lookAtSpheroid(
                self.high_alt_footprint.iloc[footprint_idx]['lat'], 
                self.high_alt_footprint.iloc[footprint_idx]['lon'], 
                1e3*self.high_alt_footprint.iloc[footprint_idx]['alt'],
                azs_i,
                tilt_i,
                ell=Ellipsoid_alt(1e3*aurora_alt)
                )
        aic_lons = aic_lons.reshape(self.azs.shape)
        aic_lats = aic_lats.reshape(self.azs.shape)
        return aic_lons, aic_lats

    def footprint_fov(self, aic_lons, aic_lats, footprint_lon, footprint_lat):
        """
        Calculate the pixel indices of the footprint that is inside the AIC FOV.

        Parameters
        ----------
        aic_lons: np.ndarray
            The longitude grid of the AIC FOV.
        aic_lats: np.ndarray
            The latitude grid of the AIC FOV.
        footprint_lon: float
            The longitude of the footprint.
        footprint_lat: float
            The latitude of the footprint.
        """
        dists = haversine(
            aic_lats,
            aic_lons,
            footprint_lat*np.ones_like(aic_lats), 
            footprint_lon*np.ones_like(aic_lats),
            r=R_e
            )
        idx = np.argmin(dists)
        if dists.flatten()[idx] > 20:
            return np.array([np.nan, np.nan])
        # plt.close()
        # plt.hist(dists.flatten(), bins=np.arange(50))
        # plt.show()
        # raise NotImplementedError
        return np.unravel_index(idx, aic_lats.shape)


@dataclasses.dataclass
class ASI_OSE_Montage():
    """
    Calculate the THEMIS ASI white-light intensity inside a space-based imager FOV and make a montage plot at n time stamps.

    Parameters
    ----------
    time_range: Tuple[datetime]
        Defines the time range for the OSE plot.
    fov: Tuple[float]
        The field of view of the AIC in degrees.
    resolution: Tuple[int]
        The resolution of the AIC in pixels (width, height).
    ona: float
        The off-nadir angle between nadir and the imager's center FOV vectors. If 0, the center of
        the FOV is pointing towards the nadir and if 90 it points at the limb.
    azimuth: float
        The azimuth angle of the AIC FOV measured clockwise from north.
    lampsat_alt: float
        The altitude of the LAMPsat in kilometers.
    themis_location_code: str
        The THEMIS location code, e.g., 'WHIT' for THEMIS ASI.
    aurora_alt: float
        The altitude of the aurora in kilometers.
    times: int | Tuple[datetime]
        The number of time stamps to plot or a tuple of specific datetime objects.
        If an integer is provided, it will plot that many evenly spaced time stamps 
        within the time_range.
    """
    time_range:Tuple[datetime]
    fov:Tuple[float]=(45, 30)
    resolution:Tuple[int]=(64, 64)
    ona:float=0,
    azimuth:float=0,
    lampsat_alt:float=500
    themis_location_code:str='WHIT'
    aurora_alt:float=110
    times:int | Tuple[datetime] = 4
    lon_bounds:Tuple[float]=None
    lat_bounds:Tuple[float]=None 
    color_bounds:Tuple[int]=None
    checkerboard:bool=True
    detrend_hilt:bool=False
    detrend_duration_s:float=5
    detrend_quantile:float=0.5
    hilt_logscale:bool=True
    hilt_ylim:tuple=(-20, 4*10**3)

    def __post_init__(self):
        self.ose_animation = ASI_OSE_Animation(
            self.time_range, 
            fov=self.fov, 
            resolution=self.resolution,
            ona=self.ona,
            azimuth=self.azimuth,
            lampsat_alt=self.lampsat_alt, 
            themis_location_code=self.themis_location_code, 
            aurora_alt=self.aurora_alt,
            lon_bounds=self.lon_bounds,
            lat_bounds=self.lat_bounds,
            color_bounds=self.color_bounds,
            detrend_hilt=self.detrend_hilt,
            detrend_duration_s=self.detrend_duration_s,
            detrend_quantile=self.detrend_quantile,
            hilt_logscale=self.hilt_logscale,
            hilt_ylim=self.hilt_ylim
        )
        if isinstance(self.times, int):
            if self.times < 1:
                raise ValueError("The number of time stamps must be at least 1.")
            time_step = (self.time_range[1] - self.time_range[0]) / (self.times+1)
            self.times = [self.time_range[0] + (i+1) * time_step for i in range(self.times)]
        return
    
    def load_data(self):
        self.ose_animation.load_data()
        self.high_alt_footprint = self.ose_animation.high_alt_footprint
        self.low_alt_footprint = self.ose_animation.low_alt_footprint
        self.asi = self.ose_animation.asi
        self.hilt = self.ose_animation.hilt
        self.asi_nearest_count_intensity = self.ose_animation.asi_nearest_count_intensity
        self.asi_557_intensity = self.ose_animation.asi_557_intensity
    
    def plot_montage(self, cmap='Greys_r', noise_floor=None, sensitivity=None, n_binned_pixels=None):
        fig = plt.figure(figsize=(9, 9))
        spec = gridspec.GridSpec(nrows=4, ncols=len(self.times), figure=fig, height_ratios=(1, 1, 1, 1))
        self.ax = [None] * len(self.times)
        for i, time in enumerate(self.times):
            self.ax[i] = asilib.map.create_map(
                lon_bounds=lon_bounds, lat_bounds=lat_bounds, fig_ax=(fig, spec[0, i])
                )
            # ax[i] = fig.add_subplot(spec[0, i], projection=ccrs.PlateCarree())
        self.bx = [None] * len(self.times)
        for i, time in enumerate(self.times):
            self.bx[i] = fig.add_subplot(spec[1, i])
            self.bx[i].set_aspect('equal')

        self.cx = fig.add_subplot(spec[2, :])
        if sensitivity is not None:
            self.count_cx = self.cx.twinx()
        self.dx = fig.add_subplot(spec[3, :], sharex=self.cx)

        plt.subplots_adjust(
            top=0.922,
            bottom=0.05,
            left=0.083,
            right=0.895,
            hspace=0.124,
            wspace=0.071
            )

        for i, (ax_i, bx_i, time) in enumerate(zip(self.ax, self.bx, self.times)):

            ax_label = string.ascii_lowercase[i]
            bx_label = string.ascii_lowercase[len(self.times)+i]

            ax_i.plot(
                self.low_alt_footprint.loc[:, 'lon'], 
                self.low_alt_footprint.loc[:, 'lat'], 
                'r:', 
                transform=ccrs.PlateCarree()
                )
            self.asi[time].plot_map(
                ax=ax_i, 
                asi_label=False, 
                lon_bounds=lon_bounds, 
                lat_bounds=lat_bounds,
                pcolormesh_kwargs={'rasterized':True},
                color_bounds=self.color_bounds,
            )

            footprint_idx = self.high_alt_footprint.index.get_indexer(
                [time], method='nearest', tolerance=pd.Timedelta(seconds=2)
                )

            lampsat_lon_lat = self.high_alt_footprint.iloc[footprint_idx][['lon', 'lat']].values[0]
            footprint_lon_lat = self.low_alt_footprint.iloc[footprint_idx][['lon', 'lat']].values[0]

            ax_i.scatter(
                lampsat_lon_lat[0], 
                lampsat_lon_lat[1], 
                c='blue', s=100, marker='X',
                transform=ccrs.PlateCarree(),
                label=f'Satellite at {lampsat_alt} km'
                )
            ax_i.scatter(
                footprint_lon_lat[0], 
                footprint_lon_lat[1], 
                c='red', s=150, marker='.',
                transform=ccrs.PlateCarree(),
                label=f'Footprint at {aurora_alt} km'
                )
            
            aic_lons, aic_lats = self.ose_animation._calc_aic_skymap(time)
            footprint_px = self.ose_animation.footprint_fov(
                aic_lons, aic_lats, footprint_lon_lat[0], footprint_lon_lat[1]
                )
            asis = asilib.Imagers(self.asi[time])
            lat_lon_points, intensities = asis.get_points()
            interp_grid = scipy.interpolate.griddata(lat_lon_points, intensities, (aic_lats, aic_lons), method='cubic')

            interp_grid_copy = interp_grid.copy()
            # We need to mask out the gridded points that are far away from the original points as NaNs and this
            # is the most efficient way (source: https://stackoverflow.com/a/31189177)
            tree = cKDTree(lat_lon_points)
            xi = np.stack((aic_lats, aic_lons), axis=-1)
            xi_finite = xi.copy()
            xi_finite[~np.isfinite(xi_finite)] = -999  # tree.query can take only finite values.
            dists, _ = tree.query(xi_finite, distance_upper_bound=0.15)
            interp_grid_copy[
                ~np.isfinite(dists) | 
                ~np.isfinite(xi[..., 0]) | 
                ~np.isfinite(xi[..., 1])
                ] = np.nan

            lon_perimeter = np.concatenate((
                aic_lons[0, :], 
                aic_lons[:, -1], 
                aic_lons[-1, ::-1], 
                aic_lons[::-1, 0]
                ))
            lat_perimeter = np.concatenate((
                aic_lats[0, :], 
                aic_lats[:, -1], 
                aic_lats[-1, ::-1], 
                aic_lats[::-1, 0]
                ))

            _perimiter_plot, = ax_i.plot(
                lon_perimeter, 
                lat_perimeter, 
                ls='--', 
                color='purple', 
                lw=2, 
                zorder=2.1,
                transform=ccrs.PlateCarree(),
                )

            if self.checkerboard:
                bx_i.pcolormesh(
                    self.ose_animation._checkerboard_xx, 
                    self.ose_animation._checkerboard_yy, 
                    self.ose_animation._checkerboard, 
                    cmap='Reds', 
                    vmin=0, 
                    vmax=1, 
                    rasterized=True,
                    )
            p3 = bx_i.pcolormesh(
                interp_grid_copy.T, 
                cmap=cmap, 
                vmin=self.color_bounds[0], 
                vmax=self.color_bounds[1]
                )
            bx_i.scatter(
                footprint_px[0], 
                footprint_px[1], 
                c='red', s=150, marker='.'
                )

            _text = ax_i.text(
                0.01, 0.99, f'({ax_label}) {time:%H:%M:%S}', va='top', transform=ax_i.transAxes, fontsize=14
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
            _text = bx_i.text(
                0.01, 0.99, f'({bx_label})', va='top', transform=bx_i.transAxes, fontsize=14
                )
            _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))

            bx_i.xaxis.set_visible(False)
            bx_i.yaxis.set_visible(False)

        self.cx.plot(self.asi.data.time, self.asi_557_intensity/2E3)
        self.cx.set_ylabel(f'391.4 nm Intensity [kR]')
        if sensitivity is not None:
            self.aic_391_counts = self.asi_557_intensity.copy()
            self.aic_391_counts -= noise_floor
            self.aic_391_counts /= sensitivity
            self.aic_391_counts *= n_binned_pixels
            self.count_cx.plot(self.asi.data.time, self.aic_391_counts/1E3, c='purple', lw=2)
            self.count_cx.set_ylabel(f'1000*cts/0.5 s/binned pixel')
        plt.setp(self.cx.get_xticklabels(), visible=False)
            
        self.dx.plot(self.hilt.index, self.hilt['counts'], c='r')
        self.dx.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=5))
        self.dx.set_xlim(*time_range)
        if self.hilt_logscale:
            self.dx.set_yscale('log')
            self.dx.set_ylim(*self.hilt_ylim)
        else:
            self.dx.set_ylim(*self.hilt_ylim)
        self.dx.set_ylabel(f'[counts/20 ms]')
        self.dx.xaxis.set_major_locator(matplotlib.dates.SecondLocator(interval=30))

        # Connect the subplots and add vertical lines to cx and dx.
        for bx_i, image_time_numeric in zip(self.bx, matplotlib.dates.date2num(self.times)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(0.5, 0), coordsA=bx_i.transAxes,
                xyB=(image_time_numeric, self.cx.get_ylim()[1]), coordsB=self.cx.transData, 
                ls='--')
            bx_i.add_artist(line)
            self.cx.axvline(image_time_numeric, c='k', ls='--', alpha=1)
            self.dx.axvline(image_time_numeric, c='k', ls='--', alpha=1)

        manylabels.ManyLabels(
            self.dx, 
            self.high_alt_footprint, 
            label_coord=(-0.07, -0.05)
            )
        self.cx.text(0, 0.99, f'({string.ascii_lowercase[2*len(self.times)]}) 391.4 nm emission and AIC counts', va='top', 
            transform=self.cx.transAxes, fontsize=14
            )
        self.dx.text(0, 0.99, f'({string.ascii_lowercase[2*len(self.times)+1]}) SAMPEX-HILT >1 MeV electrons', va='top', 
            transform=self.dx.transAxes, fontsize=14
            )
        
        plt.suptitle(
            f'LAMPsat OSE | FOV={self.fov}\nresolution={self.resolution} px | '
            f'{time_range[0].strftime("%Y-%m-%d %H:%M:%S")} - '
            f'{time_range[1].strftime("%H:%M:%S")}', fontsize=15)
        return



def haversine(
    lat1: np.array, lon1: np.array, lat2: np.array, lon2: np.array, r: float = 1
) -> np.array:
    """
    Haversine distance equation.

    Parameters
    ----------
    lat1, lat2: np.array
        The latitude of points 1 and 2 in units of degrees. Can be n-dimensional.
    lon1, lon2: np.array
        The longitude of points 1 and 2 in units of degrees. Can be n-dimensional.
    r: float
        The sphere radius.
    """
    assert (
        lat1.shape == lon1.shape == lat2.shape == lon2.shape
    ), 'All input arrays must have the same shape.'
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)

    d = (
        2
        * r
        * np.arcsin(
            np.sqrt(
                np.sin((lat1_rad - lat2_rad) / 2) ** 2
                + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin((lon2_rad - lon1_rad) / 2) ** 2
            )
        )
    )
    return d


if __name__ == '__main__':
    fov = (45, 45)  # degrees
    resolution = (64, 64)  # pixels
    ona = 0
    azimuth = 0
    lampsat_alt = 500  # km
    aurora_alt = 110  # km

    # Event used in Decadal white paper
    # time_range = (datetime(2007, 2, 14, 13, 29, 40), datetime(2007, 2, 14, 13, 31, 30))
    # lon_bounds = (-144, -127)
    # lat_bounds = (56, 66)
    # color_bounds = (3_000, 3_700)
    # themis_location_code = 'WHIT'
    # times=4

    # Event from Shumko+2021
    time_range = (
        datetime(2008, 1, 16, 10, 58, 45), 
        datetime(2008, 1, 16, 11, 1, 30)
        )
    themis_df = asilib.asi.themis_info()
    themis_location_code = 'GILL'
    themis_latlon = themis_df.loc[
        (
            (themis_df['location_code'] == themis_location_code) & 
            (themis_df['array'] == 'THEMIS')
        ),
            ['latitude', 'longitude']
        ].values[0]
    
    lon_bounds = (themis_latlon[1]-8, themis_latlon[1]+8)
    lat_bounds = (themis_latlon[0]-5, themis_latlon[0]+5)
    color_bounds = (4_100, 5_500)
    times=(
        datetime(2008, 1, 16, 10, 59, 40), 
        datetime(2008, 1, 16, 11, 0, 3), 
        datetime(2008, 1, 16, 11, 0, 15), 
        datetime(2008, 1, 16, 11, 0, 30)
        )

    ose = ASI_OSE_Montage(
        time_range,
        times=times,
        fov=fov, 
        resolution=resolution,
        ona=ona,
        azimuth=azimuth,
        lampsat_alt=lampsat_alt, 
        themis_location_code=themis_location_code, 
        aurora_alt=aurora_alt,
        lon_bounds=lon_bounds, 
        lat_bounds=lat_bounds, 
        color_bounds=color_bounds,
        detrend_hilt=True,
        detrend_quantile=0.5,
        hilt_logscale=False,
        hilt_ylim=(-20, 2_500)
    )
    ose.load_data()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'black_to_purple', ['black', 'purple']
        )
    # ose.plot_montage(cmap='Greys_r', noise_floor=287, sensitivity=1_230, n_binned_pixels=1024)
    ose.ose_animation.animate()
    # plt.show()