"""
This Observational System Experiment (OSE) reinterpolates ASI data to calculate the 
anticipated auroral images from a low Earth orbiting (LEO) satellite that is equipped
with an auroral imager.

Remember, this is different than an OSSE (Observing System Simulation Experiment) which
simulates the instrument response from model data. For this type of OSSE, the model input
will be the auroral intensity on a (lat, lon) grid as a function of time.
"""
import dataclasses
import copy
from datetime import datetime
import pathlib
import string
from typing import Tuple, List, Union
from collections import namedtuple
import dateutil.parser

import fontawesome
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.path import Path
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


@dataclasses.dataclass
class OSE:
    """
    Calculate the THEMIS ASI white-light intensity inside a space-based imager FOV.

    Parameters
    ----------
    imagers: asilib.Imagers
        The imagers to be used in the OSE.
    ephemeris: Tuple[np.ndarray, np.ndarray]
        The two-element ephemeris tuple with the first element the n timestamps, and the second 
        element either a (n, 3) array for one satellite ephemeris with columns corresponding to
        the (lat, lon, alt) (LLA) coordinates, or a (n, 3, m) array for m satellite ephemerides 
        with the LLA coordinates. Lat and lon are in degrees and alt is in kilometers.
    fov: Tuple[float]
        The field of view of the AIC in degrees.
    pixel_resolution: Tuple[int]
        The resolution of the AIC in pixels (width, height).
    ona: float
        The off-nadir angle between nadir and the imager's center FOV vectors. If 0, the center of
        the FOV is pointing towards the nadir and if 90 it points at the limb.
    azimuth: float
        The azimuth angle of the AIC FOV measured clockwise from north.
    aurora_alt: float
        The altitude of the aurora in kilometers.
    """
    imagers: asilib.Imagers
    ephemeris: np.ndarray
    fov:Tuple[float]=(45, 45)
    pixel_resolution:Tuple[int]=(64, 64)
    ona:float=0  # TODO: Implement
    azimuth:float=0 # TODO: Implement
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
            np.linspace(-self.fov[1]/2, self.fov[1]/2, self.pixel_resolution[1]),
            np.linspace(-self.fov[0]/2, self.fov[0]/2, self.pixel_resolution[0]),
            )
        
        if self.ona != 0 or self.azimuth != 0:
            raise NotImplementedError(
                "The off-nadir angle and azimuth are not yet implemented. Please submit"
                "a feature request on GitHub if you would like this functionality."
                )
        self.xx += np.sin(np.deg2rad(self.azimuth))*self.ona
        self.yy += np.cos(np.deg2rad(self.azimuth))*self.ona
        self.tilts = np.sqrt(self.xx**2 + self.yy**2)
        self.azs = np.rad2deg(np.arctan2(self.yy, self.xx))

        self._checkerboard = np.zeros((10, 10), dtype=bool)
        self._checkerboard[::2, ::2] = True
        self._checkerboard[1::2, 1::2] = True
        self._checkerboard_xx, self._checkerboard_yy = np.meshgrid(
            np.linspace(0, self.pixel_resolution[0], num=self._checkerboard.shape[0]+1),
            np.linspace(0, self.pixel_resolution[1], num=self._checkerboard.shape[1]+1)
            )
        if len(self.ephemeris[1].shape) == 2:
            self.n_satellites = 1
        elif len(self.ephemeris[1].shape) == 3:
            self.n_satellites = self.ephemeris[1].shape[-1]
        else:
            raise ValueError(
                f"Unexpected ephemeris shape: {self.ephemeris[1].shape}. Expected (n, 3) or "
                f"(n, 3, m) where n is the number of timestamps and m is the number of "
                f"satellites."
                )
        return

    def get_image(self, time):
        """
        Get image(s) from the OSE for a given time. Depending on if there is one more satellites,
        the shape of the image(s) will be (pixel_resolution[0], pixel_resolution[1]) if there is
        one satellite, or (pixel_resolution[0], pixel_resolution[1], n_satellites) if there are 
        multiple satellites.

        The satellite locations are indexed from the ephemeris timestamps.
        
        Parameters
        ----------
        time: datetime
            The time for which to get the image(s).
        
        Returns
        -------
        images: np.ndarray
            The image(s) from the OSE for the given time.
        """

        if self.n_satellites == 1:
            _ephemeris = self.ephemeris[1].reshape(*self.ephemeris[1].shape, 1)
        else:
            _ephemeris = self.ephemeris[1]

        images = np.zeros(
            (self.pixel_resolution[0], self.pixel_resolution[1], self.n_satellites)
            )

        _imagers = self.imagers[time]
        lat_lon_points, intensities = _imagers.get_points()

        ephemeris_time_np = np.array(self.ephemeris[0], dtype='datetime64')
        ephemeris_time_dt = np.abs(ephemeris_time_np-np.datetime64(time))
        closest_idt = np.argmin(ephemeris_time_dt)

        if ephemeris_time_dt[closest_idt] > ephemeris_time_np[1] - ephemeris_time_np[0]:
            raise ValueError(
                f"Time {time} is too far from the nearest ephemeris timestamps:"
                f"{self.ephemeris[0][closest_idt]}."
                )
        
        for i in range(self.n_satellites):
            lla = _ephemeris[closest_idt, :, i]
            
            lat_skymap, lon_skymap = self.imager_skymap(time, lla)

            interp_grid = scipy.interpolate.griddata(
                lat_lon_points, 
                intensities, 
                (lat_skymap, lon_skymap), 
                method='cubic'
                )

            # We need to mask out the gridded points that are too far away from the original points as
            # NaNs and this is the most efficient way (source: https://stackoverflow.com/a/31189177).
            interp_grid_nans = interp_grid.copy()
            tree = cKDTree(lat_lon_points)
            xi = np.stack((lat_skymap, lon_skymap), axis=-1)
            dists, _ = tree.query(xi, distance_upper_bound=0.15)
            interp_grid_nans[~np.isfinite(dists)] = np.nan
            images[:, :, i] = interp_grid_nans

        if self.n_satellites == 1:
            images = images[:, :, 0]
        return images

    def get_mapped_fov(self, time, lla):
        """
        Get the mapped field of view (FOV) of the imager for a given time and satellite location.
        """
        lat_skymap, lon_skymap = self.imager_skymap(time, lla)

        lon_perimeter = np.concatenate((
            lon_skymap[0, :], 
            lon_skymap[:, -1], 
            lon_skymap[-1, ::-1], 
            lon_skymap[::-1, 0]
            ))
        lat_perimeter = np.concatenate((
            lat_skymap[0, :], 
            lat_skymap[:, -1], 
            lat_skymap[-1, ::-1], 
            lat_skymap[::-1, 0]
            ))
        return lat_perimeter, lon_perimeter

    def imager_skymap(self, time, lla):
        """
        Calculate the imager latitude and longitude skymaps for a given time and satellite 
        location.

        Parameters
        ----------
        time: datetime
            The time for which to calculate the skymap.
        lla: Tuple[float]
            The satellite location in (lat, lon, alt) format, where lat and lon are in degrees 
            and alt is in kilometers.

        Returns
        -------
        imager_lats: np.ndarray
            The latitude skymap of a single-satellite imager FOV with shape 
            (pixel_resolution[0], pixel_resolution[1])
        imager_lons: np.ndarray
            The longitude skymap of a single-satellite imager FOV with shape 
            (pixel_resolution[0], pixel_resolution[1])
        """
        imager_lats = np.zeros_like(self.azs).flatten()
        imager_lons = np.zeros_like(self.azs).flatten()

        for k, (azs_i, tilt_i) in enumerate(zip(self.azs.flatten(), self.tilts.flatten())):
            imager_lats[k], imager_lons[k], _ = pymap3d.los.lookAtSpheroid(
                lla[0], 
                lla[1], 
                1e3*lla[2],
                azs_i,
                tilt_i,
                ell=Ellipsoid_alt(1e3*self.aurora_alt)
                )
 
        return imager_lats.reshape(self.azs.shape), imager_lons.reshape(self.azs.shape)


class Ellipsoid_alt:
    """
    Adapted from https://geospace-code.github.io/pymap3d/ellipsoid.html

    Distance units are meters.
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


def getmarker(mID):
	symbol = fontawesome.icons[mID]
	fp = FontProperties(fname=pathlib.Path(__file__).parent / "Font Awesome 7 Free-Solid-900.otf")

	v, codes = TextToPath().get_text_path(fp, symbol)
	v = np.array(v)
	mean = np.mean([np.max(v,axis=0), np.min(v, axis=0)], axis=0)
	return Path(v-mean, codes, closed=False)


if __name__ == '__main__':

    # TODO: Remove after debugging.
    import time

    from asilib.mission import example_satellite
    from datetime import datetime

    time_range = (datetime(2012, 2, 15, 8, 30), datetime(2012, 2, 15, 8, 45))

    location_codes = [
        'FSIM',
        'FSMI',
        'ATHA',
        'TPAS',
        'GILL',
        ]
    
    asis = asilib.Imagers([asilib.asi.themis(code, time_range=time_range) for code in location_codes])

    # Create the CINEMA constellation ephemeris.
    orbit_parameter_type = namedtuple('orbit_parameter_type', ['mean_anomaly_deg', 'ltan_hours'])
    in_track_separation_minutes = 5
    mean_anomaly_deg = 35
    delta_mean_anomaly_deg = 360*in_track_separation_minutes/95
    constellation = {
        1:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg+delta_mean_anomaly_deg, 
            ltan_hours=1
            ),
        2:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg+delta_mean_anomaly_deg, 
            ltan_hours=2
            ),
        3:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg+delta_mean_anomaly_deg, 
            ltan_hours=3
            ),
        4:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg, 
            ltan_hours=1
            ),
        5:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg, 
            ltan_hours=2
            ),
        6:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg, 
            ltan_hours=3
            ),
        7:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg-delta_mean_anomaly_deg, 
            ltan_hours=1
            ),
        8:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg-delta_mean_anomaly_deg, 
            ltan_hours=2
            ),
        9:orbit_parameter_type(
            mean_anomaly_deg=mean_anomaly_deg-delta_mean_anomaly_deg, 
            ltan_hours=3
            ),
    }
    ephemeris = [None, None]
    for key, value in constellation.items():
        ephemeris_obj = example_satellite.Example_Satellite(
            cadence_s=0.5,
            time_range=time_range,
            mean_anomaly_deg=value.mean_anomaly_deg,
            ltan_hours=value.ltan_hours,
        )
        sat_ephemeris = ephemeris_obj.ephemeris()
        # ephemeris[1][key] = sat_ephemeris[1]
        if ephemeris[0] is None:
            ephemeris[0] = sat_ephemeris[0]
            ephemeris[1] = sat_ephemeris[1].reshape(*sat_ephemeris[1].shape, 1)
        else:
            ephemeris[1] = np.concatenate(
                (ephemeris[1], sat_ephemeris[1].reshape(*sat_ephemeris[1].shape, 1)), axis=2
                )


    print('Enable breakpoints now...')
    time.sleep(2)

    ose = OSE(asis, ephemeris)
    
    images = ose.get_image(datetime(2012, 2, 15, 8, 30))

    fig = plt.figure(figsize=(4, 7), layout='tight')
    gs = gridspec.GridSpec(nrows=4, ncols=3, figure=fig, height_ratios=(3, 1, 1, 1))

    ax = asilib.map.create_map(
        lon_bounds=asis.lon_bounds, 
        lat_bounds=(asis.lat_bounds[0]-1, asis.lat_bounds[1]+1), 
        fig_ax=(fig, gs[0, :])
        )
    bx = np.nan*np.zeros((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            bx[i, j] = fig.add_subplot(gs[i+1, j])
            bx[i, j].set_aspect('equal')
            bx[i, j].xaxis.set_visible(False)
            bx[i, j].yaxis.set_visible(False)

    plt.show()

    g = asis.animate_map_gen(ax=ax, pcolormesh_kwargs={'rasterized':True}, overwrite=True)

    for i, (guide_time, image, _, im) in enumerate(g):
        if i == 0:
            for _ephemeris in ephemeris[1].values():
                ax.plot(_ephemeris['lon'], _ephemeris['lat'], 'k:', transform=ccrs.PlateCarree())
        else:
            for scatter_point in scatter_points:
                scatter_point.remove()
            for sc_label in sc_labels:
                sc_label.remove()
        
        scatter_points = []
        sc_labels = []
        for sc, _ephemeris in ephemeris[1].items():
            idx_loc = _ephemeris.index.get_indexer([guide_time], method='nearest', tolerance=pd.Timedelta(seconds=2))[0]
            sat_loc = _ephemeris.iloc[idx_loc][['lon', 'lat']].values

            scatter_points.append(
                ax.scatter(sat_loc[0], sat_loc[1], c='purple', s=200, marker=getmarker('camera'), transform=ccrs.PlateCarree())
            )
            sc_labels.append(
                ax.text(sat_loc[0]+0.5, sat_loc[1], f'SC{sc}', color='white', fontsize=12, transform=ccrs.PlateCarree(), va='center')
            )

        # if i == 0:
        #     ax.legend(loc='lower right', fontsize=12, framealpha=0.5)


    pass
