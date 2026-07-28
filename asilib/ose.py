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
            dists, _ = tree.query(xi, distance_upper_bound=0.3)
            interp_grid_nans[~np.isfinite(dists)] = np.nan
            images[:, :, i] = interp_grid_nans.T

        if self.n_satellites == 1:
            images = images[:, :, 0]
        return images

    def get_mapped_fov(self, time):
        """
        Get the mapped field of view (FOV) of the imager for a given time and satellite location.
        """
        if self.n_satellites == 1:
            _ephemeris = self.ephemeris[1].reshape(*self.ephemeris[1].shape, 1)
        else:
            _ephemeris = self.ephemeris[1]

        ephemeris_time_np = np.array(self.ephemeris[0], dtype='datetime64')
        ephemeris_time_dt = np.abs(ephemeris_time_np-np.datetime64(time))
        closest_idt = np.argmin(ephemeris_time_dt)

        if ephemeris_time_dt[closest_idt] > ephemeris_time_np[1] - ephemeris_time_np[0]:
            raise ValueError(
                f"Time {time} is too far from the nearest ephemeris timestamps:"
                f"{self.ephemeris[0][closest_idt]}."
                )

        lon_perimeter = np.zeros(
            (2*self.pixel_resolution[0]+2*self.pixel_resolution[1], self.n_satellites), 
            dtype=float
            )
        lat_perimeter = np.zeros_like(lon_perimeter)

        for i in range(self.n_satellites):
            lla = _ephemeris[closest_idt, :, i]
            lat_skymap, lon_skymap = self.imager_skymap(time, lla)

            lon_perimeter[:, i] = np.concatenate((
                lon_skymap[0, :], 
                lon_skymap[:, -1], 
                lon_skymap[-1, ::-1], 
                lon_skymap[::-1, 0]
                ))
            lat_perimeter[:, i] = np.concatenate((
                lat_skymap[0, :], 
                lat_skymap[:, -1], 
                lat_skymap[-1, ::-1], 
                lat_skymap[::-1, 0]
                ))

        if self.n_satellites == 1:
            lon_perimeter = lon_perimeter[..., 0]
            lat_perimeter = lat_perimeter[..., 0]
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

    def plot_image(self):

        return

    def animate_ose(self, ax=None, bx=None, color_bounds=None, **kwargs):
        """
        Animate the Observational System Experiment (OSE) images.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axis on which to plot the ASI mosaic, orbit tracks, and FOVs. 
            If None, a new figure and axis are created.
        bx: matplotlib.axes.Axes, optional
            The axis on which to plot the OSE images. If None, a new figure and axis 
            are created with one row with columns for each satellite.
        color_bounds: list, optional
            The color bounds for the images. If None, the default color bounds are used.

        Returns
        -------
        None
        """
        g = self.animate_ose_gen(ax=ax, bx=bx, color_bounds=color_bounds, **kwargs)
        for _ in g:
            pass
        return

    def animate_ose_gen(self, ax=None, bx=None, color_bounds=None, **kwargs):
        """
        Animate the Observational System Experiment (OSE) image generator.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axis on which to plot the ASI mosaic, orbit tracks, and FOVs. 
            If None, a new figure and axis are created.
        bx: matplotlib.axes.Axes, optional
            The axis on which to plot the OSE images. If None, a new figure and axis 
            are created with one row with columns for each satellite.
        color_bounds: list, optional
            The color bounds for the images. If None, the default color bounds are used.
        kwargs: dict
            Additional keyword arguments. The complete list of kwargs is in the 
            :py:meth:`~asilib.Imagers.animate_map_gen` documentation.

        Returns
        -------
        generator
            A generator that yields the guide_time, ax, bx, images, and 
            (lon_perimeter, lat_perimeter) tuple
        """
        if color_bounds is None:
            color_bounds = self.imagers.imagers[0].auto_color_bounds()

        if ax is None:
            fig = plt.figure(figsize=(4, 7), layout='tight')
            gs = gridspec.GridSpec(nrows=2, ncols=self.n_satellites, figure=fig, height_ratios=(1, 1))
        
            self.ax = asilib.map.create_map(
                lon_bounds=self.imagers.lon_bounds, 
                lat_bounds=(self.imagers.lat_bounds[0]-3, self.imagers.lat_bounds[1]+3), 
                fig_ax=(fig, gs[0, :])
                )
            self.bx = np.nan*np.zeros((gs.nrows-1, gs.ncols), dtype=object)
            for i in range(gs.nrows-1):
                for j in range(gs.ncols):
                    self.bx[i, j] = fig.add_subplot(gs[i+1, j])
                    self.bx[i, j].set_aspect('equal')
                    self.bx[i, j].xaxis.set_visible(False)
                    self.bx[i, j].yaxis.set_visible(False)
        else:
            self.ax = ax
            self.bx = bx
            
        g = self.imagers.animate_map_gen(
            ax=self.ax,
            color_bounds=color_bounds,
            pcolormesh_kwargs={'rasterized':True}, 
            overwrite=True
            )
        
        for i, (guide_time, image, _, im) in enumerate(g):
            if i == 0:
                for _ephemeris in np.moveaxis(self.ephemeris[1], -1, 0):
                    self.ax.plot(_ephemeris[:, 1], _ephemeris[:, 0], 'k:', transform=ccrs.PlateCarree())

                for j, bx_i in enumerate(self.bx.flatten()):
                    _text = bx_i.text(0.01, 0.99, f'SC{j+1} FOV', fontsize=10, transform=bx_i.transAxes, va='top', color='white')
                    _text.set_bbox(dict(facecolor='orange', pad=0.25))
                    if self.checkerboard:
                        bx_i.pcolormesh(self._checkerboard_xx, self._checkerboard_yy, self._checkerboard, cmap='Reds', vmin=0, vmax=1, rasterized=True)
            else:
                for scatter_point in _scatter_points:
                    scatter_point.remove()
                for sc_label in _sc_labels:
                    sc_label.remove()
                for _image in _images:
                    _image.remove()
                try:
                    for _perimeter_plot in _perimeter_plots:
                        _perimeter_plot.remove()
                except TypeError:
                    del(_perimeter_plots)
            
            _scatter_points = []
            _sc_labels = []
            _images = []
            _perimeter_plots = []

            for j, _ephemeris in enumerate(np.moveaxis(self.ephemeris[1], -1, 0)):
                ephemeris_time_np = np.array(self.ephemeris[0], dtype='datetime64')
                ephemeris_time_dt = np.abs(ephemeris_time_np-np.datetime64(guide_time))
                closest_idt = np.argmin(ephemeris_time_dt)
        
                if ephemeris_time_dt[closest_idt] > ephemeris_time_np[1] - ephemeris_time_np[0]:
                    raise ValueError(
                        f"Time {guide_time} is too far from the nearest ephemeris timestamps:"
                        f"{self.ephemeris[0][closest_idt]}."
                        )
                
                lla = _ephemeris[closest_idt, :]
    
                _scatter_points.append(
                    self.ax.scatter(lla[1], lla[0], c='purple', s=200, marker=getmarker('camera'), transform=ccrs.PlateCarree())
                )
                _sc_labels.append(
                    self.ax.text(lla[1]+0.5, lla[0], f'SC{j+1}', color='white', fontsize=12, transform=ccrs.PlateCarree(), va='center')
                )

            if i == 0 and self.checkerboard:
                for bx_i in self.bx.flatten():
                    bx_i.pcolormesh(self._checkerboard_xx, self._checkerboard_yy, self._checkerboard, cmap='Reds', vmin=0, vmax=1, rasterized=True)

            images = self.get_image(guide_time)
            for j, (bx_i, image_i) in enumerate(zip(self.bx.flatten(), np.moveaxis(images, -1, 0))):
                _images.append(bx_i.imshow(
                    image_i, 
                    cmap='Greys_r',
                    vmin=color_bounds[0],
                    vmax=color_bounds[1],
                    origin='lower',
                    zorder=2,
                    ))

            lat_perimeter, lon_perimeter = self.get_mapped_fov(guide_time)
            for i in range(self.n_satellites):
                _perimiter_plot, = self.ax.plot(
                    lon_perimeter[:, i], 
                    lat_perimeter[:, i], 
                    ls='--', 
                    color='purple', 
                    lw=2, 
                    zorder=2.1,
                    transform=ccrs.PlateCarree(),
                    )
                _perimeter_plots.append(_perimiter_plot)
            yield guide_time, ax, bx, images, (lon_perimeter, lat_perimeter)
        return


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


def getmarker(mID):
    # TODO: Consider removing this function.
	symbol = fontawesome.icons[mID]
	fp = FontProperties(fname=pathlib.Path(__file__).parent / "Font Awesome 7 Free-Solid-900.otf")

	v, codes = TextToPath().get_text_path(fp, symbol)
	v = np.array(v)
	mean = np.mean([np.max(v,axis=0), np.min(v, axis=0)], axis=0)
	return Path(v-mean, codes, closed=False)


if __name__ == '__main__':

    import itertools
    from datetime import datetime

    import cartopy.crs
    import cartopy.feature as cfeature
    from asilib.mission import example_satellite
    import asilib
    import asilib.ose

    time_range = (datetime(2012, 2, 15, 8, 30), datetime(2012, 2, 15, 8, 45))
    aurora_alt = 110

    location_codes = [
        'FSIM',
        'FSMI',
        'ATHA',
        'TPAS',
        'GILL',
        ]
    
    asis = asilib.Imagers(
        [asilib.asi.themis(code, time_range=time_range, alt=aurora_alt) for code in location_codes]
        )

    # Create the CINEMA constellation ephemeris.
    orbit_parameter_tuple_type = namedtuple(
        'orbit_parameter_tuple_type', 
        ['mean_anomaly_deg', 'ltan_hours', 'alt_km']
        )
    in_track_separation_minutes = 5
    orbit_period_minutes = 95
    mean_anomaly_deg = 35
    sat_alt = 600
    delta_mean_anomaly_deg = 360*in_track_separation_minutes/orbit_period_minutes

    ltan_hours = [0.9, 1.9, 2.9]
    mean_anomalies = [
        mean_anomaly_deg+delta_mean_anomaly_deg, 
        mean_anomaly_deg, 
        mean_anomaly_deg-delta_mean_anomaly_deg
        ]
    constellation = {
        i:orbit_parameter_tuple_type(
            mean_anomaly_deg=mean_anomaly, 
            ltan_hours=ltan,
            alt_km=sat_alt,
            ) for i, (mean_anomaly, ltan) in enumerate(itertools.product(mean_anomalies, ltan_hours))
        }
    
    ephemeris = [None, None]
    for key, value in constellation.items():
        ephemeris_obj = example_satellite.Example_Satellite(
            cadence_s=0.5,
            time_range=time_range,
            mean_anomaly_deg=value.mean_anomaly_deg,
            ltan_hours=value.ltan_hours,
            altitude_km=value.alt_km,
        )
        sat_ephemeris = ephemeris_obj.ephemeris()
        if ephemeris[0] is None:
            ephemeris[0] = sat_ephemeris[0]
            ephemeris[1] = sat_ephemeris[1].reshape(*sat_ephemeris[1].shape, 1)
        else:
            ephemeris[1] = np.concatenate(
                (ephemeris[1], sat_ephemeris[1].reshape(*sat_ephemeris[1].shape, 1)), axis=2
                )

    ose = asilib.ose.OSE(asis, ephemeris, fov=(55, 65), pixel_resolution=(124, 124))

    fig = plt.figure(figsize=(4, 7.5))
    gs = gridspec.GridSpec(nrows=4, ncols=3, figure=fig, height_ratios=(3, 1, 1, 1))

    center = (
        np.mean(asis.lon_bounds), np.mean(asis.lat_bounds)
    )
    projection = cartopy.crs.Orthographic(
        central_longitude=center[0], 
        central_latitude=center[1]
    )

    ax = fig.add_subplot(gs[0, :], projection=projection)
    ax.add_feature(cfeature.LAND, color='grey')
    ax.add_feature(cfeature.OCEAN, color='cyan')
    ax.add_feature(cfeature.COASTLINE, edgecolor='k')
    ax.gridlines(linestyle=':')
    ax.set_global()
    ax.set_extent(
        (center[0]-20, center[0]+20, center[1]-11, center[1]+11), 
        crs=cartopy.crs.PlateCarree()
        )

    bx = np.nan*np.zeros((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            bx[i, j] = fig.add_subplot(gs[i+1, j])
            bx[i, j].set_aspect('equal')
            bx[i, j].xaxis.set_visible(False)
            bx[i, j].yaxis.set_visible(False)
    plt.suptitle(
        f'CINEMA OSE | fov={ose.fov} [deg]\n'
        f'alt={sat_alt} [km] | resolution={ose.pixel_resolution} [px]', 
        fontsize=12
        )
    plt.subplots_adjust(
        bottom=0.01, top=0.95, left=0.01, right=0.99, wspace=0.03, hspace=0.03
    )

    ose.animate_ose(ax=ax, bx=bx)