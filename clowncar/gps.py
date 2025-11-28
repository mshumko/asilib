"""
Download, load, and plot the GPS CXR data from LANL for clowncar. See the 
`README <https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10/gps_readme_v1.10.pdf>`_ for more information.
"""
from __future__ import annotations  # to support the -> List[Downloader] return type
import json
import calendar
from typing import List
import pathlib
import urllib
import IRBEM
import warnings
import pprint
from datetime import timedelta, datetime, date

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy_imported = True
except ImportError as err:
    # You can also get a ModuleNotFoundError if cartopy is not installed
    # (as compared to failed to import), but it is a subclass of ImportError.
    cartopy_imported = False
import matplotlib.pyplot as plt
import matplotlib.dates
import pandas as pd
import numpy as np
import requests
import astropy.time

import asilib
import asilib.download as download

R_E = 6378.137  # km

gps_url = "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/"
download_dir = asilib.ASI_DATA_DIR / 'gps'


if not download_dir.exists():
	download_dir.mkdir(parents=True)
	print('Created directory for GPS data:', download_dir)


class GPS:
    """
    Download, load, and analyze GPS Combined X-ray Dosimeter (CXD) data from LANL.
    It can handle multiple spacecraft data and provides methods for calculating and
    plotting average electron flux in specific L-shell ranges.

    Note: This code transforms the Geographic_Longitude variable from the (0 -> 360) 
    range to the (-180 -> 180) range.

    Parameters
    ----------
    time_range : List[datetime]
        The time range for which to analyze GPS data, provided as a list of two datetime objects
        [start_time, end_time].
    sc_ids : str or list of str, optional
        Spacecraft ID(s) to analyze (e.g., "ns72" or ["ns72", "ns73"]). If None, data from all
        available spacecraft is used.
    version : str, optional
        Version of the GPS data to use. 
    redownload : bool, optional
        If True, forces redownload of existing files.
    clip_date : bool, optional
        If True, clips the data to the specified time range.
    verbose : bool, optional
        If True, prints progress messages during data processing.
    energy : float, optional
        The energy channel (in MeV) pass to Clowncar.

    Attributes
    ----------
    data : dict
        Dictionary containing the loaded GPS data for each spacecraft.
    keys : list
        List of available data keys.
    energies : numpy.ndarray
        Array of energy channels in MeV.
    l_keys : list
        List of available L-shell parameter keys.

    Methods
    -------
    plot_avg_flux()
        Plot the flux evolution for specified energy channels and L-shell range.
    avg_flux()
        Calculate average flux in a given L-shell range at specified time intervals.

    Example
    -------
    >>> from datetime import datetime
    >>> import matplotlib.pyplot as plt
    >>> time_range = (datetime(2007, 2, 10), datetime(2007, 2, 17))
    >>> L_range = [4.25, 4.75]
    >>> gps = GPS(time_range, version='1.10')
    >>> ax = gps.plot_avg_flux(
    ...     energies=(0.12, 0.3, 0.6, 1.0, 2.0),
    ...     L_range=L_range,
    ...     dt_min=180,  # 3 hours
    ...     min_samples=5
    ... )
    >>> plt.legend()
    >>> plt.show()
    """
    def __init__(
            self, 
            time_range:List[datetime], 
            sc_ids=None, 
            version='1.10', 
            redownload=False, 
            clip_date=True, 
            verbose=False,
            energy=1
            ):
        self.time_range = time_range
        self.sc_ids = sc_ids
        self.version = version
        self.redownload = redownload
        self.clip_date = clip_date
        self.verbose = verbose
        self.energy = energy
        self.data = self._find_data()
        self.sc_id_0 = list(self.data.keys())[0]
        self.keys = self.data[self.sc_id_0].keys()

        self.energies = self.data[self.sc_id_0]['electron_diff_flux_energy'][-1, :]
        self._energy_idx = np.where(energy==self.energies)[0]
        assert len(self._energy_idx) == 1, (
            f'Energy channel {energy} is not in the valid {self.energies} list.'
            )
        self._energy_idx = self._energy_idx[0]
        self.l_keys = [key for key in self.keys if 'L_' in key]
        return
    
    def gps_footprint(self, alt=110, hemi_flag=0):
        """
        Compute magnetic footprint (lat, lon, alt) for an already-interpolated gps_dict.
        Add the footprint_lat, footprint_lon, and footprint_alt keys to gps_dict.

        Parameters
        ----------
        alt : float
            Altitude (in km) at which to compute the magnetic footprint.
        hemi_flag : int
            Hemisphere flag for IRBEM MagFields.find_foot_point method. The valid options are:
            0 - Same magnetic hemisphere as starting point
            1 - northern magnetic hemisphere
            -1 - southern magnetic hemisphere
            2 - opposite magnetic hemisphere from starting point.
        """        
        if not hasattr(self, 'mag_model'):
            # Initialize the magnetic field model.
            # This is a global variable so that it can be reused in multiple calls.
            self.mag_model = IRBEM.MagFields(kext='None')

        self.footprints = {}

        for sc_key in list(self.data.keys()):
            if self.verbose:
                print(f'Calculating footprints for GPS SC ID: {sc_key}...', end='\r')
            if 'interpolated_times' not in self.data[self.sc_id_0]:
                _times = self.data[sc_key]['time']
                n = len(_times)                
            else:
                _times = self.data[sc_key]['interpolated_times']
                n = len(_times)

            _all = np.zeros((n, 3), dtype=float)
            time_loc = pd.DataFrame(data={
                'time':_times, 
                'x1':(self.data[sc_key]['Rad_Re']-1)*R_E,
                'x2':self.data[sc_key]['Geographic_Latitude'], 
                'x3':self.data[sc_key]['Geographic_Longitude'],
                })
            
            # For running the T89 model.
            # kp = _get_kp(gps_dict[sc_key]['interpolated_times'])
            
            z = zip(time_loc['time'], time_loc['x1'], time_loc['x2'], time_loc['x3'])
            for i, (time, x1, x2, x3) in enumerate(z):
                X = {'Time':time, 'x1':x1, 'x2':x2, 'x3':x3}
                _all[i, :] = self.mag_model.find_foot_point(X, {}, alt, hemi_flag)['XFOOT']

            _all[_all == -1E31] = np.nan
            # Convert from (alt, lat, lon) to (lat, lon, alt)
            lla = np.roll(_all, shift=-1, axis=1)

            self.data[sc_key]['footprint_lat'] = lla[:, 0]
            self.data[sc_key]['footprint_lon'] = lla[:, 1]
            self.data[sc_key]['footprint_alt'] = lla[:, 2]

            self.footprints[sc_key] = {
                'lat': self.data[sc_key]['footprint_lat'],
                'lon': self.data[sc_key]['footprint_lon'],
                'alt': self.data[sc_key]['footprint_alt']
            }
        return self.data
    
    def interpolate_gps_loc(self, freq='3s'):
        """
        Interpolate the GPS location data to a regular time interval.

        Parameters
        ----------
        freq : str
            The frequency string for interpolation (e.g., '3s' for 3 seconds). It is passed
            directly to pandas.date_range(freq=freq) to create the interpolated time index.
        """
        interp_times = pd.date_range(*self.time_range, freq=freq)
        interp_times_numeric = matplotlib.dates.date2num(interp_times)

        for sc_key in self.data:
            if self.verbose:
                print(f'Interpolating GPS SC ID: {sc_key} posiiton...', end='\r')
            # Jumps across the anti-meridian or poles.
            lon_jumps = np.where(np.abs(np.diff(self.data[sc_key]['Geographic_Longitude'])) > 5)[0]
            lon_jump_start_times = self.data[sc_key]['time'][lon_jumps]
            lon_jump_end_times = self.data[sc_key]['time'][lon_jumps+1]

            self.data[sc_key]['interpolated_times'] = interp_times

            # Keep track of indices where we had to interpolate across jumps, and nan them out.
            interpolated_jump_indices = np.array([], dtype=int)
            for start_time, end_time in zip(lon_jump_start_times, lon_jump_end_times):
                idt = np.where(
                    (self.data[sc_key]['interpolated_times'] >= start_time) &
                    (self.data[sc_key]['interpolated_times'] <= end_time)
                    )[0]
                interpolated_jump_indices = np.concatenate((interpolated_jump_indices, idt))
            for llr_key in ['Geographic_Latitude', 'Geographic_Longitude', 'Rad_Re']:
                self.data[sc_key][llr_key] = np.interp(
                    interp_times_numeric,
                    matplotlib.dates.date2num(self.data[sc_key]['time']),
                    self.data[sc_key][llr_key],
                    left=np.nan,
                    right=np.nan,
                )
                if interpolated_jump_indices.shape[0] > 0:
                    self.data[sc_key][llr_key][interpolated_jump_indices] = np.nan
        return self.data

    def __call__(self, time:datetime, ax=None, time_tol_min=4):
        """
        This is the method that talks to Clowncar. This method returns the GPS locations and 
        footprints, interpolated or non-interpolated, if they are within time_tol of the input
        time. If ax is provided with a Cartopy projection, the GPS units outside the ax's FOV
        are excluded.

        Note: The fluxes are not interpolated. Rather, they correspond to the nearest time stamp.

        Parameters
        ----------
        time : datetime
            The time at which to get the GPS data.
        ax: matplotlib.axes.Axes, optional
            The axes on which to check if the GPS unit footprints are in the ax's FOV.
        time_tol_min : int, optional
            The time tolerance (in minutes) for matching GPS data to the input time.        
        """
        gps_data = {}

        for sc_key in self.data:
            if 'interpolated_time' in self.data[self.sc_id_0]:
                _time_key = 'interpolated_time'
                min_idx = np.argmin(np.abs(
                    matplotlib.dates.date2num(self.data[sc_key]['interpolated_time']) - matplotlib.dates.date2num(time)
                ))
            else:
                _time_key = 'time'
                min_idx = np.argmin(np.abs(
                    matplotlib.dates.date2num(self.data[sc_key]['time']) - matplotlib.dates.date2num(time)
                ))
            if np.abs((self.data[sc_key][_time_key][min_idx] - time).total_seconds()) <= 60*time_tol_min:
                gps_data[sc_key] = {}
                gps_keys = [_time_key, 'Geographic_Latitude', 'Geographic_Longitude', 'Rad_Re']
                general_keys = [_time_key, 'sc_lat', 'sc_lon', 'sc_rad']
                for gps_key, general_key in zip(gps_keys, general_keys):
                    gps_data[sc_key][general_key] = self.data[sc_key][gps_key][min_idx]

                if hasattr(self, 'footprints'):
                    for key in ['footprint_lat', 'footprint_lon', 'footprint_alt']:
                        gps_data[sc_key][key] = self.data[sc_key][key][min_idx]

            # We are currently not interpolating fluxes, so only use the non-interpolated time.
            min_idx_flux = np.argmin(np.abs(
                matplotlib.dates.date2num(self.data[sc_key]['time']) - matplotlib.dates.date2num(time)
            ))
            if np.abs((self.data[sc_key]['time'][min_idx_flux] - time).total_seconds()) <= 60*time_tol_min:
                gps_data[sc_key]['flux'] = self.data[sc_key]['electron_diff_flux'][min_idx_flux, self._energy_idx]

        if ax is not None:
            if 'footprint_lat' not in self.data[self.sc_id_0]:
                raise ValueError(
                    "GPS footprint data not found. Please run the gps_footprint() method "
                    "before calling this method with an ax argument."
                )
            if cartopy_imported:
                ax_extent = ax.get_extent(crs=ccrs.PlateCarree())
            else:
                ax_extent = (*ax.get_xlim(), *ax.get_ylim())

            gps_units_not_in_view = []
            for sc_key, data in gps_data.items():
                if np.isnan(data['footprint_lat']) or np.isnan(data['footprint_lon']):
                    gps_units_not_in_view.append(sc_key)
                    continue
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
        if len(list(gps_data.keys())) == 0:
            return {}  # Interpolated time stamps before the first GPS time stamp
        gps_data['sc_id'] = list(gps_data.keys())
        return gps_data
    
    def cc_footprint_config(self, kwargs={}):
        """
        Configure the GPS footprint track for clowncar. These parameters are passed directly into
        plt.plot(lon, lat, **kwargs), so see the matplotlib documentation for valid options.
        """
        self._cc_footprint_params = kwargs
        self._cc_footprint_params.setdefault('linestyle', ':')
        self._cc_footprint_params.setdefault('color', 'r')
        return
    
    def cc_marker_config(self, kwargs={}):
        """
        Configure the GPS marker for clowncar. These parameters are passed directly into
        plt.scatter(lon, lat, **kwargs), so see the matplotlib documentation for valid options.

        The default options are: 
        - marker='fontawesome-satellite', 
        - norm=LogNorm(1E4, 1E7),
        - cmap='plasma',
        - s=1_500,
        """
        self._cc_marker_params = kwargs
        self._cc_marker_params.setdefault('marker', 'fontawesome-satellite')
        self._cc_marker_params.setdefault('norm', matplotlib.colors.LogNorm(1E4, 1E7))
        self._cc_marker_params.setdefault('cmap', plt.get_cmap('plasma'))
        self._cc_marker_params.setdefault('s', 1_500)
        self._cc_marker_params.setdefault('edgecolors', 'k')
        # self._cc_marker_params.setdefault('c', self._flux_value())  # TODO: Define this method.
        return
    
    def cc_marker_label_config(self, kwargs={}):
        """
        Configure the GPS marker label for clowncar. These parameters are passed directly into
        plt.text(lon, lat, label, **kwargs), so see the matplotlib documentation for valid options.
        """
        self._cc_marker_label_params = kwargs
        self._cc_marker_label_params.setdefault('fontsize', 30)
        self._cc_marker_label_params.setdefault('color', 'orange')
        self._cc_marker_label_params.setdefault('lon_offset', 1)
        self._cc_marker_label_params.setdefault('lat_offset', 0)
        return

    def plot_avg_flux(
            self, 
            ax=None,
            energies=(0.12, 0.3, 0.6, 1.0, 2.0), 
            L_range=(5, 6), 
            L_key='L_LGM_T89IGRF', 
            dt_min=15, 
            min_samples=5, 
            labels=True
            ):
        """
        Plot the flux evolution from the gps data in a given L range every dt_min minutes.

        Parameters
        ----------
        energy: float
            The energy channel (in MeV) to plot.
        L_range: tuple of float
            The L-shell range over which to average the flux.
        L_key: str
            The key in the gps data corresponding to the L-shell values. See self.l_keys for valid options.
        dt_min: int
            The time cadence (in minutes) at which to sample the flux.
        min_samples: int
            The minimum number of samples required to compute the average flux at each time step. If the number
            of samples is less than this value, NaN will be assigned for that time step.

        Returns
        -------
        None
            Displays a plot of the average flux over time.
        
        """
        for _energy in energies:
            energy_idx = np.where(_energy==self.energies)[0]
            assert len(energy_idx) == 1, \
                (f'Energy channel {_energy} is not in the valid {self.energies} list.')
            self.data[self.sc_id_0]['energy_channel_idx'] = energy_idx[0]
        _flux = self.avg_flux(
            L_range=L_range, 
            L_key=L_key, 
            dt_min=dt_min, 
            min_samples=min_samples
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        for _energy in energies:
            ax.plot(_flux.index, _flux[_energy], '-o', label=f'{_energy} MeV', markersize=2)
        if labels:
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux (1/cm^2-s-sr-MeV)')
            ax.set_title(f'GPS Electron Flux | L={L_range}')
            ax.set_yscale('log')
        return ax

    def avg_flux(self, L_range=(5, 6), L_key='L_LGM_T89IGRF', dt_min=15, min_samples=5):
        """
        Average the GPS CXD fluxes in a given L range every dt_min minutes.

        Parameters
        ----------
        energy: float
            The energy channel (in MeV) to plot.
        L_range: tuple of float
            The L-shell range over which to average the flux.
        L_key: str
            The key in the gps data corresponding to the L-shell values. Can be one of 
            'L_LGM_T89IGRF'.
        dt_min: int
            The time cadence (in minutes) at which to sample the flux.
        min_samples: int
            The minimum number of samples required to compute the average flux at each time step. If the number
            of samples is less than this value, NaN will be assigned for that time step.

        Returns
        -------
        pd.DataFrame
            The fluxes time index and energy channels as columns.
        
        """
        assert len(self.data.keys()) > 0, 'No GPS data.'
        assert L_key in self.keys, f'L_key {L_key} not in GPS data keys: {self.keys}.'

        _flux = pd.DataFrame(
            index=pd.date_range(
                self.data[self.sc_id_0]['time'][0].replace(second=0, microsecond=0), 
                self.data[self.sc_id_0]['time'][-1].replace(second=0, microsecond=0)+pd.Timedelta(minutes=dt_min), 
                freq=f'{dt_min}min'
                ),
            columns = self.data[self.sc_id_0]['electron_diff_flux_energy'][-1, :],
            dtype=float
                )
        for start_time, end_time in zip(_flux.index[:-1], _flux.index[1:]):
            flux_vals = []
            n = 0
            for sc_id in self.data.keys():
                idt = np.where(
                    (self.data[sc_id]['time'] >= start_time) & 
                    (self.data[sc_id]['time'] < end_time) & 
                    (self.data[sc_id][L_key] >= L_range[0]) & 
                    (self.data[sc_id][L_key] < L_range[1])
                    # TODO: Add the quality of fit test here.
                )[0]
                n += len(idt)
                if len(idt) > 0:
                    flux_vals.append(np.nanmean(self.data[sc_id]['electron_diff_flux'][idt, :], axis=0))

            if n < min_samples:
                _flux.loc[start_time] = np.nan
                continue
            
            if len(flux_vals) == 1:
                _flux.loc[start_time] = flux_vals
            elif len(flux_vals) > 1:
                _flux.loc[start_time] = np.nanmean(np.array(flux_vals), axis=0)
            else:
                raise ValueError('Not supposed to get here.')
        return _flux.iloc[:-1, :]
    
    def _find_data(self):
        """
        Download or load GPS CXR data.
        """
        gps_weeks = _gps_weeks(self.time_range)

        file_paths = []
        spacecraft_ids = []
        for _gps_week in gps_weeks:
            _file_paths, _spacecraft_ids = self._download_gps(
                _gps_week, sc_id=self.sc_ids, version=self.version, redownload=self.redownload
                )
            for (_file_path, _spacecraft_id) in zip(_file_paths, _spacecraft_ids):
                file_paths.append(_file_path)
                spacecraft_ids.append(_spacecraft_id)

        gps_data = {}
        loaded_sc = []
        for file_path, _sc_id in zip(file_paths, spacecraft_ids):
            if _sc_id not in loaded_sc:
                gps_data[_sc_id] = readJSONheadedASCII(file_path)
                gps_data[_sc_id].attrs['file_path'] = [str(file_path)]
                gps_data[_sc_id]['time'] = astropy.time.Time(
                    gps_data[_sc_id]['decimal_year'], format='decimalyear'
                    ).datetime
                # gps_data[_sc_id]['time'] = Dec2Datetime(gps_data[_sc_id])
                loaded_sc.append(_sc_id)
            else:
                _data = readJSONheadedASCII(file_path)
                for key, vals in _data.items():
                    if key == 'time':
                        continue
                    gps_data[_sc_id][key] = np.concatenate(
                        (gps_data[_sc_id][key], vals), axis=0
                    )
                gps_data[_sc_id].attrs['file_path'].append(str(file_path))
                gps_data[_sc_id]['time'] = np.append(
                    gps_data[_sc_id]['time'],
                    astropy.time.Time(_data['decimal_year'], format='decimalyear').datetime
                    )
                gps_data[_sc_id]['Geographic_Longitude'] = np.mod(
                    gps_data[_sc_id]['Geographic_Longitude'] + 180, 360
                    ) - 180
            
            if self.clip_date:
                idt = np.where(
                    (gps_data[_sc_id]['time'] >= self.time_range[0]) & 
                    (gps_data[_sc_id]['time'] < self.time_range[1])
                )[0]

                for key in gps_data[_sc_id].keys():
                    if isinstance(gps_data[_sc_id][key], np.ndarray):
                        gps_data[_sc_id][key] = gps_data[_sc_id][key][idt]
            
            # Some time stamp differences are either 0 or are negative ¯\_(ツ)_/¯.
            dt_seconds = np.array([dt.total_seconds() for dt in np.diff(gps_data[_sc_id]['time'])])
            bad_idtx = np.where(dt_seconds < 0)[0]
            if len(bad_idtx) > 0:
                warnings.warn(
                    f"Negative time differences found in GPS data for SC ID {_sc_id}. "
                    f"Indices: {bad_idtx}. This may indicate a problem with the data."
                )
        return gps_data

    def _download_gps(self, date, sc_id=None, version='1.10', redownload=False):
        """
        Download GPS data from all available satellites for a given date and data version.

        Parameters
        ----------
        date: datetime
            The date for which to download the weekly GPS data.
        sc_id: str or list of str (optional)
            The spacecraft ID(s) to download data. If None, it will download data for all available 
            spacecraft IDs. Valid example inputs include: "ns72", ["ns72", "ns73"].
        version: str (optional)
            The version of the GPS data to download.
        redownload: bool (optional)
            If True, will redownload existing files.

        Returns
        -------
        file_paths: list of pathlib.Path
            The paths to the downloaded files.
        spacecraft_ids: list of str
            The spacecraft IDs corresponding to the downloaded files.
        """
        if sc_id is None:
            _gps_ids = get_gps_ids(version=version) # Get all spacecraft IDs (only a subset will have data for the date).
        else:
            _gps_ids = [sc_id] if isinstance(sc_id, str) else sc_id  # Ensure sc_id is a list.

        if not isinstance(_gps_ids, list):
            raise TypeError(f"sc_id must be a list or a string, got {type(_gps_ids)}.")
        
        file_paths = []
        spacecraft_ids = []

        for _gps_id in _gps_ids:
            file_name = f"{_gps_id}_{date.strftime('%y%m%d')}_v{version}.ascii"
        
            if pathlib.Path(download_dir / file_name).exists() and not redownload:
                file_paths.append(download_dir / file_name)
                spacecraft_ids.append(_gps_id)
                continue

            _url = urllib.parse.urljoin(
                gps_url, 
                f"version_v{version}/{_gps_id}/"+file_name, 
                allow_fragments=True
                )
            
            request = requests.head(_url, timeout=10)
            try:
                request.raise_for_status()
            except requests.HTTPError as e:
                if request.status_code == 404:
                    continue
                else:
                    raise e

            _downloader = download.Downloader(_url, download_dir=download_dir)
            download_path = _downloader.download(redownload=redownload)
            
            file_paths.append(download_path)
            spacecraft_ids.append(_gps_id)   
        return file_paths, spacecraft_ids

    def __str__(self) -> str:
        s = (
            f'GPS object for times between {self.time_range} containing data '
            f'from {self.data.keys()} satellites.'
        )
        return s
    

def Dec2Datetime(gps_dict):
    """
    Convert year and decimal day to datetime (yuck).
    """
    Y = gps_dict['year'] #integer decimal_year
    D = gps_dict['decimal_day']

    N = len(Y)
    Q = np.zeros(N)
    for n in range(N):
        if (calendar.isleap(Y[n])):
            nod = 366.0
        else:
            nod = 365.0
        Q[n] = Y[n] + (D[n]-1)/nod
    E = np.min(np.abs(np.diff(Q)))

    if (E < 1.0e-10):
        warnings.warn(f"Holy shit, this is bad! {E=}")
    xTs = astropy.time.Time(Q,format='decimalyear')
    return xTs.datetime

def readJSONheadedASCII(file_path):
    """
    My simple implementation of spacepy.datamodel.readJSONheadedASCII that
    is specific for FIREBIRD-II data. You may use this if you can't install 
    spacepy for whatever reason.
    """
    # Read in the JSON header.
    header_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_list.append(line[1:])
            else:
                raw_data_str = f.readlines()

    # Massage the header
    clean_header_str = ''.join(header_list).replace('\n', '')
    parsed_header = json.loads(clean_header_str)
    # Massage the data
    raw_data_str = [row.replace('\n', '') for row in raw_data_str]
    # Parse times
    times_str = [row.split()[0] for row in raw_data_str]
    # Parse the other columns
    data_converted = np.array([row.split()[1:] for row in raw_data_str]).astype(float)

    data = attrs_dict()
    for key in parsed_header:
        key_header = parsed_header[key]
        data.attrs[key] = key_header  # Save the attribute data.

        # Header key that correspond to columns
        if isinstance(key_header, dict):
            if len(key_header['DIMENSION']) != 1:
                raise NotImplementedError(
                    "readJSONheadedASCII doesn't implement columns with more than "
                    f"1 multidimensional. Got {key_header['DIMENSION']}."
                    )
            start_column = key_header['START_COLUMN']-1
            end_column = key_header['START_COLUMN']-1+key_header['DIMENSION'][0]
            if key_header['DIMENSION'][0] == 1:
                data[key] = data_converted[:, start_column]
            else:
                data[key] = data_converted[:, start_column:end_column]
        else:
            data.attrs[key] = key_header
    return data

class attrs_dict(dict):
    """
    Expand Python's dict class to include an attr attribute dictionary.
    
    Code credit goes to Matt Anderson: https://stackoverflow.com/a/2390997.
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.attrs = {}
        return
    
    def __str__(self):
        return super().__str__() + f"\nattrs: {self.attrs}."


def get_gps_ids(version='1.10'):
    """
    Look-up the GPS spacecraft IDs online.
    """
    sc_ids_file_path = pathlib.Path(download_dir / 'sc_ids.txt')
    if sc_ids_file_path.exists():
        sc_ids = sc_ids_file_path.read_text()
        return sc_ids.split(',')
    
    _url = gps_url + f"/version_v{version}/"
    _folder_contents = download.Downloader(_url).ls("ns")
    sc_ids = [_folder_content.url.split("/")[-2] for _folder_content in _folder_contents]
    assert len(sc_ids) > 0, f"No SC IDs found at {_url}."
    for sc_id in sc_ids:
        assert sc_id.startswith("ns") and sc_id[2:].isdigit(), f"Invalid SC ID: {sc_id} at {_url}."
          
    sc_ids_file_path.write_text(','.join(sc_ids))
    return sc_ids

def _gps_weeks(_dates):
    """
    GPS files are produced at a weekly cadence, starting on a Sunday. See the online 
    README for more details.
    """
    if not isinstance(_dates, (list,tuple,np.ndarray)):
        idx = (_dates.weekday()+1) % 7
        if isinstance(_dates, date):
            return [_dates - timedelta(days=idx)]
        else:
            return [(_dates - timedelta(days=idx)).date()]
    else:
        assert _dates[0] < _dates[1], 'The times must be ordered.'
        _sundays = []
        _current_sunday = _dates[0] - timedelta(days=(_dates[0].weekday()+1) % 7)
        _current_sunday = _current_sunday.replace(
            hour=0, 
            minute=0, 
            second=0,
            microsecond=0
            )
        while _current_sunday <= _dates[1]:
            _sundays.append(_current_sunday)
            _current_sunday += timedelta(days=7)
        return _sundays

if __name__ == "__main__":
    from datetime import timedelta, datetime
    import matplotlib.ticker

    time_range = (datetime(2021, 11, 4, 6, 30), datetime(2021, 11, 4, 7, 30))
    # vline_times = (datetime(2007, 2, 13, 12), datetime(2007, 2, 14, 13))
    L_range = [4.25, 4.75]
    dt_min=60*3
    # file_paths, spacecraft_ids = download_gps(date, version='1.10', redownload=False)
    _gps = GPS(time_range, version='1.10', redownload=False, verbose=True)
    _gps.interpolate_gps_loc(freq='3s')
    _gps.gps_footprint(alt=110, hemi_flag=0)
    _gps(time_range[0]+timedelta(minutes=10))
    pass

    # ax = _gps.plot_avg_flux(
    #     energies=(0.12, 0.3, 0.6, 1.0, 2.0), 
    #     L_range=L_range, 
    #     dt_min=dt_min, 
    #     min_samples=5
    #     )
    # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    # if vline_times is not None:
    #     for vline_time in vline_times:
    #         ax.axvline(vline_time, c='k', ls=':')
    # plt.legend()
    # plt.show()