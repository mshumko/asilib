"""
Download, load, and plot the GPS CXR data from LANL. See the 
`README <https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10/gps_readme_v1.10.pdf>`_ for more information.
"""
from __future__ import annotations  # to support the -> List[Downloader] return type
import json
import calendar
import itertools
from typing import List
import pathlib
import urllib
import re
import warnings
from datetime import timedelta, datetime, date

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import astropy.time

import asilib
import asilib.download as download

gps_url = "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/"
download_dir = asilib.ASI_DATA_DIR / 'gps'


if not download_dir.exists():
	download_dir.mkdir(parents=True)
	print('Created directory for GPS data:', download_dir)


class GPS:
    def __init__(self, time_range:List[datetime], sc_ids=None, version='1.10', redownload=False, clip_date=True):
        self.time_range = time_range
        self.sc_ids = sc_ids
        self.version = version
        self.redownload = redownload
        self.clip_date = clip_date
        self.data = self._get_data()
        self.sc_id_0 = list(self.data.keys())[0]
        self.energies = self.data[self.sc_id_0]['electron_diff_flux_energy'][-1, :]
        pass

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
            The key in the gps data corresponding to the L-shell values. Can be one of 
            'L_LGM_T89IGRF'.
        dt_min: int
            The time cadence (in minutes) at which to sample the flux.
        min_samples: int
            The minimum number of samples required to compute the average flux at each time step.

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
        Plot the flux evolution from the gps data in a given L range every dt_min minutes.

        Parameters
        ----------
        energy: float
            The energy channel (in MeV) to plot.
        L_range: tuple of float
            The L-shell range over which to average the flux.
        L_key: str
            The key in the gps data corresponding to the L-shell values. Can be one of 
            'L_LGM_T89IGRF'.

        Returns
        -------
        pd.DataFrame
            The fluxes time index and energy channels as columns.
        
        """
        assert len(self.data.keys()) > 0, 'No GPS data.'

        _flux = pd.DataFrame(
            index=pd.date_range(
                self.data[self.sc_id_0]['time'][0].replace(second=0, microsecond=0), 
                self.data[self.sc_id_0]['time'][-1], 
                freq=f'{dt_min}min'
                ),
            columns = self.data[self.sc_id_0]['electron_diff_flux_energy'][-1, :]
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
        return _flux
    
    def _get_data(self):
        """
        Download or load GPS CXR data.
        """
        gps_weeks = _gps_weeks(time_range)

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
            
            if self.clip_date:
                idt = np.where(
                    (gps_data[_sc_id]['time'] >= time_range[0]) & 
                    (gps_data[_sc_id]['time'] < time_range[1])
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

    time_range = (datetime(2007, 2, 10), datetime(2007, 2, 17))
    vline_times = (datetime(2007, 2, 13, 12), datetime(2007, 2, 14, 13))
    L_range = [4.25, 4.75]
    dt_min=60*3
    # file_paths, spacecraft_ids = download_gps(date, version='1.10', redownload=False)
    _gps = GPS(time_range, version='1.10', redownload=False)
    ax = _gps.plot_avg_flux(
        energies=(0.12, 0.3, 0.6, 1.0, 2.0), 
        L_range=L_range, 
        dt_min=dt_min, 
        min_samples=5
        )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    if vline_times is not None:
        for vline_time in vline_times:
            ax.axvline(vline_time, c='k', ls=':')
    plt.legend()
    plt.show()