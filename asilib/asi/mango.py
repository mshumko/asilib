"""
The Mid-latitude All-sky-imaging Network for Geophysical Observations (MANGO) employs a combination of two powerful optical techniques used to observe the dynamics of Earth's upper atmosphere: wide-field imaging and high-resolution spectral interferometry. Both techniques observe the naturally occurring airglow emissions produced in the upper atmosphere at 630.0- and 557.7-nm wavelengths. Instruments are deployed to sites across the continental United States, providing the capability to make measurements spanning mid to sub-auroral latitudes. The current instrument suite in MANGO has six all-sky imagers (ASIs) observing the 630.0-nm emission (integrated between ~200 and 400 km altitude), six ASIs observing the 557.7-nm emission (integrated between ~90 and 100 km altitude), and four Fabry-Perot interferometers measuring neutral winds and temperature at these wavelengths. The deployment of additional imagers is planned. The network makes unprecedented observations of the nighttime thermosphere-ionosphere dynamics with the expanded field-of-view provided by the distributed network of instruments. 

Instrument paper: https://doi.org/10.1029/2023JA031589
"""
from datetime import datetime, timedelta, timezone
import re
import warnings
import pathlib
import copy
import os
from typing import Tuple, Iterable, List, Union

import requests
import matplotlib.colors
import pandas as pd
import h5py
import numpy as np

import asilib
import asilib.map
import asilib.utils as utils
import asilib.download as download

base_url = 'https://data.mangonetwork.org/data/transport/mango/archive/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'mango'


def mango(
    location_code: str,
    channel: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    acknowledge: bool = True,
    redownload: bool = False,
    missing_ok: bool = True,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance with the MANGO ASI images and skymaps.

    The imaging data are obtained through the MANGO network and operated by SRI with 
    support from US National Science Foundation award AGS-1933013. Please reach out
    to the PI Asti Bhatt (asti.bhatt@sri.com) before using the data for publication.
    Cite the `instrument paper <https://doi.org/10.1029/2023JA031589>`_ for the 
    description of the MANGO network.

    For more information see: https://www.mangonetwork.org/

    Parameters
    ----------
    location_code: str
        The three-letter ASI location code.
    channel: str
        The color channel. Could be "redline", "greenline", "r", or "g". Case insensitive. 
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    acknowledge: bool
        If True, prints the acknowledgment statement for MANGO. 
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Examples
    --------
    # Animate a MANGO map with a SAR arc, as well as the SymH index.
    >>> # You will need to install cdasws to run this example (python -m pip install cdasws)
    >>> from datetime import datetime, timedelta, timezone
    >>> 
    >>> import cdasws
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.dates
    >>> import pandas as pd
    >>> import asilib.asi
    >>> 
    >>> time_range=(datetime(2021, 11, 4, 1, 0), datetime(2021, 11, 4, 12, 24))
    >>> location_code='CFS'
    >>> asi = asilib.asi.mango(location_code, 'redline', time_range=time_range)
    >>> fig = plt.figure(layout='constrained', figsize=(6, 6.5))
    >>> gs = matplotlib.gridspec.GridSpec(2, 1, fig, height_ratios=(3, 1))
    >>> ax = asilib.map.create_map(lat_bounds=(30, 45), lon_bounds=(-125, -100), fig_ax=(fig, gs[0]))
    >>> bx = fig.add_subplot(gs[1])
    >>> bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    >>> gen = asi.animate_map_gen(ax=ax, asi_label=True, overwrite=True)
    >>>
    >>> cdas = cdasws.CdasWs()
    >>> time_range = cdasws.TimeInterval(
    >>>     datetime.fromisoformat(str(time_range[0]-timedelta(days=0.5))).replace(tzinfo=timezone.utc),
    >>>     datetime.fromisoformat(str(time_range[1]+timedelta(days=0.5))).replace(tzinfo=timezone.utc)
    >>>     )
    >>> _, data = cdas.get_data(
    >>>                 'OMNI_HRO_5MIN', ['SYM_H'], time_range
    >>>                 )
    >>> symh = pd.DataFrame(index=data['SYM_H'].Epoch.data, data={'SYM_H':data['SYM_H']})
    >>> bx.plot(symh.index, symh['SYM_H'], c='k')
    >>> bx.set(xlabel='Time [HH:MM]', ylabel='Sym-H [nT]')
    >>>
    >>> for image_time, image, _, im in gen:
    >>>     # Add your code that modifies each image here...
    >>>     # To demonstrate, lets annotate each frame with the timestamp.
    >>>     # We will need to delete the prior text object, otherwise the current one
    >>>     # will overplot on the prior one---clean up after yourself!
    >>>     if 'text_obj' in locals():
    >>>         # text_obj.remove()  # noqa: F821
    >>>         _time_guide.remove()  # noqa: F821
    >>>     text_obj = plt.suptitle(f'MANGO-{location_code} | {image_time:%F %T}', fontsize=15)
    >>>     _time_guide = bx.axvline(image_time, c='k', ls='--')  
    
    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A MANGO ASI instance with the time stamps, images, skymaps, and metadata.
    """
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_image_dir = local_base_dir / 'images' / location_code.lower()

    channel = channel.lower()
    assert channel[0] in ['r', 'g'], (f"{channel} is an invalid MANGO color channel. "
                                      f"Try either 'redline' or 'greenline'.")
    if channel[0] == 'r':
        channel = 'redline'
    else:
        channel = 'greenline'

    mango_site_info = mango_info()
    mango_site_info = mango_site_info[
        mango_site_info['location_code'] == location_code.upper()
        ].reset_index()
    assert mango_site_info.shape[0] == 1, (
        f"MANGO-{location_code.upper()} is an invalid location. Try one of these: "
        f"{list(mango_info()['location_code'])}.")

    file_paths = _get_image_files(
        location_code,
        channel,
        time,
        time_range,
        base_url,
        local_image_dir,
        redownload,
        missing_ok,
    )

    start_times = len(file_paths) * [None]
    end_times = len(file_paths) * [None]
    for i, file_path in enumerate(file_paths):
        date_match = re.search(r'\d{8}', file_path.name)
        start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d')
        end_times[i] = start_times[i] + timedelta(days=1)
    file_info = {
        'path': file_paths,
        'start_time': start_times,
        'end_time': end_times,
        'loader': _load_h5,
    }
    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time

    mango_meta = _load_h5_meta(file_paths[0])
    
    meta = {
        'array': 'MANGO',
        'channel':channel,
        'location': location_code.upper(),
        'lat': mango_site_info.loc[0, 'latitude'],
        'lon': mango_site_info.loc[0, 'longitude'],
        'alt': None,
        'cadence': mango_meta['cadence'],
        'resolution': mango_meta['resolution'],
        'acknowledgment': (
            'The imaging data are obtained through the MANGO network and operated by SRI with '
            'support from US National Science Foundation award AGS-1933013. Please reach out '
            'to the PI Asti Bhatt (asti.bhatt@sri.com) before using the data for publication. '
            'Cite https://doi.org/10.1029/2023JA031589 for the description of the MANGO network.'
            )
    }

    plot_settings = {
        'color_map': matplotlib.colors.LinearSegmentedColormap.from_list(
            'black_to_red', ['k', channel[0]]
            ),
        'color_bounds':(50, 250)
    }

    skymap = {
        'lat':mango_meta['lat'],
        'lon':mango_meta['lon'],
        'az':mango_meta['az'],
        'el':mango_meta['el'],
        'alt':mango_meta['alt'],
        'path':mango_meta['path']
    }
    if acknowledge and ('mango' not in asilib.config['ACKNOWLEDGED_ASIS']):
        print(meta['acknowledgment'])
        asilib.config['ACKNOWLEDGED_ASIS'].append('mango')
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def mango_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the MANGO ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of THEMIS imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'MANGO']
    return df.reset_index(drop=True)

def _get_image_files(
    location_code: str,
    channel:str,
    time: datetime,
    time_range: Iterable[datetime],
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
    missing_ok: bool,
    ) -> List[pathlib.Path]:
    """
    Find MANGO image files either locally or download them from the internet.

    Parameters
    ----------
    location_code:str
        The three-letter ASI location code.
    channel: str
        The color channel. Could be "redline", "greenline", "r", or "g".
    time: datetime.datetime
        Time to download one file. Either time or time_range must be specified,
        but not both.
    time_range: Iterable[datetime]
        An iterable with a start and end time. Either time or time_range must be
        specified, but not both.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Download data even if the file is found locally. This is useful if data
        is corrupt.
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).

    Returns
    -------
    list(pathlib.Path)
        Local paths to each h5 file that was successfully found.
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')
    
    if redownload:
        # Option 1/4: Download one minute of data regardless if it is already saved
        if time is not None:
            return [
                _download_one_file(
                    location_code, channel, time, base_url, save_dir, redownload
                        )
            ]

        # Option 2/4: Download the data in time range regardless if it is already saved.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='days')
            file_paths = []
            for file_time in file_times:
                try:
                    file_paths.append(
                        _download_one_file(
                            location_code, channel, file_time, base_url, save_dir, redownload
                        )
                    )
                except (FileNotFoundError, AssertionError) as err:
                    if missing_ok and (
                        ('does not contain any hyper references containing' in str(err))
                        or ('Only one href is allowed' in str(err))
                    ):
                        continue
                    raise
            return file_paths
    else:
        # Option 3/4: Download one minute of data if it is not already saved.
        if time is not None:
            file_search_str = f'mango-{location_code.lower()}-{channel}-level1-{time:%Y%m%d}.hdf5'
            local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
            if len(local_file_paths) == 1:
                return local_file_paths
            else:
                return [
                    _download_one_file(
                        location_code, channel, time, base_url, save_dir, redownload
                    )
                ]

        # Option 4/4: Download the data in time range if they don't exist locally.
        elif time_range is not None:
            time_range = utils.validate_time_range(time_range)
            file_times = utils.get_filename_times(time_range, dt='days')
            file_paths = []
            for file_time in file_times:
                file_search_str = (
                    f'mango-{location_code.lower()}-{channel}-level1-{file_time:%Y%m%d}.hdf5'
                )
                local_file_paths = list(pathlib.Path(save_dir).rglob(file_search_str))
                if len(local_file_paths) == 1:
                    file_paths.append(local_file_paths[0])
                else:
                    try:
                        file_paths.append(
                            _download_one_file(
                                location_code, channel, file_time, base_url, save_dir, redownload
                            )
                        )
                    except (FileNotFoundError, AssertionError, requests.exceptions.HTTPError) as err:
                        if missing_ok and (
                            ('does not contain any hyper references containing' in str(err)) or
                            ('Only one href is allowed' in str(err)) or
                            ('404 Client Error: Not Found for url:' in str(err))
                        ):
                            continue
                        raise
            if missing_ok and len(file_paths) == 0:
                if time_range is not None:
                    warnings.warn(
                        f'No local or online image files found for MANGO-{location_code.upper()} '
                        f'for {time_range=}.'
                        )
                else:
                    warnings.warn(
                        f'No local or online image files found for MANGO-{location_code.upper()} '
                        f'for {time=}.'
                        )
            return file_paths
    return

def _download_one_file(
    location_code: str,
    channel:str,
    time: datetime,
    base_url: str,
    save_dir: Union[str, pathlib.Path],
    redownload: bool,
) -> pathlib.Path:
    """
    Download one h5 file.

    Parameters
    ----------
    location_code: str
        The ASI three-letter location code.
    channel: str
        The ASI channel
    time: str or datetime.datetime
        A time to look for the ASI data at.
    base_url: str
        The starting URL to search for file.
    save_dir: str or pathlib.Path
        The parent directory where to save the data to.
    redownload: bool
        Will redownload an existing file.

    Returns
    -------
    pathlib.Path
        Local path to file.
    """
    start_url = base_url + f'{location_code.lower()}/{channel}/level1/{time.year}/{time:%j}/'
    file_search_str = f'mango-{location_code.lower()}-{channel}-level1-{time:%Y%m%d}.hdf5'
    d = download.Downloader(start_url)
    matched_downloaders = d.ls(file_search_str)
    assert len(matched_downloaders) == 1
    return matched_downloaders[0].download(save_dir, redownload=redownload)


def _load_h5(file_path):
    with h5py.File(file_path, 'r') as file:
        # Assigning timezone.utc and immediately removing it is necessary to correctly convert 
        # to the UTC datetime objects. 
        start_exposure_times = np.array([
            datetime.fromtimestamp(ti, tz=timezone.utc).replace(tzinfo=None) 
            for ti in file['UnixTime'][0, :]
            ])
        images = file['ImageData'][...].astype(float)
        images[np.where(images == 0)] = np.nan
        return start_exposure_times, images

def _load_h5_meta(file_path):
    with h5py.File(file_path, 'r') as file:
        dt = file['UnixTime'][0, 1:]-file['UnixTime'][0, :-1]
        assert np.all(dt-dt[0] < dt), (
            f'There is a timestamp jump in file {file_path}. {min(dt)}<dt<{max(dt)}.'
            )
        meta_dict = {
            'lon':file['Longitude'][:], 
            'lat':file['Latitude'][:],
            'az':file['Azimuth'][:],
            'el':file['Elevation'][:],
            'alt':None,  # TODO ask Leslie what altitude they mapped to.
            'resolution':file['ImageData'].shape[1:],
            'cadence':dt[0],
            'path':file_path
        }
        return meta_dict
    
if __name__ == '__main__':
    import cdasws
    import matplotlib.pyplot as plt
    import matplotlib.dates

    time_range=(datetime(2021, 11, 4, 1, 0), datetime(2021, 11, 4, 12, 24))
    location_code='CFS'
    asi = mango(location_code, 'redline', time_range=time_range)

    fig = plt.figure(layout='constrained', figsize=(6, 6.5))
    gs = matplotlib.gridspec.GridSpec(2, 1, fig, height_ratios=(3, 1))
    ax = asilib.map.create_map(lat_bounds=(30, 45), lon_bounds=(-125, -100), fig_ax=(fig, gs[0]))
    bx = fig.add_subplot(gs[1])
    bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    gen = asi.animate_map_gen(ax=ax, asi_label=True, overwrite=True)

    cdas = cdasws.CdasWs()
    time_range = cdasws.TimeInterval(
        datetime.fromisoformat(str(time_range[0]-timedelta(days=0.5))).replace(tzinfo=timezone.utc), 
        datetime.fromisoformat(str(time_range[1]+timedelta(days=0.5))).replace(tzinfo=timezone.utc)
        )
    _, data = cdas.get_data(
                    'OMNI_HRO_5MIN', ['SYM_H'], time_range
                    )
    symh = pd.DataFrame(index=data['SYM_H'].Epoch.data, data={'SYM_H':data['SYM_H']})
    bx.plot(symh.index, symh['SYM_H'], c='k')
    bx.set(xlabel='Time [HH:MM]', ylabel='Sym-H [nT]')

    for image_time, image, _, im in gen:
        # Add your code that modifies each image here...
        # To demonstrate, lets annotate each frame with the timestamp.
        # We will need to delete the prior text object, otherwise the current one
        # will overplot on the prior one---clean up after yourself!
        if 'text_obj' in locals():
            # text_obj.remove()  # noqa: F821
            _time_guide.remove()  # noqa: F821
        text_obj = plt.suptitle(f'MANGO-{location_code} | {image_time:%F %T}', fontsize=15)
        _time_guide = bx.axvline(image_time, c='k', ls='--')