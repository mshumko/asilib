"""
As part of the Canadian Space Agency's Geospace Observatory (GO) Canada initiative, the Auroral Imaging Group operates an array of redline auroral all-sky imagers designed by the AIG for the RISR-C CFI project. These systems are specifically designed to capture image data from only the 630.0nm wavelength (red coloured aurora). The Redline Geospace Observatory (REGO) array consists of 9 systems and are co-located with other auroral instrumentation that we operate to maximize the scientific impact. These imaging systems were deployed in late 2014 and continue to operate.
"""

from datetime import datetime, timedelta
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.colors
import rego_imager_readfile

import asilib
import asilib.asi.themis as themis
import asilib.utils as utils
import asilib.io.download as download


pgm_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/'
skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/skymap/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'rego'


def rego(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance with the REGO ASI images and skymaps.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters).
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    load_images: bool
        Create an Imager object without images. This is useful if you need to
        calculate conjunctions and don't need to download or load unnecessary data.
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        The THEMIS Imager instance.

    Examples
    --------
    >>> # Plot a single image.
    >>> from datetime import datetime
    >>> import matplotlib.pyplot as plt
    >>> import asilib
    >>> import asilib.map
    >>> location_code = 'RANK'
    >>> time = datetime(2017, 9, 15, 2, 34, 0)
    >>> alt_km = 110
    >>> fig = plt.figure(figsize=(10, 6))
    >>> ax = fig.add_subplot(121)
    >>> bx = asilib.map.create_map(fig_ax=(fig, 122), lon_bounds = (-102, -82), lat_bounds = (58, 68))
    >>> asi = asilib.rego(location_code, time=time, alt=alt_km)
    >>> asi.plot_fisheye(ax=ax)
    >>> asi.plot_map(ax=bx)
    >>> plt.tight_layout()
    >>> plt.show()

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A REGO ASI instance with the time stamps, images, skymaps, and metadata.
    """
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_pgm_dir = local_base_dir / 'images' / location_code.lower()

    if load_images:
        # Download and find image data
        file_paths = themis._get_pgm_files(
            'rego',
            location_code,
            time,
            time_range,
            pgm_base_url,
            local_pgm_dir,
            redownload,
            missing_ok,
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d_%H%M')
            end_times[i] = start_times[i] + timedelta(minutes=1)
        file_info = {
            'path': file_paths,
            'start_time': start_times,
            'end_time': end_times,
            'loader': _load_rego_pgm,
        }
    else:
        file_info = {
            'path': [],
            'start_time': [],
            'end_time': [],
            'loader': [],
        }
    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time

    # Download and find the appropriate skymap
    if time is not None:
        _time = time
    else:
        _time = time_range[0]
    _skymap = rego_skymap(location_code, _time, redownload=redownload)
    alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0]
    assert (
        len(alt_index) == 1
    ), f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km.'
    alt_index = alt_index[0]
    skymap = {
        'lat': _skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        'lon': _skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        'alt': _skymap['FULL_MAP_ALTITUDE'][alt_index] / 1e3,
        'el': _skymap['FULL_ELEVATION'],
        'az': _skymap['FULL_AZIMUTH'],
        'path': _skymap['PATH'],
    }

    meta = {
        'array': 'REGO',
        'location': location_code.upper(),
        'lat': float(_skymap['SITE_MAP_LATITUDE']),
        'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 3,
        'resolution': (512, 512),
    }
    plot_settings = {
        'color_map': matplotlib.colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    }
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def rego_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the REGO ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of REGO imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'REGO']
    return df.reset_index(drop=True)


def rego_skymap(location_code: str, time: utils._time_type, redownload: bool = False) -> dict:
    """
    Load a REGO ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-fomatted time string or datetime object. Must be in UT time.
    redownload: bool
        Redownload the file.

    Returns
    -------
    dict
        The skymap.
    """
    time = utils.validate_time(time)
    local_dir = local_base_dir / 'skymaps' / location_code.lower()
    local_dir.mkdir(parents=True, exist_ok=True)
    skymap_top_url = skymap_base_url + location_code.lower() + '/'

    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_top_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        )
        # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_top_url, local_dir, redownload=redownload
            )

    skymap_filenames = [local_skymap_path.name for local_skymap_path in local_skymap_paths]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_file_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for REGO-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymap_paths[closest_index]
    skymap = _load_skymap(skymap_path)
    return skymap


def _download_all_skymaps(location_code, url, save_dir, redownload):
    d = download.Downloader(url)
    # Find the dated subdirectories
    ds = d.ls(f'{location_code.lower()}')

    save_paths = []
    for d_i in ds:
        ds = d_i.ls(f'*skymap_{location_code.lower()}*.sav')
        for ds_j in ds:
            save_paths.append(ds_j.download(save_dir, redownload=redownload))
    return save_paths


def _load_skymap(skymap_path):
    """
    A helper function to load a REGO skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy.copy(skymap_file[key][0]) for key in skymap_file.dtype.names}

    skymap_dict = _tranform_longitude_to_180(skymap_dict)
    skymap_dict = _flip_skymap(skymap_dict)
    skymap_dict['PATH'] = skymap_path
    return skymap_dict


def _flip_skymap(skymap):
    """
    IDL is a column-major language while Python is row-major. This function
    tranposes the 2- and 3-D arrays to make them compatible with the images
    that are saved in row-major.
    """
    for key in skymap:
        if hasattr(skymap[key], 'shape'):
            shape = skymap[key].shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                skymap[key] = skymap[key][::-1, :]  # For Az/El maps.
            elif (len(shape) == 3) and (shape[1] == shape[2]):
                skymap[key] = skymap[key][:, ::-1, :]  # For lat/lon maps
    return skymap


def _tranform_longitude_to_180(skymap):
    """
    Transform the SITE_MAP_LONGITUDE and FULL_MAP_LONGITUDE arrays from
    (0 -> 360) to (-180 -> 180).
    """
    skymap['SITE_MAP_LONGITUDE'] = np.mod(skymap['SITE_MAP_LONGITUDE'] + 180, 360) - 180

    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['FULL_MAP_LONGITUDE']))
    skymap['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(skymap['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    return skymap


def _load_rego_pgm(path):
    images, meta, problematic_file_list = rego_imager_readfile.read(str(path))
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, :]  # Flip so north is up and .
    times = np.array(
        [
            dateutil.parser.parse(dict_i['Image request start']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images
