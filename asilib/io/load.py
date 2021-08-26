import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import List, Union, Sequence, Tuple
from copy import copy
import warnings
import re

import pandas as pd
import numpy as np
import cdflib
import scipy.io
import matplotlib.pyplot as plt

from asilib.io import download_rego
from asilib.io import download_themis
import asilib


def load_img(
    time: Union[datetime, str], mission: str, station: str, force_download: bool = False
) -> cdflib.cdfread.CDF:
    """
    Returns a full image (ASF) cdflib.CDF file object and download it if it's not found locally.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. Must contain the date and the UT hour.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    cdflib.CDF
        The handle to the full image CDF object. Use cdflib.CDF.varget() 
        to load the variables into memory (see the implementation in 
        asilib.io.load.get_frame() or asilib.io.load.get_frames())

    Raises
    ------
    FileNotFoundError
        Catches the NotADirectoryError raised by download.py, and raises
        this FileNotFoundError that clearly conveys that the file was not
        found in the file system or online.
    ValueError
        Raised if there is an error with the file finding logic (ideally 
        should not be raised).

    Example
    -------
    | import asilib
    |
    | asi_file_handle = asilib.load_img('2016-10-29T04', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    # Download data if force_download == True:
    # Check if the REGO or THEMIS data is already saved locally.
    search_path = pathlib.Path(asilib.config['ASI_DATA_DIR'], mission.lower())
    search_pattern = f'*asf*{station.lower()}*{time.strftime("%Y%m%d%H")}*'
    matched_paths = list(search_path.rglob(search_pattern))
    # Try to download files if one is not found locally.

    if (len(matched_paths) == 1) and (not force_download):
        # If a local file was found and the user does not want to force the download.
        file_path = matched_paths[0]

    elif (len(matched_paths) == 1) and (force_download):
        # If a local file was found and the user does want to force the download.
        # These downloaders are guaranteed to find a matching file unless the server
        # lost the file.
        if mission.lower() == 'themis':
            file_path = download_themis.download_themis_img(
                time, station, force_download=force_download
            )[0]
        elif mission.lower() == 'rego':
            file_path = download_rego.download_rego_img(
                time, station, force_download=force_download
            )[0]

    elif len(matched_paths) == 0:
        # Now if no local files were found, try to download it.
        if mission.lower() == 'themis':
            try:
                file_path = download_themis.download_themis_img(
                    time, station, force_download=force_download
                )[0]
            except NotADirectoryError:
                raise FileNotFoundError(
                    f'THEMIS ASI data not found for station {station} on day {time.date()}'
                )
        elif mission.lower() == 'rego':
            try:
                file_path = download_rego.download_rego_img(
                    time, station, force_download=force_download
                )[0]
            except NotADirectoryError:
                raise FileNotFoundError(
                    f'REGO ASI data not found for station {station} on day {time.date()}'
                )
    else:
        raise ValueError(f"Not sure what happend here. I found {matched_paths} matching paths.")

    # If we made it here, we either found a local file, or downloaded one
    return cdflib.CDF(file_path)

def load_img_file(time, mission: str, station: str, force_download: bool = False):
    """
    DEPRECATED for load_img()
    """
    warnings.warn('load_img_file is deprecated. Use asilib.load_img() instead', DeprecationWarning)
    return load_img(time, mission, station, force_download)


def load_skymap(mission: str, station: str, time: Union[datetime, str], force_download: bool = False) -> dict:
    """
    Loads (and downloads if it doesn't exist) the skymap file closest and before time.

    Parameters
    ----------
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    time: datetime, or str
        Time is used to find the relevant skymap file: file created nearest to, and before, the time.
    force_download: bool (optional)
        If True, download the file even if it already exists.

    Returns
    -------
    dict
        The skymap data with longitudes mapped from 0->360 to to -180->180 degrees.

    Example
    -------
    | import asilib
    | 
    | rego_skymap = asilib.load_skymap('REGO', 'GILL', '2018-10-01')
    """
    skymap_dir = pathlib.Path(asilib.config['ASI_DATA_DIR'], mission.lower(), 'skymap', station.lower())
    skymap_paths = sorted(list(skymap_dir.rglob(f'{mission.lower()}_skymap_{station.lower()}*')))

    # Download skymap files if they are not downloaded yet.
    if len(skymap_paths) == 0 and mission.lower() == 'themis':
        skymap_paths = download_themis.download_themis_skymap(station)
    elif len(skymap_paths) == 0 and mission.lower() == 'rego':
        skymap_paths = download_rego.download_rego_skymap(station)

    skymap_dates = _extract_skymap_dates(skymap_paths)

    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date. 
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = skymap_paths[closest_index]
    
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy(skymap_file[key][0]) for key in skymap_file.dtype.names}
    # Map longitude from 0 - 360 to -180 - 180.
    skymap_dict['SITE_MAP_LONGITUDE'] = np.mod(skymap_dict['SITE_MAP_LONGITUDE'] + 180, 360) - 180
    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap_dict['FULL_MAP_LONGITUDE']))
    skymap_dict['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(skymap_dict['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    skymap_dict['skymap_path'] = skymap_path
    return skymap_dict

def _extract_skymap_dates(skymap_paths):
    """
    Extract the skymap dates from each skymap_path in skymap_paths.
    """
    skymap_dates = []

    for skymap_path in sorted(skymap_paths):
        day = re.search(r'\d{8}', skymap_path.name).group(0)
        day_obj = datetime.strptime(day, "%Y%m%d")
        skymap_dates.append(day_obj)
    return skymap_dates

def load_cal(mission: str, station: str, time, force_download: bool = False):
    """
    DEPRECATED for load_skymap()
    """
    warnings.warn('asilib.load_cal() is deprecated, use asilib.load_skymap() instead', 
        DeprecationWarning
        )
    return load_skymap(mission, station, time, force_download)

def load_cal_file(mission: str, station: str, force_download: bool = False):
    """
    DEPRECATED for load_cal()
    """
    warnings.warn('asilib.load_cal_file() is deprecated, use asilib.load_skymap() instead', 
        DeprecationWarning
        )
    return load_cal(mission, station, force_download)

def get_frame(
    time: Union[datetime, str],
    mission: str,
    station: str,
    force_download: bool = False,
    time_thresh_s: float = 3,
) -> Tuple[datetime, np.ndarray]:
    """
    Gets one ASI image frame given the mission (THEMIS or REGO), station, and
    the day date-time parameters. If a file does not locally exist, this
    function will attempt to download it.

    Parameters
    ----------
    time: datetime.datetime or str
        The date and time to download the data from. If time is string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.

    Returns
    -------
    frame_time: datetime
        The frame timestamp.
    frame: np.ndarray
        A 2D array of the ASI image at the date-time nearest to the
        time argument.

    Raises
    ------
    AssertionError
        If a unique time stamp was not found within time_thresh_s of 
        time.

    Example
    -------
    | import asilib
    | 
    | time, frame = asilib.get_frame('2016-10-29T04:15:00', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    cdf_obj = load_img(time, mission, station, force_download=force_download)

    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        frame_key = f'thg_asf_{station.lower()}'
        time_key = f'thg_asf_{station.lower()}_epoch'

    # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
    # to datetime objects.
    epoch = _get_epoch(cdf_obj, time_key, time, mission, station)
    # Find the closest time stamp to time
    idx = np.where((epoch >= time) & (epoch < time + timedelta(seconds=time_thresh_s)))[0]
    assert len(idx) == 1, (
        f'{len(idx)} number of time stamps were found '
        f'within {time_thresh_s} seconds of {time}. '
        f'You can change the time_thresh_s kwarg to find a '
        f'time stamp further away.'
    )
    return epoch[idx[0]], cdf_obj.varget(frame_key)[idx[0], :, :]


def get_frames(
    time_range: Sequence[Union[datetime, str]],
    mission: str,
    station: str,
    force_download: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets multiple ASI image frames given the mission (THEMIS or REGO), station, and
    the time_range date-time parameters. If a file does not locally exist, this
    function will attempt to download it.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        frames. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.

    Returns
    -------
    frame_time: datetime
        The frame timestamps contained in time_range, inclduing the start
        and end times.
    frame: np.ndarray
        An (nTime x nPixelRows x nPixelCols) array containing the ASI images
        for times contained in time_range.

    Raises
    ------
    NotImplementedError
        If the image dimensions are not specified for an ASI mission.
    AssertionError
        If the data file exists with no time stamps contained in time_range.
    AssertionError
        If len(time_range) != 2.
        
    Example
    -------
    | from datetime import datetime
    | 
    | import asilib
    | 
    | time_range = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    | times, frames = asilib.get_frames(time_range, 'REGO', 'GILL')
    """
    time_range = _validate_time_range(time_range)

    # Figure out the data keys to load.
    if mission.lower() == 'rego':
        frame_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        frame_key = f'thg_asf_{station.lower()}'
        time_key = f'thg_asf_{station.lower()}_epoch'

    # Determine if we need to load in one hour or multiple hours worth of data.
    start_time_rounded = time_range[0].replace(minute=0, second=0, microsecond=0)
    end_time_rounded = time_range[1].replace(minute=0, second=0, microsecond=0)
    if start_time_rounded == end_time_rounded:
        # If the start/end date-hour are the same than load one data file, otherwise
        # load however many is necessary and concatinate the frames and times.
        cdf_obj = load_img(time_range[0], mission, station, force_download=force_download)

        # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
        # to datetime objects.
        epoch = _get_epoch(cdf_obj, time_key, time_range[0], mission, station)

        # Get the frames 3d array
        frames = cdf_obj.varget(frame_key)
    else:
        # If the time_range is across two or more hours
        epoch = np.array([])
        if mission.lower() == 'themis':
            frames = np.ones((0, 256, 256))
        elif mission.lower() == 'rego':
            frames = np.ones((0, 512, 512))
        else:
            raise NotImplementedError

        if (time_range[1].minute == 0) and (time_range[1].second == 0):
            hourly_date_times = pd.date_range(start=time_range[0], end=time_range[1], freq='H')
        else:
            # The timedelta offset is needed to include the end hour.
            hourly_date_times = pd.date_range(
                start=time_range[0], end=time_range[1]+pd.Timedelta(hours=1), freq='H'
            )
        for hour_date_time in hourly_date_times:
            cdf_obj = load_img(hour_date_time, mission, station, force_download=force_download)

            # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
            # to datetime objects.
            epoch = np.append(
                    epoch, _get_epoch(cdf_obj, time_key, hour_date_time, mission, station)
            )

            # Get the frames 3d array and concatenate.
            frames = np.concatenate((frames, cdf_obj.varget(frame_key)), axis=0)

    # Find the time stamps in between time_range.
    idx = np.where((epoch >= time_range[0]) & (epoch <= time_range[1]))[0]
    assert len(idx) > 0, (f'The data exists for {mission}/{station}, but no '
                        f'data between {time_range}')
    return epoch[idx], frames[idx, :, :]

def get_frames_generator(time_range, mission, station, force_download=False):
    """
    Gets multiple ASI image frames given the mission (THEMIS or REGO), station, and
    the time_range date-time parameters. If a file does not locally exist, this
    function will attempt to download it. This generator yields the ASI data, file 
    by file, bounded by time_range. This generator is useful for loading lots of data
    for keograms.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        frames. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool (optional)
        If True, download the file even if it already exists.
    time_thresh_s: float
        The maximum allowed time difference between a frame's time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.

    Returns
    -------
    frame_time: datetime
        The frame timestamps contained in time_range, inclduing the start
        and end times.
    frame: np.ndarray
        An (nTime x nPixelRows x nPixelCols) array containing the ASI images
        for times contained in time_range.
    """

    raise NotImplementedError

def _validate_time_range(time_range):
    """
    Checks that len(time_range) == 2 and that it can be converted to datetime objects,
    if necessary.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        frames. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.

    Returns
    -------
    time_range: np.array
        An array of length two with the sorted datetime.datetime objects.

    Raises
    ------
    AssertionError
        If len(time_range) != 2.
    """
    assert len(time_range) == 2, f'len(time_range) = {len(time_range)} is not 2.'

    # Create a list version of time_range in case it is a tuple.
    time_range_list = []

    for t_i in time_range:
        if isinstance(t_i, str):
            # Try to parse it if passed a string.
            time_range_list.append(dateutil.parser.parse(t_i))
        elif isinstance(t_i, (datetime, pd.Timestamp)):
            # If passed a the native or pandas datetime object. 
            time_range_list.append(t_i)
        else:
            raise ValueError(f'Unknown time_range format. Got {time_range}. Start/end times must be '
                            'strings that can be parsed by dateutil.parser.parse, or '
                            'datetime.datetime, or pd.Timestamp objects.')

    # Sort time_range if the user passed it in out of order.
    time_range_list = sorted(time_range_list)
    return time_range_list

def _get_epoch(cdf_obj, time_key, hour_date_time, mission, station):
    """
    Gets the CDF epoch array and modifies a ValueError when a CDF file is corrupted.
    """
    try:
        epoch = np.array(
            cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key))
            )
    except ValueError as err:
        if str(err) == 'read length must be non-negative or -1':
            raise ValueError(str(err) + '\n\n ASI data is probably corrupted for '
            f'time={hour_date_time}, mission={mission}, station={station}. '
            'download the data again with force_download=True).'
            )
        else:
            raise
    return epoch

if __name__ == '__main__':
    skymap = load_skymap('THEMIS', 'FSMI', '2015-10-16')
    print(skymap)
    print(skymap['skymap_path'])
    pass