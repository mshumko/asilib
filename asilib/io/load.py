from os import stat
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


def _find_img_path(
    time: Union[datetime, str], mission: str, station: str, force_download: bool = False
) -> cdflib.cdfread.CDF:
    """
    Returns a path to an all sky full-resolution image (THEMIS:ASF, REGO:rgf) file.
    If a file is not found locally, it will attempt to download it.

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
    pathlib.Path
        The path to the full image file. See the implementation in
        asilib.io.load._load_image() or asilib.io.load._load_images() on
        how to use cdflib to load the image cdf files.

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
    | asi_file_path = asilib._find_img_path('2016-10-29T04', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    if force_download:
        if mission.lower() == 'themis':
            file_path = download_themis.download_themis_img(
                time, station, force_download=force_download
            )[0]
        elif mission.lower() == 'rego':
            file_path = download_rego.download_rego_img(
                time, station, force_download=force_download
            )[0]
    else:
        # If the user does not want to force a download, look for a file on the
        # computer. If a local file is not found, try to download one.
        search_path = pathlib.Path(asilib.config['ASI_DATA_DIR'], mission.lower())
        if mission.lower() == 'themis':
            search_pattern = f'*asf*{station.lower()}*{time.strftime("%Y%m%d%H")}*'
        elif mission.lower() == 'rego':
            search_pattern = f'*rgf*{station.lower()}*{time.strftime("%Y%m%d%H")}*'
        matched_paths = list(search_path.rglob(search_pattern))

        if len(matched_paths) == 1:  # A local file found
            file_path = matched_paths[0]

        elif len(matched_paths) == 0:  # No local file found
            if mission.lower() == 'themis':
                try:
                    file_path = download_themis.download_themis_img(
                        time, station, force_download=force_download
                    )[0]
                except NotADirectoryError:
                    raise FileNotFoundError(
                        f'THEMIS ASI data not found for station {station} at {time}'
                    )
            elif mission.lower() == 'rego':
                try:
                    file_path = download_rego.download_rego_img(
                        time, station, force_download=force_download
                    )[0]
                except NotADirectoryError:
                    raise FileNotFoundError(
                        f'REGO ASI data not found for station {station} at {time}'
                    )
        else:  # Multiple files found?
            raise ValueError(f"Not sure what happend here. I found {matched_paths} matching paths.")

    return file_path


def load_image(asi_array_code: str, location_code: str, time=None, time_range=None, 
    force_download: bool = False, time_thresh_s: float = 3, ignore_missing_data: bool = True):
    """
    Wrapper for the _load_image and _load_images functions. 
    """
    # A bunch of if statements that download image files only when either time or time_range
    # is specified (not both).
    if (time is None) and (time_range is None):
        raise AttributeError('Neither time or time_range is specified.')
    elif ((time is not None) and (time_range is not None)):
        raise AttributeError('Both time and time_range can not be simultaneously specified.')
    elif time is not None:
        return _load_image(time, asi_array_code, location_code, force_download=force_download,
                    time_thresh_s=time_thresh_s)
    elif time_range is not None:
        return _load_images(time_range, asi_array_code, location_code, force_download=force_download, 
                    ignore_missing_data=ignore_missing_data)
    return


def load_skymap(
    mission: str, station: str, time: Union[datetime, str], force_download: bool = False
) -> dict:
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
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    if force_download:
        if mission.lower() == 'themis':
            skymap_paths = download_themis.download_themis_skymap(
                station, force_download=force_download
            )
        elif mission.lower() == 'rego':
            skymap_paths = download_rego.download_rego_skymap(
                station, force_download=force_download
            )

    else:
        # If the user does not want to force download skymap files,
        # look for the appropriate file on the computer. If a local
        # skymap file is not found, download them all and look for the
        # appropriate file.
        skymap_dir = pathlib.Path(
            asilib.config['ASI_DATA_DIR'], mission.lower(), 'skymap', station.lower()
        )
        skymap_paths = sorted(
            list(skymap_dir.rglob(f'{mission.lower()}_skymap_{station.lower()}*'))
        )

        # Download skymap files if they are not downloaded yet.
        if len(skymap_paths) == 0:
            if mission.lower() == 'themis':
                skymap_paths = download_themis.download_themis_skymap(
                    station, force_download=force_download
                )
            elif mission.lower() == 'rego':
                skymap_paths = download_rego.download_rego_skymap(
                    station, force_download=force_download
                )

    skymap_dates = _extract_skymap_dates(skymap_paths)

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for {mission}/{station} is before first '
            f'skymap file: {skymap_paths[0].name}. This skymap file will be used.'
        )
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

def get_frame(
        time: Union[datetime, str],
        asi_array_code: str,
        location_code: str,
        force_download: bool = False,
        time_thresh_s: float = 3,
        ) -> Tuple[datetime, np.ndarray]:

    warnings.warn('asilib.get_frame is deprecated for asilib.load_image')

    return _load_image(time, asi_array_code, location_code,
                force_download=force_download, 
                time_thresh_s=time_thresh_s)

def _load_image(
    time: Union[datetime, str],
    mission: str,
    station: str,
    force_download: bool = False,
    time_thresh_s: float = 3,
) -> Tuple[datetime, np.ndarray]:
    """
    Gets one ASI image image given the mission (THEMIS or REGO), station, and
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
        The maximum allowed time difference between a image time stamp
        and the time argument in seconds. Will raise a ValueError if no
        image time stamp is within the threshold.

    Returns
    -------
    image_time: datetime
        The image timestamp.
    image: np.ndarray
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
    | time, image = asilib.io.load._load_image('2016-10-29T04:15:00', 'REGO', 'GILL')
    """
    # Try to convert time to datetime object if it is a string.
    if isinstance(time, str):
        time = dateutil.parser.parse(time)

    cdf_path = _find_img_path(time, mission, station, force_download=force_download)
    cdf_obj = cdflib.CDF(cdf_path)

    if mission.lower() == 'rego':
        image_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        image_key = f'thg_asf_{station.lower()}'
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
    return epoch[idx[0]], cdf_obj.varget(image_key)[idx[0], :, :]

def get_frames(
        time_range: Sequence[Union[datetime, str]],
        asi_array_code: str,
        location_code: str,
        force_download: bool = False,
        ) -> Tuple[datetime, np.ndarray]:

    warnings.warn('asilib.get_frames is deprecated for asilib.load_image.')

    return _load_images(time_range, asi_array_code, location_code,
                force_download=force_download)

def _load_images(
    time_range: Sequence[Union[datetime, str]],
    mission: str,
    station: str,
    force_download: bool = False,
    ignore_missing_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets multiple ASI image images given the mission (THEMIS or REGO), station, and
    the time_range date-time parameters. If a file does not locally exist, this
    function will attempt to download it. The returned time stamps span a range
    from time_range[0], up to, but excluding a time stamp exactly matching time_range[1].

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        images. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool
        If True, download the file even if it already exists.
    ignore_missing_data: bool
        Flag to ignore the FileNotFoundError that is raised when ASI
        data is unavailable for that date-hour.

    Returns
    -------
    times: datetime
        The image timestamps contained in time_range, including the start time
        and excluding the end time (if time_range[1] exactly matches a ASI time
        stamp).
    image: np.ndarray
        An (nTime x nPixelRows x nPixelCols) array containing the ASI images
        for times contained in time_range.

    Raises
    ------
    NotImplementedError
        If the image dimensions are not specified for an ASI mission.
    AssertionError
        If len(time_range) != 2.

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | time_range = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    | times, images = asilib.io.load._load_images(time_range, 'REGO', 'GILL')
    """
    times, images = _create_empty_data_arrays(mission, time_range, 'images')
    image_generator = load_image_generator(time_range, mission, station, 
        force_download=force_download, ignore_missing_data=ignore_missing_data)

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]

        images[start_time_index:end_time_index, :, :] = file_images
        times[start_time_index:end_time_index] = file_image_times

        start_time_index += file_images.shape[0]

    i_nan = np.where(~np.isnan(images[:, 0, 0]))[0]
    images = images[i_nan, :, :]
    times = times[i_nan]
    return times, images


def load_image_generator(
    time_range: Sequence[Union[datetime, str]],
    mission: str,
    station: str,
    force_download: bool = False,
    ignore_missing_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yields multiple ASI image files given the mission (THEMIS or REGO), station, and
    time_range parameters. If a file does not locally exist, this function will attempt
    to download it. This generator yields the ASI data, file by file, bounded by time_range.
    This generator is useful for loading lots of data---useful for keograms. The returned
    time stamps span a range from time_range[0], up to, but excluding a time stamp
    exactly matching time_range[1].

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        images. If either start or end time is a string,
        dateutil.parser.parse will attempt to parse it into a datetime
        object. The user must specify the UT hour and the first argument
        is assumed to be the start_time and is not checked.
    mission: str
        The mission id, can be either THEMIS or REGO.
    station: str
        The station id to download the data from.
    force_download: bool
        If True, download the file even if it already exists.
    ignore_missing_data: bool
        Flag to ignore the FileNotFoundError that is raised when ASI
        data is unavailable for that date-hour.

    Yields
    -------
    times: datetime
        The image timestamps contained in time_range, including the start time
        and excluding the end time (if time_range[1] exactly matches a ASI time
        stamp).
    images: np.ndarray
        An (nTime x nPixelRows x nPixelCols) array containing the ASI images
        for times contained in time_range.
    """

    time_range = _validate_time_range(time_range)

    # Figure out the data keys to load.
    if mission.lower() == 'rego':
        image_key = f'clg_rgf_{station.lower()}'
        time_key = f'clg_rgf_{station.lower()}_epoch'
    elif mission.lower() == 'themis':
        image_key = f'thg_asf_{station.lower()}'
        time_key = f'thg_asf_{station.lower()}_epoch'

    hours = _get_hours(time_range)

    for hour in hours:
        try:
            cdf_path = _find_img_path(hour, mission, station, force_download=force_download)
            cdf_obj = cdflib.CDF(cdf_path)
        except FileNotFoundError:
            if ignore_missing_data:
                pass
            else:
                raise

        epoch = _get_epoch(cdf_obj, time_key, hour, mission, station)

        idx = np.where((epoch >= time_range[0]) & (epoch < time_range[1]))[0]
        yield epoch[idx], cdf_obj.varget(image_key, startrec=idx[0], endrec=idx[-1])


def _validate_time_range(time_range):
    """
    Checks that len(time_range) == 2 and that it can be converted to datetime objects,
    if necessary.

    Parameters
    ----------
    time_range: List[Union[datetime, str]]
        A list with len(2) == 2 of the start and end time to get the
        images. If either start or end time is a string,
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
            raise ValueError(
                f'Unknown time_range format. Got {time_range}. Start/end times must be '
                'strings that can be parsed by dateutil.parser.parse, or '
                'datetime.datetime, or pd.Timestamp objects.'
            )

    # Sort time_range if the user passed it in out of order.
    time_range_list = sorted(time_range_list)
    return time_range_list


def _get_epoch(cdf_obj, time_key, hour_date_time, mission, station):
    """
    Gets the CDF epoch array and modifies a ValueError when a CDF file is corrupted.
    """
    try:
        epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
    except ValueError as err:
        if str(err) == 'read length must be non-negative or -1':
            raise ValueError(
                str(err) + '\n\n ASI data is probably corrupted for '
                f'time={hour_date_time}, mission={mission}, station={station}. '
                'download the data again with force_download=True).'
            )
        else:
            raise
    return epoch


def _get_hours(time_range):
    """
    Helper function to figure out what date-hour times are between the times in time_range.
    This function is useful to figure out what hourly ASI files to download.
    """
    time_range = _validate_time_range(time_range)

    # Modify time_range. If time_range[0] is not at the top of the hour, we zero the minutes
    # seconds, and milliseconds. This helps with keeping the + 1 hour offsets aligned to the
    # start of the hour.
    time_range[0] = time_range[0].replace(minute=0, second=0, microsecond=0)

    current_hour = copy(time_range[0])
    hours = []

    # Not <= because we down want to download the final hour if time_range[1] is, for example,
    # 05:00:00 [HH:MM:SS]
    while current_hour < time_range[1]:
        hours.append(current_hour)
        current_hour += timedelta(hours=1)
    return hours


def _create_empty_data_arrays(mission, time_range, type):
    """
    Creates two appropriately sized np.arrays full of np.nan. The first is a 1d times array,
    and the second is either: a 2d array (n_steps, n_pixels) if type=='keogram', or a 3d array
    (n_times, n_pixels, n_pixels) if type='images'.
    """
    if mission.lower() == 'themis':
        img_size = 256
        cadence_s = 3
        max_n_timestamps = int((time_range[1] - time_range[0]).total_seconds() / cadence_s)
    elif mission.lower() == 'rego':
        img_size = 512
        cadence_s = 3
        max_n_timestamps = int((time_range[1] - time_range[0]).total_seconds() / cadence_s)
    else:
        raise NotImplementedError

    if type.lower() == 'keogram':
        data_shape = (max_n_timestamps, img_size)
    elif type.lower() == 'images':
        data_shape = (max_n_timestamps, img_size, img_size)
    else:
        raise ValueError('type must be "keogram" or "images".')

    # object is the only dtype that can contain datetime objects
    times = np.nan * np.zeros(max_n_timestamps, dtype=object)
    data = np.nan * np.zeros(data_shape)
    return times, data


if __name__ == '__main__':
    d = asilib.load_skymap('THEMIS', 'GILL', '2000-01-01', force_download=False)