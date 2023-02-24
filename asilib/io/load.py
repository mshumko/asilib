# TODO: Remove when cleaning up Imager
import pathlib
from datetime import datetime, timedelta
import dateutil.parser
from typing import Tuple
from copy import copy
import warnings
import re

import numpy as np
import cdflib
import scipy.io

from asilib.io.download import download_image
from asilib.io.download import download_skymap
from asilib.io import utils
import asilib


def load_image(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    redownload: bool = False,
    time_thresh_s: float = 3,
    ignore_missing_data: bool = True,
):
    """
    Load into memory an ASI image and time stamp when ``time`` is given, or images with time stamps
    when ``time_range`` is given.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    time_thresh_s: float
        The maximum allowable time difference between ``time`` and an ASI time stamp.
        This is relevant only when ``time`` is specified.
    ignore_missing_data: bool
        Flag to ignore the ``FileNotFoundError`` that is raised when ASI
        data is unavailable for that date-hour. Only useful when ``time_range``
        is passed.

    Returns
    -------
    times: datetime, or List[datetime]
        The image timestamp if ``time`` is passed, or a list of timestamps if
        ``time_range`` is passed. When ``time_range`` is passed, the timestamps
        can include start time if a timestamp exactly matches, but will exclude
        the timestamp that exactly matches the end time.
    images: np.ndarray
        Either an (nPixelRows x nPixelCols) or (nTime x nPixelRows x nPixelCols)
        array containing the ASI images.

    Example
    -------
    | # Load a single image
    | asi_array_code = 'THEMIS'
    | location_code = 'ATHA'
    | time = datetime(2008, 3, 9, 9, 18, 0)
    | image_time, image = asilib.load_image(asi_array_code, location_code, time=time, redownload=False)
    |
    | # Load multiple images
    | asi_array_code = 'REGO'
    | location_code = 'LUCK'
    | time_range = [datetime(2017, 9, 27, 7, 15), datetime(2017, 9, 27, 8, 15)]
    | image_times, images = asilib.load_image(asi_array_code, location_code, time_range=time_range, redownload=False)
    """
    if (time is None) and (time_range is None):
        raise AttributeError('Neither time or time_range is specified.')
    elif (time is not None) and (time_range is not None):
        raise AttributeError('Both time and time_range can not be simultaneously specified.')

    elif time is not None:
        return _load_image(
            asi_array_code,
            location_code,
            time,
            redownload=redownload,
            time_thresh_s=time_thresh_s,
        )

    elif time_range is not None:
        return _load_images(
            asi_array_code,
            location_code,
            time_range,
            redownload=redownload,
            ignore_missing_data=ignore_missing_data,
        )
    else:
        raise ValueError("Not supposed to get here.")


def load_image_generator(
    asi_array_code: str,
    location_code: str,
    time_range: utils._time_range_type,
    redownload: bool = False,
    ignore_missing_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yields multiple ASI image files one by one and crops the time stamps by ``time_range``.

    This generator is useful for loading lots of data---useful for keograms. The returned
    time stamps span a range from time_range[0], up to, but excluding a time stamp
    exactly matching time_range[1].

    See asilib._load_images() for an example on how to use this function.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    ignore_missing_data: bool
        Flag to ignore the ``FileNotFoundError`` that is raised when ASI
        data is unavailable for that date-hour. Only useful when ``time_range``
        is passed.

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

    time_range = utils._validate_time_range(time_range)

    # Figure out the data keys to load.
    if asi_array_code.lower() == 'rego':
        image_key = f'clg_rgf_{location_code.lower()}'
        time_key = f'clg_rgf_{location_code.lower()}_epoch'
    elif asi_array_code.lower() == 'themis':
        image_key = f'thg_asf_{location_code.lower()}'
        time_key = f'thg_asf_{location_code.lower()}_epoch'

    hours = utils._get_hours(time_range)

    for hour in hours:
        try:
            cdf_path = _find_img_path(asi_array_code, location_code, hour, redownload=redownload)
            cdf_obj = cdflib.CDF(cdf_path)
        except FileNotFoundError:
            if ignore_missing_data:
                continue
            else:
                raise

        epoch = _get_epoch(cdf_obj, time_key, hour, asi_array_code, location_code)

        idx = np.where((epoch >= time_range[0]) & (epoch < time_range[1]))[0]
        if len(idx) == 0:
            # This happens occasionally if, for example, time_range[0].minute = 57 while
            # epoch[-1].minute = 55. Continue to not try to yield an empty file.
            warnings.warn(
                f'{asi_array_code}-{location_code}, {hour} UT has no images in the time range.'
            )
            continue
        yield epoch[idx], cdf_obj.varget(image_key, startrec=idx[0], endrec=idx[-1])


def load_skymap(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type,
    redownload: bool = False,
) -> dict:
    """
    Loads the appropriate THEMIS or REGO skymap file (closest and before ``time``) into memory.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.

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

    if redownload:
        skymap_paths = download_skymap(asi_array_code.lower(), location_code, redownload=redownload)

    else:
        # If the user does not want to force download skymap files,
        # look for the appropriate file on the computer. If a local
        # skymap file is not found, download them all and look for the
        # appropriate file.
        skymap_dir = pathlib.Path(
            asilib.config['ASI_DATA_DIR'], asi_array_code.lower(), 'skymap', location_code.lower()
        )
        skymap_paths = sorted(
            list(skymap_dir.rglob(f'{asi_array_code.lower()}_skymap_{location_code.lower()}*'))
        )

        # Download skymap files if they are not downloaded yet.
        if len(skymap_paths) == 0:
            skymap_paths = download_skymap(
                asi_array_code.lower(), location_code, redownload=redownload
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
            f'The requested skymap time={time} for {asi_array_code}/{location_code} is before first '
            f'skymap file: {skymap_paths[0].name}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = skymap_paths[closest_index]

    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy(skymap_file[key][0]) for key in skymap_file.dtype.names}

    skymap_dict = _tranform_longitude_to_180(skymap_dict)
    skymap_dict = _flip_skymap(skymap_dict)
    skymap_dict['SKYMAP_PATH'] = skymap_path
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


def _load_image(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type,
    redownload: bool = False,
    time_thresh_s: float = 3,
) -> Tuple[datetime, np.ndarray]:
    """
    Gets one ASI image image given the ASI array code (THEMIS or REGO), imager location (location_code), and
    the day date-time parameters. If a file does not locally exist, this
    function will attempt to download it.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    time_thresh_s: float
        The maximum allowable time difference between ``time`` and an ASI time stamp.
        This is relevant only when ``time`` is specified.

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
    time = utils._validate_time(time)

    cdf_path = _find_img_path(asi_array_code, location_code, time, redownload=redownload)
    cdf_obj = cdflib.CDF(cdf_path)

    if asi_array_code.lower() == 'rego':
        image_key = f'clg_rgf_{location_code.lower()}'
        time_key = f'clg_rgf_{location_code.lower()}_epoch'
    elif asi_array_code.lower() == 'themis':
        image_key = f'thg_asf_{location_code.lower()}'
        time_key = f'thg_asf_{location_code.lower()}_epoch'

    # Convert the CDF_EPOCH (milliseconds from 01-Jan-0000 00:00:00)
    # to datetime objects.
    epoch = _get_epoch(cdf_obj, time_key, time, asi_array_code, location_code)
    # Find the closest time stamp to time
    idx = np.where((epoch >= time) & (epoch < time + timedelta(seconds=time_thresh_s)))[0]
    assert len(idx) == 1, (
        f'{len(idx)} number of time stamps were found '
        f'within {time_thresh_s} seconds of {time}. '
        f'You can change the time_thresh_s kwarg to find a '
        f'time stamp further away.'
    )
    return epoch[idx[0]], cdf_obj.varget(image_key)[idx[0], :, :]


def _load_images(
    asi_array_code: str,
    location_code: str,
    time_range: utils._time_range_type,
    redownload: bool = False,
    ignore_missing_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets multiple ASI image images given the asi_array_code (THEMIS or REGO), imager location code, and
    the time_range date-time parameters. If a file does not locally exist, this
    function will attempt to download it. The returned time stamps span a range
    from time_range[0], up to, but excluding a time stamp exactly matching time_range[1].

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    time_thresh_s: float
        The maximum allowable time difference between ``time`` and an ASI time stamp.
        This is relevant only when ``time`` is specified.
    ignore_missing_data: bool
        Flag to ignore the ``FileNotFoundError`` that is raised when ASI
        data is unavailable for that date-hour. Only useful when ``time_range``
        is passed. If no data is loaded at all, an AssertionError is raised.

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
        If the image dimensions are not specified for an ASI array.
    AssertionError
        If len(time_range) != 2.
    AssertionError
        If no time stamps were found in time_range, regardless of the
        ignore_missing_data value.

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | time_range = [datetime(2016, 10, 29, 4, 15), datetime(2016, 10, 29, 4, 20)]
    | times, images = asilib.io.load._load_images('REGO', 'GILL', time_range)
    """
    times, images = _create_empty_data_arrays(asi_array_code, time_range, 'images')
    image_generator = load_image_generator(
        asi_array_code,
        location_code,
        time_range,
        redownload=redownload,
        ignore_missing_data=ignore_missing_data,
    )

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]

        images[start_time_index:end_time_index, :, :] = file_images
        times[start_time_index:end_time_index] = file_image_times

        start_time_index += file_images.shape[0]

    i_not_nan = np.where(~np.isnan(images[:, 0, 0]))[0]
    assert len(i_not_nan) > 0, (
        f'{len(i_not_nan)} number of time stamps' ' were found in time_range={time_range}'
    )
    images = images[i_not_nan, :, :]
    times = times[i_not_nan]
    return times, images


def _find_img_path(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type,
    redownload: bool = False,
) -> pathlib.Path:
    """
    Returns a path to an all sky full-resolution image (THEMIS:ASF, REGO:rgf) file.
    If a file is not found locally, it will attempt to download it.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    redownload: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.

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
    | asi_file_path = asilib._find_img_path('REGO', 'GILL', '2016-10-29T04')
    """
    time = utils._validate_time(time)

    if redownload:
        file_path = download_image(
            asi_array_code.lower(), location_code, time=time, redownload=redownload
        )[0]
    else:
        # If the user does not want to force a download, look for a file on the
        # computer. If a local file is not found, try to download one.
        search_path = pathlib.Path(asilib.config['ASI_DATA_DIR'], asi_array_code.lower())
        if asi_array_code.lower() == 'themis':
            search_pattern = f'*asf*{location_code.lower()}*{time.strftime("%Y%m%d%H")}*'
        elif asi_array_code.lower() == 'rego':
            search_pattern = f'*rgf*{location_code.lower()}*{time.strftime("%Y%m%d%H")}*'
        matched_paths = list(search_path.rglob(search_pattern))

        if len(matched_paths) == 1:  # A local file found
            file_path = matched_paths[0]

        elif len(matched_paths) == 0:  # No local file found
            try:
                file_path = download_image(
                    asi_array_code.lower(), location_code, time=time, redownload=redownload
                )[0]
            except NotADirectoryError:
                raise FileNotFoundError(
                    f'{asi_array_code.upper()} ASI data not found for location_code={location_code} at {time}'
                )
        else:  # Multiple files found?
            raise ValueError(f"Not sure what happend here. I found {matched_paths} matching paths.")

    return file_path


def _get_epoch(cdf_obj, time_key, hour_date_time, asi_array_code, location_code):
    """
    Gets the CDF epoch array and modifies a ValueError when a CDF file is corrupted.
    """
    try:
        epoch = np.array(cdflib.cdfepoch.to_datetime(cdf_obj.varget(time_key)))
    except ValueError as err:
        if ('read length must be non-negative or -1' in str(err)) or ('not found' in str(err)):
            raise ValueError(
                str(err) + '\n\n ASI data is probably corrupted for '
                f'time={hour_date_time}, asi_array_code={asi_array_code}, '
                f'location_code={location_code}. This can happen when the '
                f'download was interrupted. Try to redownload the data using '
                f'redownload=True).'
            )
        else:
            raise
    return epoch


def _create_empty_data_arrays(asi_array_code, time_range, type):
    """
    Creates two appropriately sized np.arrays full of np.nan. The first is a 1d times array,
    and the second is either: a 2d array (n_steps, n_pixels) if type=='keogram', or a 3d array
    (n_times, n_pixels, n_pixels) if type='images'.
    """
    if asi_array_code.lower() == 'themis':
        img_size = 256
        cadence_s = 3
    elif asi_array_code.lower() == 'rego':
        img_size = 512
        cadence_s = 3
    else:
        raise NotImplementedError

    time_range = utils._validate_time_range(time_range)
    # + 1 just in case. The THEMIS and REGO cadence is not
    # always exactly 3 seconds, so in certain circumstances
    # there may be one or two extra data points.
    max_n_timestamps = int((time_range[1] - time_range[0]).total_seconds() / cadence_s) + 1

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


def _flip_skymap(skymap):
    """
    IDL is a column-major language while Python is row-major. This function
    tranposes the 2- and 3-D arrays to make them compatable with the images
    that are saved in row-major.
    """
    for key in skymap:
        if hasattr(skymap[key], 'shape'):
            shape = skymap[key].shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                skymap[key] = skymap[key][::-1, ::-1]  # For Az/El maps.
            elif (len(shape) == 3) and (shape[1] == shape[2]):
                skymap[key] = skymap[key][:, ::-1, ::-1]  # For lat/lon maps
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
