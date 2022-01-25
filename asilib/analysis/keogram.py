import numpy as np
import pandas as pd
import warnings

from asilib.io import utils
from asilib.io.load import (
    load_image_generator,
    load_skymap,
    _create_empty_data_arrays,
)


def keogram(
    asi_array_code: str, location_code: str, time_range: utils._time_range_type, 
    map_alt: int = None, path: np.array = None, aacgm=False
):
    """
    Makes a keogram pd.DataFrame along the central meridian.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    map_alt: int
        The mapping altitude, in kilometers, used to index the mapped latitude in the
        skymap data. If None, will plot pixel index for the y-axis.
    path: array
        Make a keogram along a custom path. Path shape must be (n, 2) and contain the 
        lat/lon coordinates that are mapped to map_alt. If the map_alt kwarg is 
        unspecified, this function will raise a ValueError.
    aacgm: bool (NOT IMPLEMENTED)
        TODO: Add a flag to convert the vertical axis to AACGM coordinates.
        https://github.com/aburrell/aacgmv2

    Returns
    -------
    keo: pd.DataFrame
        The 2d keogram with the time index. The columns are the geographic latitude
        if map_alt != None, otherwise it is the image pixel values (0-265) or (0-512).

    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the skymap mapped values.
    ValueError
        If no images are in ``time_range``.
    ValueError
        If a custom path is provided but not map_alt.
    """
    if aacgm:
        raise NotImplementedError
    
    if (map_alt is None) and (path is not None):
        raise ValueError(f'If you need a keogram along a path, you need to provide the map altitude.')
    image_generator = load_image_generator(asi_array_code, location_code, time_range)
    keo_times, keo = _create_empty_data_arrays(asi_array_code, time_range, 'keogram')
    skymap = load_skymap(asi_array_code, location_code, time_range[0])

    # Check for a valid map_alt.
    if map_alt is not None:
        assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]

    # Determine what pixels to use.
    if path is not None:
        nearest_pixels, valid_pixels = _path_to_pixels(path, map_alt, skymap)
        nearest_pixels = nearest_pixels[valid_pixels, :]
        keogram_latitude = skymap['FULL_MAP_LATITUDE'][
            alt_index, 
            nearest_pixels[:, 0], 
            nearest_pixels[:, 1]
            ]
        keo = keo[:, valid_pixels]

    # Load and slice the image data.
    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]
        if path is None:
            keo[start_time_index:end_time_index, :] = file_images[
                :, :, keo.shape[1] // 2
            ]  
        else:
            keo[start_time_index:end_time_index, :] = file_images[
                :, nearest_pixels[:, 0], nearest_pixels[: ,1]
            ]  
        keo_times[start_time_index:end_time_index] = file_image_times
        start_time_index += file_images.shape[0]

    # This code block removes any filler nan values if the ASI images were not sampled at the instrument
    # cadence throughout time_range.
    i_valid = np.where(~np.isnan(keo[:, 0]))[0]
    keo = keo[i_valid, :]
    keo_times = keo_times[i_valid]

    if not keo.shape[0]:
        raise ValueError(
            f'The keogram is empty for {asi_array_code}/{location_code} '
            f'during {time_range}. The image data probably does not exist '
            f'in this time interval'
        )

    if map_alt is None:
        keogram_latitude = np.arange(keo.shape[1])  # Dummy index values for latitudes.
    elif (map_alt is not None) and (path is None):
        keogram_latitude = skymap['FULL_MAP_LATITUDE'][alt_index, :-1, keo.shape[1] // 2]

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lats = np.where(~np.isnan(keogram_latitude))[0]
        keogram_latitude = keogram_latitude[valid_lats]
        keo = keo[:, valid_lats]
    return pd.DataFrame(data=keo, index=keo_times, columns=keogram_latitude)


def ewogram(
    asi_array_code: str, location_code: str, time_range: utils._time_range_type, map_alt: int = None
):
    """
    Makes a East-West ewogram.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    map_alt: int
        The mapping altitude, in kilometers, used to index the mapped longitude in the
        skymap data. If None, will plot pixel index for the y-axis.

    Returns
    -------
    ewo: pd.DataFrame
        The 2d ewogram with the time index. The columns are the geographic longitude
        if map_alt != None, otherwise it is the image pixel indices.

    Raises
    ------
    AssertionError
        If map_alt does not equal the mapped altitudes in the skymap mapped values.
    ValueError
        If no imager data is found in ``time_range``.
    """
    # TODO: Add tests for an ewogram.
    image_generator = load_image_generator(asi_array_code, location_code, time_range)
    ewo_times, ewo = _create_empty_data_arrays(asi_array_code, time_range, 'keogram')

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]
        ewo[start_time_index:end_time_index, :] = file_images[
                :, ewo.shape[1] // 2, :
            ]  

        ewo_times[start_time_index:end_time_index] = file_image_times
        start_time_index += file_images.shape[0]

    # This code block removes any filler nan values if the ASI images were not sampled at the instrument
    # cadence throughout time_range.
    i_valid = np.where(~np.isnan(ewo[:, 0]))[0]
    ewo = ewo[i_valid, :]
    ewo_times = ewo_times[i_valid]

    if not ewo.shape[0]:
        raise ValueError(
            f'The keogram is empty for {asi_array_code}/{location_code} '
            f'during {time_range}. The image data probably does not exist '
            f'in this time interval'
        )

    if map_alt is None:
        ewo_longitude = np.arange(ewo.shape[0])  # Dummy index values for longitudes.
    else:
        skymap = load_skymap(asi_array_code, location_code, time_range[0])
        assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
        ewo_longitude = skymap['FULL_MAP_LONGITUDE'][alt_index, ewo.shape[0] // 2, :]

        # ewo_longitude array are at the pixel edges. Remap it to the centers
        dl = ewo_longitude[1:] - ewo_longitude[:-1]
        ewo_longitude = ewo_longitude[0:-1] + dl / 2

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lons = np.where(~np.isnan(ewo_longitude))[0]
        ewo_longitude = ewo_longitude[valid_lons]
        keo = ewo[:, valid_lons]
    return pd.DataFrame(data=keo, index=ewo_times, columns=ewo_longitude)

def _path_to_pixels(path, map_alt, skymap, threshold=1):
    """
    Convert the lat/lon path that is mapped to map_alt in kilometers to 
    the x- and y-pixels in the skymap lat/lon mapped file.

    Parameters
    ----------
    path: np.array
        The lat/lon array of shape (n, 2).
    map_alt: int
        The map altitude of the path, also used to reference the appropriate 
        skymap map array.
    skymap: dict
        The ASI skymap file.
    threshold: float
        The maximum distance threshold, in degrees, between the path (lat, lon) 
        and the skymap (lat, lon)

    Returns
    -------
    np.array
        The valid x pixels corresponding to the path.
    np.array
        The valid y pixels corresponding to the path.
    np.array
        Path indices corresponding to rows with a pixel within threshold 
        degrees distance.
    """
    if np.any(np.isnan(path)):
        raise ValueError("The lat/lon path can't contain NaNs.")
    if np.any(np.max(path) > 180) or np.any(np.min(path) < -180): 
        raise ValueError("The lat/lon values must be in the range -180 to 180.")
    assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
    
    alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]

    nearest_pixels = np.nan*np.zeros_like(path) 

    for i, (lat, lon) in enumerate(path):
        distances = np.sqrt(
            (skymap['FULL_MAP_LATITUDE'][alt_index, :, :] - lat)**2 +
            (skymap['FULL_MAP_LONGITUDE'][alt_index, :, :] - lon)**2
            )
        idx = np.where(distances == np.nanmin(distances))

        # Keep NaNs if distanace is larger than threshold.
        if distances[idx][0] > threshold:
            warnings.warn(
                f'Some of the keogram path coordinates are outside of the '
                f'maximum {threshold} degrees distance from the nearest '
                f'skymap map coordinate.'
                )
            continue
        nearest_pixels[i, :] = [idx[0][0], idx[1][0]]

    valid_pixels = np.where(np.isfinite(nearest_pixels[:, 0]))[0]
    if valid_pixels.shape[0] == 0:
        raise ValueError('The keogram path is completely outside of the skymap.')
    return nearest_pixels.astype(int), valid_pixels