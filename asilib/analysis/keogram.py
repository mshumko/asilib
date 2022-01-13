import numpy as np
import pandas as pd

from asilib.io import utils
from asilib.io.load import (
    load_image_generator,
    load_skymap,
    _create_empty_data_arrays,
)


def keogram(
    asi_array_code: str, location_code: str, time_range: utils._time_range_type, 
    map_alt: int = None, mode: str = 'keo', path: np.array = None
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
    mode: str
        The keogram mode. If ``keo`` then this makes a keogram and if ``ewo`` to make
        an ewogram.
    path: array
        Make a keogram along a custom path. Path shape must be (n, 2) and contain the 
        lat/lon coordinates that are mapped to map_alt. If map_alt is unspecified, will
        raise a ValueError.

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
        If no imager data is found in ``time_range``.
    ValueError
        If the mode kwarg is not ``keo`` or ``ewo``.
    ValueError
        If a custom path is supplied but not map_alt.
    """
    image_generator = load_image_generator(asi_array_code, location_code, time_range)
    keo_times, keo = _create_empty_data_arrays(asi_array_code, time_range, 'keogram')

    start_time_index = 0
    for file_image_times, file_images in image_generator:
        end_time_index = start_time_index + file_images.shape[0]
        if path is None:
            if mode == 'keo':  # Slice the meridian
                keo[start_time_index:end_time_index, :] = file_images[
                    :, :, keo.shape[2] // 2
                ]  
            elif mode == 'ewo':  # East-West slice
                keo[start_time_index:end_time_index, :] = file_images[
                    :, keo.shape[1] // 2, :
                ]  
            else:
                raise ValueError(f'The asilib.keogram() mode kwarg must be "keo" or "ewo", or '
                                 f'you must supply a path; Got mode={mode}.')
        else:
            if map_alt is None:
                raise ValueError(f'If you need a keogram along a path, you need to supply the map altitude.')
        

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
    else:
        skymap = load_skymap(asi_array_code, location_code, time_range[0])
        assert (
            map_alt in skymap['FULL_MAP_ALTITUDE'] / 1000
        ), f'{map_alt} km is not in skymap altitudes: {skymap["FULL_MAP_ALTITUDE"]/1000} km'
        alt_index = np.where(skymap['FULL_MAP_ALTITUDE'] / 1000 == map_alt)[0][0]
        keogram_latitude = skymap['FULL_MAP_LATITUDE'][alt_index, :, keo.shape[1] // 2]

        # keogram_latitude array are at the pixel edges. Remap it to the centers
        dl = keogram_latitude[1:] - keogram_latitude[:-1]
        keogram_latitude = keogram_latitude[0:-1] + dl / 2

        # Since keogram_latitude values are NaNs near the image edges, we want to filter
        # out those indices from keogram_latitude and keo.
        valid_lats = np.where(~np.isnan(keogram_latitude))[0]
        keogram_latitude = keogram_latitude[valid_lats]
        keo = keo[:, valid_lats]
    return pd.DataFrame(data=keo, index=keo_times, columns=keogram_latitude)
