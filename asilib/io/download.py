from typing import List
import pathlib

import asilib
import asilib.io.utils as utils


themis_dir = pathlib.Path(asilib.config['ASI_DATA_DIR'], 'themis')
if not themis_dir.exists():
    themis_dir.mkdir(parents=True)

rego_dir = pathlib.Path(asilib.config['ASI_DATA_DIR'], 'rego')
if not rego_dir.exists():
    rego_dir.mkdir(parents=True)


def download_image(
    asi_array_code: str,
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    force_download: bool = False,
    ignore_missing_data: bool = True,
) -> List[pathlib.Path]:
    """
    Download a CDF image file when ``time`` is given, or multiple files
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
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    ignore_missing_data: bool
        Flag to ignore the ``FileNotFoundError`` that is raised when ASI
        data is unavailable for that date-hour. Only useful when ``time_range``
        is passed.

    Returns
    -------
    download_paths: list
        A list of pathlib.Path objects that contain the downloaded file
        path(s).

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | asi_array_code = 'THEMIS'
    | location_code = 'LUCK'
    | time = datetime(2017, 4, 13, 5)
    | download_path = asilib.download_image(asi_array_code, location_code, time)
    """

    if asi_array_code.lower() == 'themis':
        paths = download_themis_img(
            location_code,
            time=time,
            time_range=time_range,
            force_download=force_download,
            ignore_missing_data=ignore_missing_data,
        )
    elif asi_array_code.lower() == 'rego':
        paths = download_rego_img(
            location_code,
            time=time,
            time_range=time_range,
            force_download=force_download,
            ignore_missing_data=ignore_missing_data,
        )
    return paths


def download_skymap(
    asi_array_code: str, location_code: str, force_download: bool = False
) -> List[pathlib.Path]:
    """
    Download all of the THEMIS or REGO skymap IDL .sav files.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.

    Returns
    -------
    download_paths: list
        A list of pathlib.Path objects that contain the skymap file
        path(s).

    Example
    -------
    | import asilib
    |
    | asi_array_code = 'THEMIS'
    | location_code = 'LUCK'
    | asilib.download_skymap(asi_array_code, location_code)
    """
    if asi_array_code.lower() == 'themis':
        paths = download_themis_skymap(location_code, force_download=force_download)
    elif asi_array_code.lower() == 'rego':
        paths = download_rego_skymap(location_code, force_download=force_download)
    return paths


def download_themis_img(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    force_download: bool = False,
    ignore_missing_data: bool = True,
) -> List[pathlib.Path]:
    """
    Download one hourly THEMIS cdf file given the imager location and ``time``, or
    multiple hourly files given ``time_range``.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    ignore_missing_data: bool
        Flag to ignore the FileNotFoundError that is raised when ASI
        data is unavailable for that date-hour. Only used when
        ``time_range`` is specified.

    Returns
    -------
    download_paths: list
        A list of pathlib.Path objects that contain the downloaded file
        path(s).

    Example
    -------
    from datetime import datetime

    import asilib

    | location_code = 'LUCK'
    | time = datetime(2017, 4, 13, 5)
    | asilib.download_themis_img(location_code, time)
    """
    base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi'
    if (time is None) and (time_range is None):
        raise AttributeError('Neither time or time_range is specified.')
    elif (time is not None) and (time_range is not None):
        raise AttributeError('Both time and time_range can not be simultaneously specified.')

    elif time is not None:
        time = utils._validate_time(time)
        download_path = _download_one_img_file(
            'themis', location_code, base_url, time, force_download
        )
        download_paths = [
            download_path
        ]  # List for constancy with the time_range code chunk output.

    elif time_range is not None:
        time_range = utils._validate_time_range(time_range)
        download_hours = utils._get_hours(time_range)
        download_paths = []

        for hour in download_hours:
            try:
                download_path = _download_one_img_file(
                    'themis', location_code, base_url, hour, force_download
                )
                download_paths.append(download_path)
            except NotADirectoryError:
                if ignore_missing_data:
                    continue
                else:
                    raise

    return download_paths


def download_themis_skymap(location_code: str, force_download: bool = False) -> List[pathlib.Path]:
    """
    Download all of the THEMIS skymap IDL .sav files.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.

    Returns
    -------
    None

    Example
    -------
    import asilib

    | location_code = 'LUCK'
    | asilib.download_themis_skymap(location_code)
    """
    # Create the skymap directory in data/themis/skymap/location_code
    save_dir = asilib.config['ASI_DATA_DIR'] / 'themis' / 'skymap' / location_code.lower()
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
        print(f'Made directory at {save_dir}')

    base_url = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/skymaps'
    url = f'{base_url}/{location_code.lower()}/'

    # Look for all of the skymap hyperlinks, go in each one of them, and
    # download the .sav file.
    skymap_folders_relative = utils._search_hrefs(url, search_pattern=location_code.lower())
    download_paths = []

    for skymap_folder in skymap_folders_relative:
        skymap_folder_absolute = url + skymap_folder

        # Lastly, research for the skymap .sav file.
        skymap_name = utils._search_hrefs(skymap_folder_absolute, search_pattern=f'.sav')[0]
        skymap_save_name = skymap_name.replace('-%2B', '')  # Replace the unicode '+'.

        # Download if force_download=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if force_download or (not download_path.is_file()):
            utils._stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths


def download_rego_img(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    force_download: bool = False,
    ignore_missing_data: bool = True,
) -> List[pathlib.Path]:
    """
    Download one hourly REGO cdf file given the imager location and ``time``, or
    multiple hourly files given ``time_range``.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    time: datetime.datetime or str
        The date and time to download of the data. If str, ``time`` must be in the
        ISO 8601 standard.
    time_range: list of datetime.datetimes or stings
        Defined the duration of data to download. Must be of length 2.
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.
    ignore_missing_data: bool
        Flag to ignore the FileNotFoundError that is raised when ASI
        data is unavailable for that date-hour. Only used when
        ``time_range`` is specified.

    Returns
    -------
    download_paths: list
        A list of pathlib.Path objects that contain the downloaded file
        path(s).

    Example
    -------
    | from datetime import datetime
    |
    | import asilib
    |
    | location_code = 'LUCK'
    | time = datetime(2017, 4, 13, 5)
    | asilib.download_rego_img(location_code, time=time)
    """
    base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/reg'

    if (time is None) and (time_range is None):
        raise AttributeError('Neither time or time_range is specified.')
    elif (time is not None) and (time_range is not None):
        raise AttributeError('Both time and time_range can not be simultaneously specified.')

    elif time is not None:
        time = utils._validate_time(time)
        download_path = _download_one_img_file(
            'rego', location_code, base_url, time, force_download
        )
        download_paths = [
            download_path
        ]  # List for constancy with the time_range code chunk output.

    elif time_range is not None:
        time_range = utils._validate_time_range(time_range)
        download_hours = utils._get_hours(time_range)
        download_paths = []

        for hour in download_hours:
            try:
                download_path = _download_one_img_file(
                    'rego', location_code, base_url, hour, force_download
                )
                download_paths.append(download_path)
            except NotADirectoryError:
                if ignore_missing_data:
                    continue
                else:
                    raise

    return download_paths


def download_rego_skymap(location_code: str, force_download: bool = False) -> List[pathlib.Path]:
    """
    Download all of the REGO skymap IDL .sav files.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    force_download: bool
        If True, download the file even if it already exists. Useful if a prior
        data download was incomplete.

    Returns
    -------
    None

    Example
    -------
    | import asilib
    |
    | location_code = 'LUCK'
    | asilib.download_rego_skymap(location_code)
    """
    base_url = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/skymap'

    save_dir = asilib.config['ASI_DATA_DIR'] / 'rego' / 'skymap' / location_code.lower()
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
        print(f'Made directory at {save_dir}')

    url = f'{base_url}/{location_code.lower()}/'

    # Look for all of the skymap hyperlinks, go in each one of them, and
    # download the .sav file.
    skymap_folders_relative = utils._search_hrefs(url, search_pattern=location_code.lower())
    download_paths = []

    for skymap_folder in skymap_folders_relative:
        skymap_folder_absolute = url + skymap_folder

        # Lastly, research for the skymap .sav file.
        skymap_name = utils._search_hrefs(skymap_folder_absolute, search_pattern=f'.sav')[0]
        skymap_save_name = skymap_name.replace('-%2B', '')  # Replace the unicode '+'.

        # Download if force_download=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if force_download or (not download_path.is_file()):
            utils._stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths


def _download_one_img_file(asi_array_code, location_code, base_url, time, force_download):
    """
    Download one hour-long file.
    """
    # Add the location/year/month url folders onto the url
    url = f'{base_url}/{location_code.lower()}/{time.year}/{str(time.month).zfill(2)}/'

    search_pattern = f'{location_code.lower()}_{time.strftime("%Y%m%d%H")}'
    file_names = utils._search_hrefs(url, search_pattern=search_pattern)

    server_url = url + file_names[0]
    download_path = pathlib.Path(
        asilib.config['ASI_DATA_DIR'], asi_array_code.lower(), file_names[0]
    )
    if force_download or (not download_path.is_file()):
        utils._stream_large_file(server_url, download_path)
    return download_path
