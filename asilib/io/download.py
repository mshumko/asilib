from typing import List, Union
import pathlib
import re

from bs4 import BeautifulSoup
import requests

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
        skymap_name = utils._search_hrefs(skymap_folder_absolute, search_pattern='.sav')[0]
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
        skymap_name = utils._search_hrefs(skymap_folder_absolute, search_pattern='.sav')[0]
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

#TODO: When cleaning up imager, remove all above
class Downloader:
    def __init__(self, base_url:str) -> None:
        """
        Handles traversing the directory structure on a server and download files.
        Parameters
        ----------
        base_url: str
            The base URL for the dataset (excludes directories contaning dates, 
            for example).
        Example
        -------
        d = Download(
            'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/'
            )
        d.find_file(
            ['2014', '05', '05', 'gill*', 'ut05', '20140505_0505_gill*.pgm.gz']
            )
        d.download()
        """
        self.base_url = base_url

        # Check that the server status code is not
        # between 400-599 (error).
        r = requests.get(self.base_url)
        status_code = r.status_code
        if status_code // 100 in [4, 5]:
            raise ConnectionError(f'{self.base_url} returned a {status_code} response (error).')

        if self.base_url[-1] != '/':
            self.base_url += '/'
        return

    def find_url(self, subdirectories=[], filename=None):
        """
        Descends the directory tree starting from self.base_url using folders
        specified by path.
        Parameters
        ----------
        subdirectories: list
            A list of strings containing subfolders and a filename at the end.
            You can use a wildcard character (*), to search a unique subdirectory 
            or file name. The search must only result in exactly one match or an
            AssertionError will be raised.
        filename: str
            An optional filename pattern (including a possible wildcard) that will
            be used to search the final subdirectory for urls.
        
        Return
        ------
        list
            A list of full URLs.
        """
        self.url = self.base_url
        assert all([isinstance(i, str) for i in subdirectories]), 'The subdirectories must be all strings.'

        for subdirectory in subdirectories:
            if '*' not in subdirectory:
                self.url = self.url + subdirectory
            else:
                matched_hrefs = self._search_hrefs(self.url, search_pattern=subdirectory)
                assert len(matched_hrefs) == 1, (f'Only one href is allowed but '
                    f'{len(matched_hrefs)} were found in {self.url}. matched_hrefs (links)={matched_hrefs}')
                self.url = self.url + matched_hrefs[0]

            if self.url[-1] != '/':
                self.url += '/'

        if filename is not None:
            file_hrefs = self._search_hrefs(self.url, search_pattern=filename)
            _url = []
            for file_href in file_hrefs:
                _url.append(self.url + file_href)
            self.url = _url
        else:
            self.url = [self.url]  # For consistency
        return self.url

    def download(self, save_dir, overwrite=False) -> None:
        """
        Downloads (streams) a file from self.url to the save_dir directory. I decided
        to use streaming so that a large file wont overwhelm the user's RAM.
        Parameters
        ----------
        save_dir: str or pathlib.Path
            The parent directory where to save the data to.
        overwrite: bool
            Will download overwrite an existing file. 
        Returns
        -------
        list
            A list of local paths where the data was saved to. 
        """
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = []

        for url in self.url:
            save_name = url.split('/')[-1]
            _save_path = save_dir / save_name
            self.save_path.append(_save_path)

            if (_save_path.exists()) and (not overwrite):
                continue

            r = requests.get(url, stream=True)
            file_size = int(r.headers.get('content-length'))
            downloaded_bites = 0

            megabyte = 1024 * 1024

            with open(_save_path, 'wb') as f:
                for data in r.iter_content(chunk_size=5*megabyte):
                    f.write(data)
                    # Update the downloaded % in the terminal.
                    downloaded_bites += len(data)
                    download_percent = round(100 * downloaded_bites / file_size)
                    download_str = "#" * (download_percent // 5)
                    print(f'Downloading {save_name}: |{download_str:<20}| {download_percent}%', end='\r')
            print()  # Add a newline
        return self.save_path


    def _search_hrefs(self, url: str, search_pattern: str = '.cdf') -> List[str]:
        """
        Given a url string, this function returns all of the
        hyper references (hrefs, or hyperlinks). If search_pattern is not
        specified, a default '.cdf' value is assumed and this function
        will return all hrefs with the CDF extension. If no hrefs containing
        search_pattern are found, this function raises a FileNotFound.
        The search is case-insensitive.
        Parameters
        ----------
        url: str
            A url in string format
        search_pattern: str (optional)
            Find the exact search_pattern text contained in the hrefs.
            By default all hrefs matching the extension ".cdf" are returned.
        Returns
        -------
        hrefs: List(str)
            A list of hrefs that contain the search_pattern.
        Raises
        ------
        FileNotFoundError
            If a hyper reference (a folder or a file) is not found on the
            server. This is raised if the data does not exist.
        """
        matched_hrefs = []

        request = requests.get(url)
        # request.status_code
        soup = BeautifulSoup(request.content, 'html.parser')

        search_pattern = search_pattern.replace('*', '.*')  # regex wildcard
        for href in soup.find_all('a', href=True):
            match = re.search(search_pattern, href['href'])
            if match:
                matched_hrefs.append(href['href'])
        if len(matched_hrefs) == 0:
            raise FileNotFoundError(
                f'The url {url} does not contain any hyper '
                f'references containing the search_pattern="{search_pattern}".'
            )
        return matched_hrefs

if __name__ == '__main__':
    d = Downloader('https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/')
    d.find_file(['2014', '05', '05', 'gill*', 'ut05', '20140505_0505_gill*.pgm.gz'])
    d.download() 