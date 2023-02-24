from __future__ import annotations  # to support the -> List[Downloader] return type
from typing import List
import pathlib
import urllib
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
    redownload: bool = False,
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
    redownload: bool
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
            redownload=redownload,
            ignore_missing_data=ignore_missing_data,
        )
    elif asi_array_code.lower() == 'rego':
        paths = download_rego_img(
            location_code,
            time=time,
            time_range=time_range,
            redownload=redownload,
            ignore_missing_data=ignore_missing_data,
        )
    return paths


def download_skymap(
    asi_array_code: str, location_code: str, redownload: bool = False
) -> List[pathlib.Path]:
    """
    Download all of the THEMIS or REGO skymap IDL .sav files.

    Parameters
    ----------
    asi_array_code: str
        The imager array name, i.e. ``THEMIS`` or ``REGO``.
    location_code: str
        The ASI station code, i.e. ``ATHA``
    redownload: bool
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
        paths = download_themis_skymap(location_code, redownload=redownload)
    elif asi_array_code.lower() == 'rego':
        paths = download_rego_skymap(location_code, redownload=redownload)
    return paths


def download_themis_img(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    redownload: bool = False,
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
    redownload: bool
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
        download_path = _download_one_img_file('themis', location_code, base_url, time, redownload)
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
                    'themis', location_code, base_url, hour, redownload
                )
                download_paths.append(download_path)
            except NotADirectoryError:
                if ignore_missing_data:
                    continue
                else:
                    raise

    return download_paths


def download_themis_skymap(location_code: str, redownload: bool = False) -> List[pathlib.Path]:
    """
    Download all of the THEMIS skymap IDL .sav files.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    redownload: bool
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

        # Download if redownload=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if redownload or (not download_path.is_file()):
            utils._stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths


def download_rego_img(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    redownload: bool = False,
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
    redownload: bool
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
        download_path = _download_one_img_file('rego', location_code, base_url, time, redownload)
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
                    'rego', location_code, base_url, hour, redownload
                )
                download_paths.append(download_path)
            except NotADirectoryError:
                if ignore_missing_data:
                    continue
                else:
                    raise

    return download_paths


def download_rego_skymap(location_code: str, redownload: bool = False) -> List[pathlib.Path]:
    """
    Download all of the REGO skymap IDL .sav files.

    Parameters
    ----------
    location_code: str
        The ASI station code, i.e. ``ATHA``
    redownload: bool
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

        # Download if redownload=True or the file does not exist.
        download_path = pathlib.Path(save_dir, skymap_save_name)
        download_paths.append(download_path)
        if redownload or (not download_path.is_file()):
            utils._stream_large_file(skymap_folder_absolute + skymap_name, download_path)
    return download_paths


def _download_one_img_file(asi_array_code, location_code, base_url, time, redownload):
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
    if redownload or (not download_path.is_file()):
        utils._stream_large_file(server_url, download_path)
    return download_path


# TODO: When cleaning up imager, remove all above
class Downloader:
    """
    Traverses and lists the directory structure on a server and download files.

    Parameters
    ----------
    url: str
        The dataset URL. If it points to a folder it must end with a "/".
    download_dir: str or pathlib.Path
        The download directory. Must either specify here, or when you call
        Downloader.download().

    Example
    -------
    | # List all of the SAMPEX-HILT State4 files and download the first one.
    |
    | import sampex
    |
    | d = sampex.Downloader(
    |     'https://izw1.caltech.edu/sampex/DataCenter/DATA/HILThires/State4/',
    |     download_dir=sampex.config['data_dir']
    | )
    | paths = d.ls(match='*.txt*')
    | print(f"The downloaded files are: {paths}")
    |
    | print(f"The first file's name is: {paths[0].name} at url {paths[0].url}")
    | path = paths[0].download(stream=False)
    | print(f'The file was downloaded to {path}')
    """

    def __init__(self, url: str, download_dir=None) -> None:
        self.url = url
        self.download_dir = download_dir
        return

    def ls(self, match: str = '*') -> List[Downloader]:
        """
        List files and folders in self.url.

        Parameters
        ----------
        match: str
            An optional string pattern to match.

        Return
        ------
        list
            A list of full URLs.
        """
        matched_hrefs = self._search_hrefs(self.url, match=match)
        cls = type(self)
        downloaders = [None] * len(matched_hrefs)
        for i, matched_href in enumerate(matched_hrefs):
            new_url = urllib.parse.urljoin(self.url, matched_href, allow_fragments=True)
            downloaders[i] = cls(new_url, download_dir=self.download_dir)
        return downloaders

    def download(
        self, download_dir=None, redownload: bool = False, stream: bool = False
    ) -> pathlib.Path:
        """
        Downloads file from self.url to the download_dir directory.

        Parameters
        ----------
        download_dir: str or pathlib.Path
            The parent directory where to save the data to. Set it either
            here or when initializing the class. This
        redownload: bool
            Will redownload an existing file.
        stream: bool
            Download the data in one chunk if False, or break it up and download it
            in 5 Mb chunks if True.

        Returns
        -------
        pathlib.Path
            The full path to the file.
        """
        if download_dir is None and self.download_dir is None:
            raise ValueError(
                f'download_dir kwarg needs to be set either '
                f'in Downloader() or Downloader.download.'
            )
        if download_dir is not None:
            self.download_dir = download_dir

        self.download_dir = pathlib.Path(self.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        file_name = pathlib.Path(self.url).name
        download_path = self.download_dir / file_name

        if (download_path.exists()) and (not redownload):
            return download_path

        if stream:
            r = requests.get(self.url, stream=True, timeout=5)
            r.raise_for_status()
            file_size = int(r.headers.get('content-length'))
            downloaded_bites = 0

            megabyte = 1024 * 1024
            try:
                with open(download_path, 'wb') as f:
                    for data in r.iter_content(chunk_size=10 * megabyte):
                        f.write(data)
                        # Update the downloaded % in the terminal.
                        downloaded_bites += len(data)
                        download_percent = round(100 * downloaded_bites / file_size)
                        download_str = "#" * (download_percent // 5)
                        print(
                            f'Downloading {file_name}: |{download_str:<20}| {download_percent}%',
                            end='\r',
                        )
                print()  # Add a newline
            except (Exception, KeyboardInterrupt, SystemExit) as err:
                download_path.unlink()
                raise RuntimeError(
                    f'Download interrupted. Partially downloaded file ' f'{download_path} deleted.'
                ) from err
        else:
            r = requests.get(self.url, timeout=5)
            with open(download_path, 'wb') as f:
                f.write(r.content)
            print(f'Downloaded {file_name}.')
        return download_path

    @property
    def name(self):
        """
        Get the url filename
        """
        _url = pathlib.Path(self.url)
        if _url.suffix != '':
            # url points to a filename. pathlib is not designed for
            # urls, but this is the easiest way to get the name.
            return _url.name
        else:
            return None

    def _search_hrefs(self, url: str, match: str = '*') -> List[str]:
        """
        Given a url string, find all hyper references matching the
        string match. The re module does the matching.

        Parameters
        ----------
        url: str
            A url
        match: str (optional)
            The regex match to compare the hyper references to. The default is
            to match everything (a wildcard)

        Returns
        -------
        List(str)
            A list of hyper references that matched the match string.

        Raises
        ------
        FileNotFoundError
            If no hyper references were found.
        """
        request = requests.get(url, timeout=5)
        request.raise_for_status()
        soup = BeautifulSoup(request.content, 'html.parser')

        match = match.replace('*', '.*')  # regex wildcard
        href_objs = soup.find_all('a', href=re.compile(match))
        # I found "?" to be in the column names so we should ignore them.
        matched_hrefs = [
            href_obj['href'] for href_obj in href_objs if href_obj['href'].find('?') == -1
        ]
        if len(matched_hrefs) == 0:
            raise FileNotFoundError(
                f'The url {url} does not contain any hyper '
                f'references containing the match kwarg="{match}".'
            )
        return matched_hrefs

    def __repr__(self) -> str:
        params = f'{self.url}, download_dir={self.download_dir},'
        return f'{self.__class__.__qualname__}(' + params + ')'

    def __str__(self) -> str:
        return (
            f'{self.__class__.__qualname__} with url={self.url} and '
            f'download_dir={self.download_dir}'
        )
