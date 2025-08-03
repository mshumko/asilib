from __future__ import annotations  # to support the -> List[Downloader] return type
from typing import List
import pathlib
import urllib
import re

from bs4 import BeautifulSoup
import requests


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
    TODO: Add an example.
    """

    def __init__(self, url: str, download_dir=None, headers={}) -> None:
        self.url = url
        self.download_dir = download_dir
        self.headers = headers
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
            downloaders[i] = cls(new_url, download_dir=self.download_dir, headers=self.headers)
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
            with requests.Session() as s:
                r = s.get(self.url, stream=True, timeout=5, headers=self.headers)
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
            r = requests.get(self.url, timeout=5, headers=self.headers)
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
        with requests.Session() as s:
            request = s.get(url, timeout=5, headers=self.headers)
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
