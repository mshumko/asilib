import pathlib

import asilib.io.download as download


class File_Manager:
    def __init__(self, base_dir, base_url=None) -> None:
        """
        Finds locally, downloads, and deletes files.
        """
        self.base_dir = base_dir
        self.base_url = base_url
        return

    def find_file(self, filename, subdirectories=[], overwrite=False) -> pathlib.Path:
        """
        Finds a local file path that matches the filename pattern. If the file is not
        found locally, it will use download.Download() to download it and then return
        the local file path.

        Parameters
        ----------
        filename: str
            Search for a file name containing optional wildcard characters.
        subdirectories: list
            The sub-directories of base_url to traverse to search for a file that matches
            filename.
        overwrite: bool
            Download the file regardless of it is exists locally.
        """
        self.overwrite = overwrite
        base_url_missing_error_str = (
            f'The File_Manager is unable to download a file matching {filename} '
            f'because the base_url kwarg is unspecified.')
        
        if overwrite:
            if self.base_url is not None:
                file_path = self._download_file(subdirectories, filename)
            else:
                raise ValueError(base_url_missing_error_str)
        else:
            # First, look for a local file.
            file_path = self._find_local_path(filename, error=False)
            if file_path is None:
                # No local file found, try to download.
                if self.base_url is None:
                    raise ValueError(base_url_missing_error_str)
                file_path = self._download_file(subdirectories, filename)
        return file_path

    def delete(self, filename):
        raise NotImplementedError

    def _find_local_path(self, filename_pattern, error=False):
        """
        Checks if a file locally exists in the parent_dir and its subdirectories.

        Parameters
        ---------- 
        filename_pattern: str
            The filename pattern to search for. It is passed into 
            pathlib.Path.rglob(). It can have wildcards, but will raise
            a FileNotFoundError if exactly one file is not found.
        error: bool
            Raise a FileNotFound error if True and a file is not found
        """
        matched_paths = list(pathlib.Path(self.base_dir).rglob(filename_pattern))
        if len(matched_paths) != 1:
            if error:
                raise FileNotFoundError(
                    f'{len(matched_paths)} file paths found in {self.base_dir} '
                    f'(and its subdirectories) that match the '
                    f'{filename_pattern} pattern.'
                    )
            else:
                return None
        return matched_paths[0]
    
    def _download_file(self, subdirectories, filename):
        """
        A wrapper to download a file from self.base_url/path_list.
        """
        self.download = download.Downloader(self.base_url)
        self.download.find_url(subdirectories=subdirectories, filename=filename)
        return self.download.download(self.base_dir, overwrite=self.overwrite)[0]
