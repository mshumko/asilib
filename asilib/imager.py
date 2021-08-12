import importlib
import pkgutil

class Imager:
    """
    The core class for the downloading, loading, analyzing, and plotting ASI data.
    TODO: Rewrite doctring
    Methods 
    ------- 
    download(time_range=None)
        Download data in bulk.
    load(time_range=None)
        Load data from the array camera system for a list of stations (all by default)
    plot(type, ax=None)
        Plots the ASI data in a few supported formats.

    Attributes
    ----------
    array_attributes: pd.DataFrame
        A table of array imagers and their locations
    data_avaliability: pd.DataFrame
        A table of data avaliability with station codes for the index and time for columns.
        The table values are yes for avaliable data and no for unavailable data (doesn't exist).
    data: dict  
        A dictionary full of ASI time stamps and images
        # TODO: Think about how to store self.data for:
        #   1) one station for one hour 
        #   2) one station for multiple hours
        #   3) multiple stations for one hour
        #   4) multiple stations for multiple hours.
    cal: dict
        A dictionary contaning the calibration data from all of the loaded stations. 
    array: str
        The camera array
    stations: str or list[str]
        The stations represented in this instance.
    time_range: list[datetime]
        The ASI time_range.
    """
    def __init__(self, array, location, instrument=None) -> None:
        # Discover and initialize the Plugin class that matches the array code.
        # https://packaging.python.org/guides/creating-and-discovering-plugins/
        for _, name, _ in pkgutil.iter_modules():
            if name == f'asilib_{array.lower()}':
                self._plugin = importlib.import_module(name).Plugin(
                    location, instrument=instrument
                    )

        if not hasattr(self, '_plugin'):
            raise ImportError(
                f'Did not find an asilib plugin package for {array.upper()}. '
                f'It should be named "asilib_{array.lower()}".'
                )
        return

    def download_img(self, time):
        self._check_time(time)
        self._plugin.download_img(time)
        return

    def download_cal(self, time):
        self._check_time(time)
        self._plugin.download_cal(time)
        return

    def load_img(self, time):
        self._check_time(time)
        self._plugin.load_img(time)
        return

    def load_cal(self, time):
        self._check_time(time)
        self._plugin.load_cal(time)
        return

    def _check_time(self, time):
        """
        Checks and, if necessary, parses the time(s) into datetime objects.
        """
        return

if __name__ == '__main__':
    im = Imager('THEMIS', 'GILL')
