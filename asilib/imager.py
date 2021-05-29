"""
The Imager class handles the ASI data downloading, loading, analyzing, and plotting.
"""

supported_arrays = ['REGO', 'THEMIS']


class Imager:
    """
    Handles the downloading, loading, analyzing, and plotting of ASI data.

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------

    """

    def __init__(self, array, stations=None, time_range=None):
        """
        Initializes the Imager class.

        Parameters
        ----------
        array: str
            The ASI array. Must be either 'THEMIS' or 'REGO'
        stations: str, or list of strings, optional
            Station or stations to load and analyze. If `station=None`, all 
            stations from the array will be loaded.

        Methods
        -------
        download(time_range=None)


        """
        self.array = array.upper()
        self.stations = stations
        self.time_range = time_range

        self._load_array_attributes()
        self._check_array_code()
        self._check_station_codes()

        return

    def download(self, time_range=None):
        """
        Downloads data from time_range.

        Parameters
        ----------
        time_range: list , optional
            The start and end time to download the ASI data. 
        """
        self._check_time_range_exists(time_range)

        raise NotImplementedError
        return

    def load(self, time_range=None):
        """
        Downloads data from time_range.

        Parameters
        ----------
        time_range: list , optional
            The start and end time to download the ASI data. 
        """
        
        raise NotImplementedError
        return

    def plot(self, type):
        """
        Plots the ASI data
        """
        raise NotImplementedError
        return

    def _load_array_attributes(self):
        """
        Loads the stations from asilib/data/asi_stations.csv that match the self.array code
        into a pd.DataFrame.

        Returns
        -------
        self.array_attributes
        """

        # TODO: Isaac, this method should load asilib/data/asi_stations.csv into a pd.DataFrame and filter by camera array (self.array).
        raise NotImplementedError
        return

    def _check_array_code(self):
        """
        Checks that the array code is valid.
        """
        assert self.array in supported_arrays, (f"{self.array} array code is invalid, must be "
            f" in {supported_arrays}. Case insensitive.")
        return

    def _check_station_codes(self):
        """
        Checks that the station or station codes are all valid stations.
        """
        if hasattr(self.stations, '__len__'):
            invalid_stations = []

            for station in self.stations:
                if station not in self.array_attributes['station']:
                    invalid_stations.append(station)
            
            assert len(invalid_stations) > 0, (f'{invalid_stations} stations '
                f'were not in {self.array_attributes["station"]}')
        
        else:
            assert self.station in self.array_attributes['station'], (f'{self.stations} not in '
                f'{self.array_attributes["station"]}')
        return

    def _check_time_range(self, time_range):
        """
        Check that the time_range variable exists in the class and is valid.
        """
        if (time_range == None) and (not hasattr(self, 'time_range')):
            raise AttributeError('Imager.time_range not found. It must be supplied '
                                'to __init__() or the called method')
        return
