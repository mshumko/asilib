"""
An ASI for testing asilib.Imager.
"""
import pandas as pd
import numpy as np

import asilib


def test_asi(location_code, time, time_range):
    """
    A fake ASI array to test asilib.Imager. This array consists of three locations.
    1. GILL,
    2. ATHA, and
    3. TPAS.
    """
    location_code = location_code.upper()
    locations = asi_info()
    _location = locations.loc[locations['name'] == location_code, :]

    # The metadata about the chosen ASI. 
    meta = {
        'array': 'TEST',
        'location': location_code,
        'lat': _location['lat'],
        'lon': _location['lon'],
        'alt': _location['alt'] / 1e3,  # km 
        'cadence': 10,
        'resolution': (512, 512),
    }

    skymap = {
        'lat': _skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        'lon': _skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        'alt': _skymap['FULL_MAP_ALTITUDE'][alt_index] / 1e3,
        'el': _skymap['FULL_ELEVATION'],
        'az': _skymap['FULL_AZIMUTH'],
        'path': _skymap['PATH'],
    }

    return asilib.Imager(data, meta, skymap)

def asi_info()->pd.DataFrame:
    """
    The test ASI has three locations, a, b, and c.
    """
    locations = pd.DataFrame(data=np.array([
        ['GILL', 56.3494, -94.7056, 500],
        ['ATHA', 54.7213, -113.285, 1000],
        ['TPAS', 53.8255, -101.2427, 0],
        ]), columns=['name', 'lat', 'lon', 'alt'])
    return locations

if __name__ == '__main__':
    print(asi_info('GILL'))