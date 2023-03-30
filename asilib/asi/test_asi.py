"""
An ASI for testing asilib.Imager.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import asilib


def test_asi(location_code, time=None, time_range=None, alt=110):
    """
    A fake ASI array to test asilib.Imager. This array consists of three locations.
    1. GILL,
    2. ATHA, and
    3. TPAS.
    """
    location_code = location_code.upper()
    locations = asi_info()
    _location = locations.loc[locations.index == location_code, :]

    # The metadata about the chosen ASI. 
    meta = {
        'array': 'TEST',
        'location': location_code,
        'lat': _location['lat'].to_numpy(),
        'lon': _location['lon'].to_numpy(),
        'alt': _location['alt'].to_numpy() / 1e3,  # km 
        'cadence': 10,
        'resolution': (512, 512),
    }

    skymap = get_skymap(meta, alt=alt)

    return
    # return asilib.Imager(None, meta, skymap)

def asi_info()->pd.DataFrame:
    """
    The test ASI has three locations, a, b, and c.
    """
    locations = pd.DataFrame(data=np.array([
        [56.3494, -94.7056, 500],
        [54.7213, -113.285, 1000],
        [53.8255, -101.2427, 0],
        ]), 
        columns=['lat', 'lon', 'alt'],
        index=['GILL', 'ATHA', 'TPAS']
        )
    return locations

def get_skymap(meta, alt):
    """
    Create a skymap based on the ASI location in the metadata.
    """
    assert alt in [90, 110, 150], (f'The {alt} km altitude does not have a corresponding map: '
                                   'valid_altitudes=[90, 110, 150].')
    skymap = {}
    lon_bounds = [meta['lon']-10*(alt/110), meta['lon']+10*(alt/110)]
    lat_bounds = [meta['lat']-10*(alt/110), meta['lat']+10*(alt/110)]

    _lons, _lats = np.meshgrid(
        np.linspace(*lon_bounds, num=meta['resolution'][0]),
        np.linspace(*lat_bounds, num=meta['resolution'][1])
        )
    std = 5*(alt/110)
    dst = np.sqrt((_lons-meta['lon'])**2 + (_lats-meta['lat'])**2)
    # the 105 multiplier could be 90, but I chose 105 (and -15 offset) to make the 
    # skymap realistic: the edges are NaNs.
    elevations =  105*np.exp(-dst**2 / (2.0 * std**2))-15
    elevations[elevations < 0] = np.nan
    # These are the required skymap keys for asilib.Imager to work.
    skymap['el'] = elevations
    skymap['alt'] = alt
    skymap['lon'] = _lons
    skymap['lon'][~np.isfinite(skymap['el'])] = np.nan
    skymap['lat'] = _lats
    skymap['lat'][~np.isfinite(skymap['el'])] = np.nan
    skymap['path'] = __file__ 

    # Calculate angle using dot product equation of a northward-pointing unit vector and
    # the (_lons, _lats) grid.
    dot_product = 0*(_lons-meta['lon']) + 1*(_lats-meta['lat'])
    norm = 1*np.sqrt((_lons-meta['lon'])**2 + (_lats-meta['lat'])**2)
    skymap['az'] = np.arccos(dot_product/norm)
   
    p = plt.pcolormesh(skymap['az'])
    plt.colorbar(p)
    plt.show()
    return skymap

if __name__ == '__main__':
    asi = test_asi('GILL')
    pass