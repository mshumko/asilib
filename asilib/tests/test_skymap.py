import asilib.asi
import asilib.skymap

import matplotlib.pyplot as plt


def test_geodetic_skymap():
    """
    Compare the (lat, lon) skymaps between the asilib and the official implementation.
    """
    time = '2020-01-01'
    asi = asilib.asi.themis('GILL', time=time, load_images=False)
    
    asilib_lat_skymap, asilib_lon_skymap = asilib.skymap.geodetic_skymap(
        (asi.meta['lat'], asi.meta['lon'], asi.meta['alt']),
        asi.skymap['az'], asi.skymap['el'], 110
    )
    fig, ax = plt.subplots(2, sharex=True, sharey=True)

    asi._pcolormesh_nan(asi.skymap['lon'], asi.skymap['lat'], asi.skymap['lat'], ax[0])
    asi._pcolormesh_nan(asilib_lon_skymap, asilib_lat_skymap, asilib_lat_skymap, ax[1])
    ax[0].set(title='Reference')
    plt.show()
    return