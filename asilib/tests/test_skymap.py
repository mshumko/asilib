import asilib.asi
import asilib.skymap

import matplotlib.testing.decorators
import matplotlib.pyplot as plt

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_geodetic_skymap'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
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
    fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(4, 8))

    _ref_lon_skymap, _ref_lat_skymap, _= asi._mask_low_horizon(
        asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'], 1E-9
        )

    asi._pcolormesh_nan(_ref_lon_skymap, _ref_lat_skymap, _ref_lat_skymap, ax[0])
    asi._pcolormesh_nan(asilib_lon_skymap, asilib_lat_skymap, asilib_lat_skymap, ax[1])
    return