import asilib.asi

import matplotlib.testing.decorators
import matplotlib.pyplot as plt

from asilib.imager import Skymap_Cleaner

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_geodetic_skymap'],
    tol=20,
    remove_text=True,
    extensions=['png'],
)
def test_geodetic_skymap():
    """
    Compare the (lat, lon) skymaps between the asilib and the official implementation.
    """
    time = '2020-01-01'
    ref_asi = asilib.asi.themis('GILL', time=time, load_images=False, alt=110)
    asi = asilib.asi.themis('GILL', time=time, load_images=False, custom_alt=True, alt=110)

    _ref_cleaner = Skymap_Cleaner(ref_asi.skymap['lon'], ref_asi.skymap['lat'], ref_asi.skymap['el'])
    _ref_lon, _ref_lat = _ref_cleaner.remove_nans()

    _custom_cleaner = Skymap_Cleaner(asi.skymap['lon'], asi.skymap['lat'], asi.skymap['el'])
    _custom_lon, _custom_lat = _custom_cleaner.remove_nans()

    _, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(4, 8))
    ax[0].pcolormesh(_ref_lon, _ref_lat, _ref_lat)
    ax[1].pcolormesh(_custom_lon, _custom_lat, _custom_lat)
    return