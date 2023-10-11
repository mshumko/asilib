import asilib.asi

import matplotlib.testing.decorators
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(4, 8))
    ref_asi._pcolormesh_nan(
        ref_asi.skymap['lon'], ref_asi.skymap['lat'], ref_asi.skymap['lat'], ax[0]
        )
    asi._pcolormesh_nan(asi.skymap['lon'], asi.skymap['lat'], asi.skymap['lat'], ax[1])
    return