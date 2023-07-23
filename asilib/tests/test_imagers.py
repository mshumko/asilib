"""
Tests for the asilib.Imagers class.
"""
import matplotlib.testing.decorators
import numpy as np


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_donovan_arc_example'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_donovan_arc_example():
    from datetime import datetime
    import matplotlib.pyplot as plt
    import asilib
    import asilib.map
    import asilib.asi
    time = datetime(2007, 3, 13, 5, 8, 45)
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
    map_alt = 110
    min_elevation = 2
    ax = asilib.map.create_simple_map(lon_bounds=(-140, -60), lat_bounds=(40, 82))
    _imagers = []
    for location_code in location_codes:
        _imagers.append(asilib.asi.themis(location_code, time=time, alt=map_alt))
    asis = asilib.Imagers(_imagers)
    asis.plot_map(ax=ax, overlap=False, min_elevation=min_elevation)
    ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_fisheye'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_plot_fisheye():
    from datetime import datetime

    import matplotlib.pyplot as plt
    import asilib
    import asilib.asi

    time = datetime(2007, 3, 13, 5, 8, 45)
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']

    fig, ax = plt.subplots(1, len(location_codes), figsize=(12, 3.5))

    _imagers = []

    for location_code in location_codes:
        _imagers.append(asilib.asi.themis(location_code, time=time))

    for ax_i in ax:
        ax_i.axis('off')

    asis = asilib.Imagers(_imagers)
    asis.plot_fisheye(ax=ax)

    plt.suptitle('Donovan et al. 2008 | First breakup of an auroral arc')
    plt.tight_layout()
    return


def test_get_points():
    from datetime import datetime
    
    import asilib
    import asilib.asi
    
    time = datetime(2007, 3, 13, 5, 8, 45)
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
    map_alt = 110
    min_elevation = 2
    
    _imagers = []
    
    for location_code in location_codes:
        _imagers.append(asilib.asi.themis(location_code, time=time, alt=map_alt))
    
    asis = asilib.Imagers(_imagers)
    lon_lat_points, intensities = asis.get_points(min_elevation=min_elevation)
    assert lon_lat_points.shape == (175759, 2)
    assert intensities.shape == (175759,)
    np.testing.assert_almost_equal(
        lon_lat_points[:10, :],
        np.array([[-126.16152954,   54.36151505],
            [-126.27648926,   54.36903   ],
            [-126.39419556,   54.37219238],
            [-126.51489258,   54.37098694],
            [-126.63867188,   54.36539459],
            [-126.76580811,   54.35538483],
            [-126.89642334,   54.34091568],
            [-127.0307312 ,   54.32193756],
            [-127.16894531,   54.29838562],
            [-127.31124878,   54.27019119]])
       )
    np.testing.assert_equal(
        intensities[:10],
        np.array([2930., 2949., 2954., 2881., 2865., 2823., 2752., 2738., 2738., 2746.])
    )
    return