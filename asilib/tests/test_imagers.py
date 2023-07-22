"""
Tests for the asilib.Imagers class.
"""
import matplotlib.testing.decorators

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