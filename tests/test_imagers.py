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
    baseline_images=['test_plot_bad_image'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_plot_bad_image():
    """
    There is a bad image in TREX-RGB data taken at PINA. This test will fail
    once the data file is fixed, and this tests the asilib.Imagers() initialization
    with a single asilib.Imager() instance.
    """
    import matplotlib.pyplot as plt

    import asilib
    import asilib.asi

    asi = asilib.asi.trex_rgb('PINA', time='2021-11-04T05:25:00')
    asis = asilib.Imagers(asi)
    _, ax = plt.subplots()
    asis.plot_fisheye(ax)
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
    lat_lon_points, intensities = asis.get_points(min_elevation=min_elevation)
    assert lat_lon_points.shape == (174857, 2)
    assert intensities.shape == (174857,)
    np.testing.assert_almost_equal(
        lat_lon_points[:10, :],
        np.array([
            [  54.35538483, -126.76580811],
            [  54.34091568, -126.89642334],
            [  54.32193756, -127.0307312 ],
            [  54.29838562, -127.16894531],
            [  54.27019119, -127.31124878],
            [  54.23726654, -127.45788574],
            [  54.19952011, -127.60906982],
            [  54.15684509, -127.76501465],
            [  54.10913086, -127.92602539],
            [  54.0562439 , -128.09228516]
            ])
    )
    np.testing.assert_almost_equal(
        lat_lon_points[-10:, :],
        np.array([
            [ 64.15480804, -82.1703186 ],
            [ 64.12185669, -82.30700684],
            [ 64.09317017, -82.44506836],
            [ 64.06869507, -82.58477783],
            [ 64.04837036, -82.72647095],
            [ 64.03216553, -82.87042236],
            [ 64.02003479, -83.01693726],
            [ 64.01195526, -83.16635132],
            [ 64.00791168, -83.31893921],
            [ 64.00789642, -83.47509766]
            ])
       )
    np.testing.assert_equal(
        intensities[:10],
        np.array([2823., 2752., 2738., 2738., 2746., 2702., 2685., 2616., 2592., 2569.])
        )
    return

def test_iterate_trex_imagers():
    """
    Test the TREx imagers (time, image) iterations are in sync, and if not, that the image is 
    correctly masked. 
    """
    import asilib
    import asilib.asi

    time_range = ('2023-02-24T05:10', '2023-02-24T05:15')
    # Load all TREx imagers.
    trex_metadata = asilib.asi.trex_rgb_info()
    asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code, time_range=time_range) 
        for location_code in trex_metadata['location_code']]
        )

    _guide_times = []
    _times = []
    _images = [] 
    for _guide_time, _asi_times, _asi_images in asis:
        _guide_times.append(_guide_time)
        _times.append(_asi_times)
        _images.append(_asi_images)

    dt = np.full(np.array(_times).shape, np.nan)
    for i, (_guide_time, imager_times_i) in enumerate(zip(_guide_times, _times)):
        dt[i, :] = [(_guide_time-j).total_seconds() for j in imager_times_i]
    
    dt[np.abs(dt) > 3600*24] = np.nan

    assert np.nanmax(np.abs(dt)) == 3.297666  # Maximum unsynchronized time difference.
    assert np.all(~np.isnan(dt[:-1, :]))  # All 
    assert np.all(np.isnan(dt[-1, :]) == np.array([False, False, False,  True, False, False]))
    return

def test_iterate_rego_imagers():
    """
    Test the REGO imagers (time, image) iterations are in sync, and if not, that the image is 
    correctly masked. 

    Event from Panov+2019 https://doi.org/10.1029/2019JA026521
    """
    import asilib
    import asilib.asi

    time_range = ('2016-08-09T08:00', '2016-08-09T08:05')
    
    asis = asilib.Imagers(
        [asilib.asi.rego(location_code, time_range=time_range) 
        for location_code in ['GILL', 'FSMI', 'FSIM']]
        )

    _guide_times = []
    _times = []
    _images = [] 
    for _guide_time, _asi_times, _asi_images in asis:
        _guide_times.append(_guide_time)
        _times.append(_asi_times)
        _images.append(_asi_images)

    dt = np.full(np.array(_times).shape, np.nan)
    for i, (_guide_time, imager_times_i) in enumerate(zip(_guide_times, _times)):
        dt[i, :] = [(_guide_time-j).total_seconds() for j in imager_times_i]

    assert np.max(np.abs(dt)) == 0
    
    # 100 time stamps, 3 imagers, and x- and y- pixels for each image.
    # This will fail if any images are labeled as None. The error is: 
    # "ValueError: setting an array element with a sequence. The requested 
    # array has an inhomogeneous shape after 2 dimensions. The detected 
    # shape was (100, 3) + inhomogeneous part.
    assert np.array(_images).shape == (100, 3, 512, 512)
    return

def test_iterate_themis_imagers():
    """
    Test the THEMIS imagers (time, image) iterations are in sync, and if not, that the image is 
    correctly masked. 

    Event from Panov+2019 https://doi.org/10.1029/2019JA026521
    """
    from datetime import datetime

    import asilib
    import asilib.asi

    time_range = (
        datetime(2007, 3, 13, 5, 5, 0),
        datetime(2007, 3, 13, 5, 10, 0)
        )
    
    asis = asilib.Imagers(
        [asilib.asi.themis(location_code, time_range=time_range) 
        for location_code in ['FSIM', 'ATHA', 'TPAS', 'SNKQ']]
        )

    _guide_times = []
    _times = []
    _images = [] 
    for _guide_time, _asi_times, _asi_images in asis:
        _guide_times.append(_guide_time)
        _times.append(_asi_times)
        _images.append(_asi_images)

    dt = np.full(np.array(_times).shape, np.nan)
    for i, (_guide_time, imager_times_i) in enumerate(zip(_guide_times, _times)):
        dt[i, :] = [(_guide_time-j).total_seconds() for j in imager_times_i]

    assert np.max(np.abs(dt)) == 0.075671
    
    # 100 time stamps, 4 imagers, and x- and y- pixels for each image.
    # This will fail if any images are labeled as None. The error is: 
    # "ValueError: setting an array element with a sequence. The requested 
    # array has an inhomogeneous shape after 2 dimensions. The detected 
    # shape was (100, 3) + inhomogeneous part.
    assert np.array(_images).shape == (100, 4, 256, 256)
    return

def test_animate_map_sync_bug():
    """
    Test Imagers.animate_map() and ensure that the images are all synchronized.
    """
    import asilib
    import asilib.asi

    asi_list = [asilib.asi.themis(location, time_range=('2008-02-11T03:35', '2008-02-11T03:36'))
                for location in ['ATHA', 'FSIM', 'FSMI']]
    asis = asilib.Imagers(asi_list)
    asis.animate_map(overwrite=True)
    return

def test_animate_map():
    """
    Just check that if it crashes or not.
    """
    import asilib
    import asilib.asi

    time_range = ('2021-11-04T06:59', '2021-11-04T07:10')
    asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code, time_range=time_range) 
        for location_code in ['LUCK', 'PINA', 'GILL', 'RABB']]
        )
    asis.animate_map(lon_bounds=(-115, -85), lat_bounds=(43, 63), overwrite=True)
    return

# @matplotlib.testing.decorators.image_comparison(
#     baseline_images=['test_plot_map_eq'],
#     tol=10,
#     remove_text=True,
#     extensions=['png'],
# )
# def test_plot_map_eq():
#     """
#     Test the Imagers.plot_map_eq() method using IGRF.
#     """
#     import asilib
#     import asilib.asi
#     import matplotlib.pyplot as plt

#     time = '2021-11-04T06:59'
#     location_codes = ['GILL']
#     asis = asilib.Imagers(
#         [asilib.asi.trex_rgb(location_code, time=time) 
#         for location_code in location_codes]
#         )
#     fig, ax = plt.subplots()
#     asis.plot_map_eq(ax=ax, b_model="IGRF")
#     return