"""
Tests the rego() data loading and the example plotting functions.
"""
from datetime import datetime

import requests
import pytest
import matplotlib.testing.decorators

import asilib.asi

##########################################
############# TEST LOADERS ###############
##########################################


def test_trex_nir_time():
    """
    Tests that one file is loaded and themis() returns the correct file.
    """
    # Calls the download function
    asi = asilib.asi.trex_nir('rabb', time='2022-03-25T08:40', redownload=True)
    assert asi.data.time == datetime(2022, 3, 25, 8, 40)
    assert asi.data.image[0, 0] == 324
    assert asi.skymap['path'].name == 'nir_skymap_rabb_20220301-%2B_v01.sav'
    return


def test_trex_nir_time_range():
    """
    Tests that multiple files are loaded using themis()'s time_range kwarg.
    """
    time_range = ('2020-03-21T05:00', '2020-03-21T05:10')
    asi = asilib.asi.trex_nir('gill', time_range=time_range)
    assert asi.file_info['path'][0].name == '20200321_0500_gill_nir-219_8446.pgm.gz'
    assert asi.file_info['path'][-1].name == '20200321_0509_gill_nir-219_8446.pgm.gz'
    asi.data
    assert asi.data.time[0] == datetime(2020, 3, 21, 5, 0, 0)
    assert asi.data.time[-1] == datetime(2020, 3, 21, 5, 9, 54)
    assert asi.data.time.shape == (100,)
    assert asi.data.image.shape == (100, 256, 256)
    return


def test_trex_nir_no_file():
    """
    Tests that trex_nir() returns a FileNotFound error when we try to download
    non-existent data.
    """
    with pytest.raises(requests.exceptions.HTTPError):
        asilib.asi.trex_nir('gill', time='2011/07/07T02:00')
    return


# def test_trex_nir_partial_files():
#     """
#     Tests that themis() returns a FileNotFound error when we try to download
#     non-existant data.
#     """
#     with pytest.raises(FileNotFoundError):
#         themis.themis('pina', time_range=['2011/07/07T04:15', '2011/07/07T04:23'], missing_ok=False)

#     img = themis.themis(
#         'pina',
#         time_range=['2011/07/07T04:15', '2011/07/07T04:23:30'],
#         missing_ok=True,
#         redownload=True,
#     )
#     assert img.file_info['path'][0].name == '20110707_0421_pina_themis18_full.pgm.gz'
#     assert img.file_info['path'][1].name == '20110707_0422_pina_themis18_full.pgm.gz'
#     assert img.file_info['path'][2].name == '20110707_0423_pina_themis18_full.pgm.gz'
#     return


# def test_trex_nir_asi_meta():
#     """
#     Checks that the THEMIS ASI metadata is correct.
#     """
#     img = themis.themis(
#         'pina',
#         time_range=['2011/07/07T04:20', '2011/07/07T04:22:00'],
#         missing_ok=True,
#         redownload=False,
#     )
#     assert img.meta == {
#         'array': 'THEMIS',
#         'location': 'PINA',
#         'lat': 50.15999984741211,
#         'lon': -96.07000732421875,
#         'alt': 0.0,
#         'cadence': 3,
#         'resolution': (256, 256),
#     }
#     return


##########################################
############# TEST EXAMPLES ##############
##########################################
@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_trex_nir_fisheye_map_example'],
    tol=20,
    remove_text=True,
    extensions=['png'],
)
def test_trex_nir_fisheye_map_example():
    """
    Plot a fisheye lens image and map it onto a map.
    """
    import asilib
    import asilib.map
    import asilib.asi
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121)
    bx = asilib.map.create_map(fig_ax=(fig, 122), lon_bounds=(-102, -86), lat_bounds=(51, 61))

    asi = asilib.asi.trex_nir('gill', time='2020-03-21T06:00')
    asi.plot_fisheye(ax=ax)
    asi.plot_map(ax=bx)
    plt.tight_layout()
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_trex_nir_keogram_example'], tol=20, remove_text=True, extensions=['png']
)
def test_trex_nir_keogram_example():
    """
    Plot a keogram.
    """
    import asilib
    import asilib.map
    import asilib.asi
    import matplotlib.pyplot as plt

    time_range = ('2020-03-21T05:00', '2020-03-21T05:10')
    fig, ax = plt.subplots(2, sharex=True)
    asi = asilib.asi.trex_nir('gill', time_range=time_range)
    asi.plot_keogram(ax=ax[0])
    asi.plot_keogram(ax=ax[1], aacgm=True)
    ax[0].set_title(f'TREX_NIR GILL keogram | {time_range}')
    ax[0].set_ylabel('Geo Lat')
    ax[1].set_ylabel('Mag Lat')
    ax[1].set_xlabel('Time')
    return
