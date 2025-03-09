"""
Tests the themis() data loading and the example plotting functions.
"""
from datetime import datetime

import requests
import pytest
import matplotlib.testing.decorators

import asilib.asi

##########################################
############# TEST LOADERS ###############
##########################################


def test_themis_time():
    """
    Tests that one file is loaded and themis() returns the correct file.
    """
    # Calls the download function
    asi = asilib.asi.themis('gill', time='2014-05-05T04:49:10', redownload=True)
    # asi.file_info should not be accessed by the user.
    assert asi.file_info['time'] == datetime(2014, 5, 5, 4, 49, 10)
    # But asi.data should be
    assert asi.data.time == datetime(2014, 5, 5, 4, 49, 9, 37070)
    assert asi.data[0] == datetime(2014, 5, 5, 4, 49, 9, 37070)
    assert asi.data.image[0, 0] == 3464
    assert asi.data[1][0, 0] == 3464
    assert asi.data.image[-1, -1] == 3477
    assert asi.skymap['path'].name == 'themis_skymap_gill_20130103-%2B_vXX.sav'
    # Does not call the download function
    asi2 = asilib.asi.themis('gill', time='2014-05-05T04:49:10', redownload=False)
    assert asi2.file_info['time'] == datetime(2014, 5, 5, 4, 49, 10)
    return


def test_themis_time_range():
    """
    Tests that multiple files are loaded using themis()'s time_range kwarg.
    """
    img = asilib.asi.themis('gill', time_range=['2014-05-05T05:10', '2014-05-05T05:12'])
    assert img.file_info['path'][0].name == '20140505_0510_gill_themis19_full.pgm.gz'
    assert img.file_info['path'][1].name == '20140505_0511_gill_themis19_full.pgm.gz'

    assert img.data.time[0] == datetime(2014, 5, 5, 5, 10, 0, 30996)
    assert img.data.time[-1] == datetime(2014, 5, 5, 5, 11, 57, 23046)
    assert img.data.time.shape == (40,)
    assert img.data.image.shape == (40, 256, 256)
    return


def test_themis_no_file():
    """
    Tests that themis() returns a FileNotFound error when we try to download
    non-existant data.
    """
    with pytest.raises(requests.exceptions.HTTPError):
        asilib.asi.themis('pina', time='2011/07/07T02:00')
    return


def test_themis_partial_files():
    """
    Tests that themis() returns a FileNotFound error when we try to download
    non-existant data.
    """
    with pytest.raises(FileNotFoundError):
        asilib.asi.themis(
            'pina', time_range=['2011/07/07T04:15', '2011/07/07T04:23'], missing_ok=False
        )

    img = asilib.asi.themis(
        'pina',
        time_range=['2011/07/07T04:15', '2011/07/07T04:23:30'],
        missing_ok=True,
        redownload=False,
    )
    assert img.file_info['path'][0].name == '20110707_0421_pina_themis18_full.pgm.gz'
    assert img.file_info['path'][1].name == '20110707_0422_pina_themis18_full.pgm.gz'
    assert img.file_info['path'][2].name == '20110707_0423_pina_themis18_full.pgm.gz'
    return


def test_themis_asi_meta():
    """
    Checks that the THEMIS ASI metadata is correct.
    """
    img = asilib.asi.themis(
        'pina',
        time_range=['2011/07/07T04:20', '2011/07/07T04:22:00'],
        missing_ok=True,
        redownload=False,
    )
    assert img.meta == {
        'array': 'THEMIS',
        'location': 'PINA',
        'lat': 50.15999984741211,
        'lon': -96.07000732421875,
        'alt': 0.0,
        'cadence': 3,
        'resolution': (256, 256),
    }
    return


##########################################
############# TEST EXAMPLES ##############
##########################################
@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_themis_fisheye'], tol=10, remove_text=True, extensions=['png']
)
def test_themis_fisheye():
    """
    Plot a fisheye lens image.
    """
    from datetime import datetime
    import matplotlib.pyplot as plt
    import asilib.asi

    asi = asilib.asi.themis('RANK', time=datetime(2017, 9, 15, 2, 34, 0))
    ax, im = asi.plot_fisheye(cardinal_directions='NE', origin=(0.95, 0.05))
    plt.colorbar(im)
    ax.axis('off')
    return

def test_themis_download_bug_fix():
    """
    Test a bug if an imager does not have data for that hour, and that if missing_ok is set,
    the Imager.keogram() raises a ValueError.
    """
    location_code = 'KAPU'
    time_range = (datetime(2013, 3, 17, 11), datetime(2013, 3, 17, 11, 1))

    with pytest.warns(UserWarning) as warninfo:
        asi = asilib.asi.themis(location_code, time_range=time_range)

    with pytest.raises(ValueError) as excinfo:
        asi.plot_keogram()
    
    # check that only one warning was raised
    assert len(warninfo) == 1
    # check that the message matches
    assert 'No local or online image files found' in warninfo[0].message.args[0]
    assert 'The keogram is empty' in str(excinfo.value)
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_themis_map'], tol=10, remove_text=True, extensions=['png']
)
def test_themis_map():
    """
    Plot a fisheye lens image on a map.
    """
    from datetime import datetime
    import matplotlib.pyplot as plt
    import asilib.asi
    import asilib.map

    asi = asilib.asi.themis('RANK', time=datetime(2017, 9, 15, 2, 34, 0))
    ax = asilib.map.create_simple_map()
    ax, im = asi.plot_map(ax=ax)
    plt.colorbar(im)
    ax.axis('off')
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_themis_keogram'], tol=10, remove_text=True, extensions=['png']
)
def test_themis_keogram():
    """
    Plot a keogram.
    """
    from datetime import datetime
    import matplotlib.pyplot as plt
    import asilib.asi

    asi = asilib.asi.themis(
        'RANK', time_range=(datetime(2017, 9, 15, 2, 30, 0), datetime(2017, 9, 15, 2, 35, 0))
    )
    ax, im = asi.plot_keogram()
    plt.colorbar(im)
    ax.axis('off')
    return
