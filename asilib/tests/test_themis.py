"""
Tests the themis() data loading function.
"""
from datetime import datetime

import pytest

import asilib.array.themis as themis


def test_themis_time():
    """
    Tests that one file is loaded and themis() returns the correct file.
    """
    img = themis.themis('gill',  time='2014-05-05T04:49:10')
    assert img._data['time'] == datetime(2014, 5, 5, 4, 49, 9, 37070)
    return


def test_themis_time_range():
    """
    Tests that multiple files are loaded using themis()'s time_range kwarg.
    """
    img = themis.themis(
        'gill',  
        time_range=['2014-05-05T05:10', '2014-05-05T05:12'])
    assert img._data['path'][0].name == '20140505_0510_gill_themis19_full.pgm.gz'
    assert img._data['path'][1].name == '20140505_0511_gill_themis19_full.pgm.gz'
    return


def test_themis_no_file():
    """
    Tests that themis() returns a FileNotFound error when we try to download 
    non-existant data.
    """
    with pytest.raises(FileNotFoundError):
        themis.themis('pina',  time='2011/07/07T02:00')
    return


def test_themis_partial_files():
    """
    Tests that themis() returns a FileNotFound error when we try to download 
    non-existant data.
    """
    with pytest.raises(FileNotFoundError):
        themis.themis('pina', time_range=['2011/07/07T04:15', '2011/07/07T04:23'], 
            missing_ok=False)
    
    img = themis.themis('pina', time_range=['2011/07/07T04:15', '2011/07/07T04:23:30'], 
            missing_ok=True, overwrite=True)
    assert img._data['path'][0].name == '20110707_0421_pina_themis18_full.pgm.gz'
    assert img._data['path'][1].name == '20110707_0422_pina_themis18_full.pgm.gz'
    assert img._data['path'][2].name == '20110707_0423_pina_themis18_full.pgm.gz'
    return

def test_themis_asi_meta():
    """
    Checks that the THEMIS ASI metadata is correct.
    """
    img = themis.themis('pina', time_range=['2011/07/07T04:20', '2011/07/07T04:22:00'], 
            missing_ok=True, overwrite=True)
    assert img.meta == {'array': 'THEMIS', 'location': 'PINA', 'lat': 50.15999984741211, 
        'lon': -96.07000732421875, 'alt': 0.0, 'cadence': 3, 'resolution': (256, 256)}
    return