from datetime import datetime

import numpy as np
import matplotlib.testing.decorators

from asilib.asi.fake_asi import fake_asi


##########################################
########## TEST DATA ACCESS ##############
##########################################
def test_time():
    """
    Test if the fake_asi timestamp is correctly accessed.
    """
    asi = fake_asi('GILL', time='2015-01-01T15:14:00.17')
    assert asi.data.times == datetime(2015, 1, 1, 15, 14)
    assert asi.data.images.shape == (512, 512)
    assert np.isclose(asi.data.images.mean(), 14.4014892578125)
    # See https://numpy.org/doc/stable/reference/generated/numpy.argmax.html for the
    # unravel_index example to get the maximum index for a N-d array.
    ind = np.unravel_index(np.argmax(asi.data.images, axis=None), asi.data.images.shape)
    assert ind == (318, 0)
    return

def test_time_range():
    """
    Test if fake_asi timestamps are correctly accessed.
    """
    asi = fake_asi('GILL', time_range=('2015-01-01T15:00:15.17', '2015-01-01T20:00'))
    assert asi.data.times.shape == (1798,)
    assert asi.data.images.shape == (1798, 512, 512)
    assert np.isclose(asi.data.images.mean(), 14.048684923216552)
    assert asi._data['path'] == [
        '20150101_150000_GILL_fake_asi.images', 
        '20150101_160000_GILL_fake_asi.images',
        '20150101_170000_GILL_fake_asi.images',
        '20150101_180000_GILL_fake_asi.images',
        '20150101_190000_GILL_fake_asi.images'
        ]
    return

def test_iter_files():
    """
    Test Imager.iter_files()
    """
    asi = fake_asi('GILL', time_range=('2015-01-01T15:00:15.17', '2015-01-01T19:30'))
    n_timestamps_reference = [358, 360, 360, 360, 181]

    counter = 0
    for (_times, _images), ref_n in zip(asi.iter_files(), n_timestamps_reference):
        assert _times.shape[0] == ref_n
        assert _images.shape[0] == ref_n
        counter += 1

    assert counter == len(n_timestamps_reference)
    return

##########################################
############# TEST PLOTTING ##############
##########################################
@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_fisheye'], tol=10, remove_text=True, extensions=['png']
)
def test_fisheye():
    """
    Tests that the fake_asi produces the correct image.
    """
    asi = fake_asi('GILL', time='2015-01-01T15:14:00.17')
    asi.plot_fisheye(color_bounds=(1, 255), origin=(0.85, 0.15), cardinal_directions='NEWS')
    return

@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_plot_keogram'], tol=10, remove_text=True, extensions=['png']
)
def test_plot_keogram():
    """
    Tests that the fake_asi produces the correct keogram.
    """
    asi = fake_asi('GILL', time_range=('2015-01-01T15:00:15.17', '2015-01-01T19:30'))
    asi.plot_keogram()
    return