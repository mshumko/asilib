import unittest
from datetime import datetime

import asilib


class TestPlotFrame(unittest.TestCase):
    def test_plot_movie_generator(self):
        """ Checks that the example in plot_movie_generator() works. """
        time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        movie_generator = asilib.plot_movie_generator(
            time_range, 'THEMIS', 'FSMI', azel_contours=True, overwrite=True
        )

        for frame_time, frame, im, ax in movie_generator:
            # The code that modifies each frame here.
            pass
        return

    def test_plot_movie(self):
        """ Checks that plot_movie() works. """
        time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        asilib.plot_movie(time_range, 'THEMIS', 'FSMI', azel_contours=True, overwrite=True)
        return


if __name__ == '__main__':
    unittest.main()
