import unittest
from datetime import datetime

import asilib


class TestPlotImage(unittest.TestCase):
    def test_plot_movie_generator(self):
        """Checks that the example in plot_movie_generator() works."""
        time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        movie_generator = asilib.plot_movie_generator(
            time_range, 'THEMIS', 'FSMI', azel_contours=True, overwrite=True
        )

        for image_time, image, im, ax in movie_generator:
            # The code that modifies each image here.
            pass
        return

    def test_plot_movie(self):
        """Checks that plot_movie() works."""
        time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        asilib.plot_movie(time_range, 'THEMIS', 'FSMI', azel_contours=True, overwrite=True)
        return

    def test_plot_movie_generator(self):
        """Check that the generator function and the .send() method works"""
        time_range = (datetime(2015, 3, 26, 6, 7), datetime(2015, 3, 26, 6, 12))
        gen = asilib.plot_movie_generator(
            time_range, 'THEMIS', 'FSMI', azel_contours=True, overwrite=True
        )
        tup = gen.send("get_image_data")
        self.assertEqual(tup.time.shape, (100,))
        self.assertEqual(tup.images.shape, (100, 256, 256))


if __name__ == '__main__':
    unittest.main()
