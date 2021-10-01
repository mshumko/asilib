""" 
Tests for plot_map.py.
"""
import unittest
import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import asilib
from asilib.plot.plot_map import plot_map


class Test_plot_map(unittest.TestCase):
    def test_steve_map(self):
        """
        Tests that plot_map() doesn't crash when making this plot:
        http://themis.igpp.ucla.edu/nuggets/nuggets_2018/Gallardo-Lacourt/fig2.jpg.
        """
        plot_map(datetime(2010, 4, 5, 6, 7, 0), 'THEMIS', 'ATHA', 110)
        return

    def test_plot_map_altitude_error(self):
        """
        Checks that plto_map() raises an AssertionError if map_alt is not
        in the list of skymap calibration altitudes.
        """
        with self.assertRaises(AssertionError):
            plot_map(datetime(2010, 4, 5, 6, 7, 0), 'THEMIS', 'ATHA', 500)
        return

    def test_donovan_et_al_2008_plot(self):
        """
        Checks that Figure 2b from Donovan et al. 2008 successfully plots.

        https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2008GL033794
        """
        time = datetime(2007, 3, 13, 5, 8, 45)
        mission = 'THEMIS'
        stations = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        map_alt = 110
        min_elevation = 2

        fig = plt.figure(figsize=(8, 5))
        plot_extent = [-160, -52, 40, 82]
        central_lon = np.mean(plot_extent[:2])
        central_lat = np.mean(plot_extent[2:])
        projection = ccrs.Orthographic(central_lon, central_lat)
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(linestyle=':')

        # image_time, image, skymap, ax = plot_map(time, mission, stations[0], map_alt,
        #     min_elevation=min_elevation)
        for station in stations:
            plot_map(time, mission, station, map_alt, ax=ax, min_elevation=min_elevation)

        ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')


if __name__ == '__main__':
    unittest.main()
