"""
Tests the asilib's Conjunction class.
"""
from datetime import datetime, timedelta
import dateutil.parser

import numpy as np
import pytest

import asilib

t0 = dateutil.parser.parse('2014-05-05T04:49:10')


def test_find():
    n_times = 200
    img = asilib.themis('gill', time='2014-05-05T04:49:10', load_images=False, alt=110)
    lats = np.linspace(90, 0, num=n_times)
    lons = img.meta['lon'] * np.ones_like(lats)
    alts = 110 * np.ones_like(lats)
    lla = np.array([lats, lons, alts]).T
    times = np.array([t0 + timedelta(seconds=i * 0.5) for i in range(n_times)])
    c = asilib.Conjunction(img, times, lla)
    c.find()
    return


# def test_load_images():
#     return
