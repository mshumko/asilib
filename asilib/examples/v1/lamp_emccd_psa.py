from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import asilib.map
from asilib.asi.psa_emccd import psa_emccd


n_avg = 10  # Average 10 image frames (the effective imager cadence is 10 Hz)

asi = psa_emccd(
    'vee',
    time_range=(datetime(2022, 3, 5, 11, 0), datetime(2022, 3, 5, 11, 2)),
    n_avg=n_avg
)

times, images = asi.data

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
asi.animate_fisheye(ax=ax, ffmpeg_params={'framerate':100})