"""
The Imagers class combines multiple Imager objects to coordinate plotting and animating multiple
fisheye lens images, as well as mapped images (also called mosaics).   
"""

from typing import Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from asilib.imager import Imager


class Imagers:
    def __init__(self, imagers:Tuple[Imager]) -> None:
        self.imagers = imagers
        return
    
    def plot_fisheye(self, ax):
        raise NotImplementedError
    
    def plot_map(self, ax=None, overlap=False):
        raise NotImplementedError
    
    def animate_fisheye(self):
        raise NotImplementedError
    
    def animate_fisheye_gen(self):
        raise NotImplementedError
    
    def animate_map(self):
        raise NotImplementedError
    
    def animate_map_gen(self):
        raise NotImplementedError
    
    
if __name__ == '__main__':
    import asilib.map
    import asilib.asi

    time = datetime(2007, 3, 13, 5, 8, 45)
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
    map_alt = 110
    min_elevation = 2

    ax = asilib.map.create_map(lon_bounds=(-140, -60), lat_bounds=(40, 82))

    _imagers = []

    for location_code in location_codes:
        _imagers.append(asilib.asi.themis(location_code, time=time, alt=map_alt))

    asis = Imagers(_imagers)
    asis.plot_map(ax=ax, min_elevation=min_elevation, overlap=True)

    ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
    plt.show()