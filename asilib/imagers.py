"""
The Imagers class combines multiple Imager objects to coordinate plotting and animating multiple
fisheye lens images, as well as mapped images (also called mosaics).   
"""

from typing import Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from asilib.imager import Imager, _haversine


class Imagers:
    def __init__(self, imagers:Tuple[Imager]) -> None:
        self.imagers = imagers
        return
    
    def plot_fisheye(self, ax):
        raise NotImplementedError
    
    def plot_map(self, ax=None, overlap=False):
        if overlap:
            self._calc_overlap_mask()
        raise NotImplementedError
    
    def animate_fisheye(self):
        raise NotImplementedError
    
    def animate_fisheye_gen(self):
        raise NotImplementedError
    
    def animate_map(self):
        
        raise NotImplementedError
    
    def animate_map_gen(self, overlap=False):
        if overlap:
            self._calc_overlap_mask()
        raise NotImplementedError
    
    def _calc_overlap_mask(self):
        """
        Calculate which pixels to plot for overlapping imagers by the criteria that the ith 
        imager's pixel must be closest to that imager (and not a neighboring one).

        Algorithm:
        1. Loop over ith imager
        2. Loop over jth imager within 500 km distance of the ith imager
        3. Mask low-elevations with np.nan
        4. Create a distance array with shape (resolution[0], resolution[1], j_total)
        5. For all pixels in ith imager, calculate the haversine distance to jth imager and 
        assign it to distance[..., j].
        6. For all pixels calculate the nearest imager out of all j's.
        7. If the minimum j is not the ith imager, mask as np.nan.
        """
        if hasattr(self, '_overlap_masks'):
            return self._overlap_masks
        self._overlap_masks = {
                asi.meta['location']:np.ones(
                    (asi.meta['resolution'][0], asi.meta['resolution'][1])
                ) for asi in self.imagers
                }
        for asi in self.imagers:
            pass
        return
    
    
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