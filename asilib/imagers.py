"""
The Imagers class handles multiple Imager objects and coordinates plotting and animating multiple
fisheye lens and mapped images. The mapped images are also called mosaics.
"""

from typing import Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from asilib.imager import Imager, _haversine


class Imagers:
    def __init__(self, imagers:Tuple[Imager]) -> None:
        """
        Plot and animate multiple :py:meth:`~asilib.imager.Imager` objects.

        Parameters
        ----------
        imagers: Tuple
            The Imager objects to plot and animate. 
        """
        self.imagers = imagers
        return
    
    def plot_fisheye(self, ax:Tuple[plt.Axes], **kwargs):
        """
        Plots one fisheye image in each subplot, oriented with North on the top, and East on the 
        left of the image.

        Parameters
        ----------
        ax: Tuple[plt.Axes]
            Subplots corresponding to each fisheye lens image.
        kwargs: dict
            Keyword arguments directly passed into each :py:meth:`~asilib.imager.Imager.plot_fisheye()`
            method.

        Example
        -------
        >>> from datetime import datetime
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> import asilib
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> 
        >>> fig, ax = plt.subplots(1, len(location_codes), figsize=(12, 3.5))
        >>> 
        >>> _imagers = []
        >>> 
        >>> for location_code in location_codes:
        >>>     _imagers.append(asilib.asi.themis(location_code, time=time))
        >>> 
        >>> for ax_i in ax:
        >>>     ax_i.axis('off')
        >>> 
        >>> asis = asilib.Imagers(_imagers)
        >>> asis.plot_fisheye(ax=ax)
        >>> 
        >>> plt.suptitle('Donovan et al. 2008 | First breakup of an auroral arc')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        assert len(ax) == len(self.imagers), 'Number of subplots must equal the number of imagers.'

        for ax_i, imager_i in zip(ax, self.imagers):
            imager_i.plot_fisheye(ax=ax_i, **kwargs)
        return
    
    def plot_map(self, overlap=False, **kwargs):
        """
        Projects multiple ASI images onto a map at an altitude that is defined in the skymap 
        calibration file.

        Parameters
        ----------
        overlap: bool
            If True, pixels that overlap between imager FOV's are overplotted such that only the 
            final imager's pixels are shown.
        kwargs: dict
            Keyword arguments directly passed into each :py:meth:`~asilib.imager.Imager.plot_map()`
            method.

        Example
        -------
        >>> from datetime import datetime
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> import asilib
        >>> import asilib.map
        >>> import asilib.asi
        >>> 
        >>> time = datetime(2007, 3, 13, 5, 8, 45)
        >>> location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']
        >>> map_alt = 110
        >>> min_elevation = 2
        >>> 
        >>> ax = asilib.map.create_map(lon_bounds=(-140, -60), lat_bounds=(40, 82))
        >>> 
        >>> _imagers = []
        >>> 
        >>> for location_code in location_codes:
        >>>     _imagers.append(asilib.asi.themis(location_code, time=time, alt=map_alt))
        >>> 
        >>> asis = asilib.Imagers(_imagers)
        >>> asis.plot_map(ax=ax, overlap=False, min_elevation=min_elevation)
        >>> 
        >>> ax.set_title('Donovan et al. 2008 | First breakup of an auroral arc')
        >>> plt.show()
        """
        if not overlap:
            self._calc_overlap_mask()

        for imager in self.imagers:
            imager.plot_map(**kwargs)
        return
    
    # def animate_fisheye(self):
    #     raise NotImplementedError
    
    # def animate_fisheye_gen(self):
    #     raise NotImplementedError
    
    # def animate_map(self):
        
    #     raise NotImplementedError
    
    # def animate_map_gen(self, overlap=False):
    #     if overlap:
    #         self._calc_overlap_mask()
    #     raise NotImplementedError
    
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
        7. If the minimum j is not the ith imager, mask the imager.skymap['lat'] and 
        imager.skymap['lon'] as np.nan.
        """
        if hasattr(self, '_masked'):
            return

        for i, imager in enumerate(self.imagers):
            _distances = np.nan*np.ones((*imager.skymap['lat'].shape, len(self.imagers)))
            for j, other_imager in enumerate(self.imagers):
                # Calculate the distance between all imager pixels and every other imager 
                # location (including itself).
                _distances[:, :, j] = _haversine(
                    imager.skymap['lat'], imager.skymap['lon'],
                    np.broadcast_to(other_imager.meta['lat'], imager.skymap['lat'].shape), 
                    np.broadcast_to(other_imager.meta['lon'], imager.skymap['lat'].shape)
                    )
            # Without this small reduction in the distance of pixels to its own imager,
            # there are gaps between the imager boundaries. In other words, this scaling
            # slightly biases to plotting pixels nearest to the imager. 
            _distances[:, :, i] *= 0.95
            # Need a masked array so that np.nanargmin correctly handles all NaN slices.
            _distances = np.ma.masked_array(_distances, np.isnan(_distances))
            # For each pixel, calculate the nearest imager. If the pixel is not closest to 
            # the imager that it's from, mask it as np.nan. Then the Imager._pcolormesh_nan() 
            # method then won't plot that pixel.
            min_distances = np.argmin(_distances, axis=2)
            far_pixels = np.where(min_distances != i)
            imager.skymap['lat'][far_pixels] = np.nan
            imager.skymap['lon'][far_pixels] = np.nan
        self._masked = True  # A flag to not run again.
        return
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import asilib.map
    import asilib.asi

    time = datetime(2007, 3, 13, 5, 8, 45)
    location_codes = ['FSIM', 'ATHA', 'TPAS', 'SNKQ']

    fig, ax = plt.subplots(1, len(location_codes), figsize=(12, 3.5))

    _imagers = []

    for location_code in location_codes:
        _imagers.append(asilib.asi.themis(location_code, time=time))

    for ax_i in ax:
        ax_i.axis('off')

    asis = Imagers(_imagers)
    asis.plot_fisheye(ax=ax)

    plt.suptitle('Donovan et al. 2008 | First breakup of an auroral arc')
    plt.tight_layout()
    plt.show()