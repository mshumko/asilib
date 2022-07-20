import matplotlib.projections
from matplotlib.projections.geo import GeoAxes, _GeoTransform
import numpy as np

class MollweideAxes(GeoAxes):
    name = 'mollweide'

    class MollweideTransform(_GeoTransform):
        """The base Mollweide transform."""

        def transform_non_affine(self, ll):
            # docstring inherited
            def d(theta):
                delta = (-(theta + np.sin(theta) - pi_sin_l)
                         / (1 + np.cos(theta)))
                return delta, np.abs(delta) > 0.001

            longitude, latitude = ll.T

            clat = np.pi/2 - np.abs(latitude)
            ihigh = clat < 0.087  # within 5 degrees of the poles
            ilow = ~ihigh
            aux = np.empty(latitude.shape, dtype=float)

            if ilow.any():  # Newton-Raphson iteration
                pi_sin_l = np.pi * np.sin(latitude[ilow])
                theta = 2.0 * latitude[ilow]
                delta, large_delta = d(theta)
                while np.any(large_delta):
                    theta[large_delta] += delta[large_delta]
                    delta, large_delta = d(theta)
                aux[ilow] = theta / 2

            if ihigh.any():  # Taylor series-based approx. solution
                e = clat[ihigh]
                d = 0.5 * (3 * np.pi * e**2) ** (1.0/3)
                aux[ihigh] = (np.pi/2 - d) * np.sign(latitude[ihigh])

            xy = np.empty(ll.shape, dtype=float)
            xy[:, 0] = (2.0 * np.sqrt(2.0) / np.pi) * longitude * np.cos(aux)
            xy[:, 1] = np.sqrt(2.0) * np.sin(aux)

            return xy

        def inverted(self):
            # docstring inherited
            return MollweideAxes.InvertedMollweideTransform(self._resolution)

    class InvertedMollweideTransform(_GeoTransform):

        def transform_non_affine(self, xy):
            # docstring inherited
            x, y = xy.T
            # from Equations (7, 8) of
            # https://mathworld.wolfram.com/MollweideProjection.html
            theta = np.arcsin(y / np.sqrt(2))
            longitude = (np.pi / (2 * np.sqrt(2))) * x / np.cos(theta)
            latitude = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            # docstring inherited
            return MollweideAxes.MollweideTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.MollweideTransform(resolution)

matplotlib.projections.register_projection(MollweideAxes)

if __name__ == '__main__':
    from datetime import datetime

    import matplotlib.pyplot as plt

    import asilib

    fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'})
    ax.plot([-1, 1, 1], [-1, -1, 1], "o-")
    ax.grid()

    plt.show()

#     asi_array_code = 'THEMIS'
#     location_code = 'ATHA'
#     time = datetime(2008, 3, 9, 9, 18, 0) 
#     map_alt_km = 110
#     asilib.plot_map(asi_array_code, location_code, time, map_alt_km)
#     plt.tight_layout()
#     plt.show()