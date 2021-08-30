"""
This example generates a keogram of a STEVE aurora that moved equatorward. This event was studied
in: 

Gallardo-Lacourt, B., Nishimura, Y., Donovan, E., Gillies, D. M., Perry, G. W., Archer, W. E., et al. (2018). A statistical analysis of STEVE. Journal of Geophysical Research: Space 
Physics, 123, 9893– 9905. https://doi.org/10.1029/2018JA025368
"""

import matplotlib.pyplot as plt

import asilib

mission = 'REGO'
station = 'LUCK'
map_alt_km = 230

fig, ax = plt.subplots(figsize=(8, 6))
ax, im = asilib.plot_keogram(
    ['2017-09-27T07', '2017-09-27T09'],
    mission,
    station,
    ax=ax,
    map_alt=map_alt_km,
    color_bounds=(300, 800),
)
plt.colorbar(im, label='Intensity')
ax.set_xlabel('UTC')
ax.set_ylabel(f'Emission Latitude [deg] at {map_alt_km} km')
plt.tight_layout()
plt.show()
