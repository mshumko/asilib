"""
This example generates an image of an omega band aurora. This event was studied
in: 

Liu, J., Lyons, L. R., Archer, W. E., Gallardo-Lacourt, B., Nishimura, Y., Zou, 
Y., … Weygand, J. M. (2018). Flow shears at the poleward boundary of omega bands 
observed during conjunctions of Swarm and THEMIS ASI. Geophysical Research Letters, 
45, 1218– 1227. https://doi.org/10.1002/2017GL076485
"""

import matplotlib.pyplot as plt

import asilib

mission = 'THEMIS'
station = 'KAPU'

asilib.plot_image('2008-03-09T04:39:00', mission, station)
plt.tight_layout()
plt.show()
