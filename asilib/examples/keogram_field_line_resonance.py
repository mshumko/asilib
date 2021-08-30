"""
This is an example aurora due to a field line resonance studied in: 

Gillies, D. M., Knudsen, D., Rankin, R., Milan, S., & Donovan, E. (2018). A statistical survey
of the 630.0‐nm optical signature of periodic auroral arcs resulting from magnetospheric field 
line resonances. Geophysical Research Letters, 45(10), 4648-4655.
"""
import matplotlib.pyplot as plt

import asilib

mission = 'REGO'
station = 'GILL'

fig, ax = plt.subplots(figsize=(8, 6))
ax, im = asilib.plot_keogram(
    ['2015-02-02T10', '2015-02-02T11'],
    mission,
    station,
    ax=ax,
    map_alt=230,
    pcolormesh_kwargs={'cmap': 'Greys_r'},
)
plt.colorbar(im)
plt.tight_layout()
plt.show()
