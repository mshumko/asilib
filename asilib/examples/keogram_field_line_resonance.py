"""
This is an example aurora due to a field line resonance studied in: 

Gillies, D. M., Knudsen, D., Rankin, R., Milan, S., & Donovan, E. (2018). A statistical survey
of the 630.0‐nm optical signature of periodic auroral arcs resulting from magnetospheric field 
line resonances. Geophysical Research Letters, 45(10), 4648-4655.
"""
import matplotlib.pyplot as plt

import asilib

asi_array_code = 'REGO'
location_code = 'GILL'
time_range = ['2015-02-02T10', '2015-02-02T11']

fig, ax = plt.subplots(figsize=(8, 6))
ax, im = asilib.plot_keogram(
    asi_array_code,
    location_code,
    time_range,
    ax=ax,
    map_alt=230,
    pcolormesh_kwargs={'cmap': 'Greys_r'},
)
plt.xlabel('Time')
plt.ylabel('Geographic Latitude [deg]')
plt.colorbar(im)
plt.tight_layout()
plt.show()
