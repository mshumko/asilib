"""
This is an example aurora due to a field line resonance studied in:

Gillies, D. M., Knudsen, D., Rankin, R., Milan, S., & Donovan, E. (2018). A statistical survey
of the 630.0-nm optical signature of periodic auroral arcs resulting from magnetospheric field
line resonances. Geophysical Research Letters, 45(10), 4648-4655.
"""
import matplotlib.pyplot as plt

import asilib.asi

location_code = 'GILL'
time_range = ['2015-02-02T10', '2015-02-02T11']

asi = asilib.asi.rego(location_code, time_range=time_range, alt=230)
ax, p = asi.plot_keogram(color_map='Greys_r')
plt.xlabel('Time')
plt.ylabel('Geographic Latitude [deg]')
plt.colorbar(p)
plt.tight_layout()
plt.show()
