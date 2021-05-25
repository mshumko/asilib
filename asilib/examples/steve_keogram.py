import matplotlib.pyplot as plt

import asilib

mission='REGO'
station='LUCK'

fig, ax = plt.subplots(figsize=(8, 6))
ax, im = asilib.plot_keogram(['2017-09-27T07', '2017-09-27T09'], mission, station, 
                ax=ax, map_alt=230, color_bounds=(300, 800), pcolormesh_kwargs={'cmap':'turbo'})
plt.colorbar(im)
plt.tight_layout()
plt.show()