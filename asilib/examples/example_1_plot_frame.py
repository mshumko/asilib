from datetime import datetime

import matplotlib.pyplot as plt

import asilib

# A bright auroral arc that was analyzed by Imajo et al., 2021 "Active
# auroral arc powered by accelerated electrons from very high altitudes"
image_time, image, ax, im = asilib.plot_image(
    datetime(2017, 9, 15, 2, 34, 0), 'THEMIS', 'RANK', color_norm='log', force_download=False
)
plt.colorbar(im)
ax.axis('off')
plt.show()
