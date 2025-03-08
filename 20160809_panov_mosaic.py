import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

time_range = ('2016-08-09T08:30', '2016-08-09T09:30')
location_codes = ('FSMI', 'TPAS', 'GILL', 'PINA')
alt=110

fig = plt.figure(figsize=(5, 5.5))
ax = asilib.map.create_map(fig_ax=(fig, 111), lon_bounds=(-115, -85), lat_bounds=(43, 65))
plt.subplots_adjust(top=0.88)
plt.tight_layout()

# asis = asilib.Imagers(
#     [asilib.asi.themis(location_code=location_code, time_range=time_range, alt=alt) 
#     for location_code in location_codes]
#     )
color_bounds = [
    (4_000, 10_000),
    (2_500, 10_000),
    (4_000, 9_000),
    (3_000, 9_000),
]
ax.set_title(f'THEMIS ASI | {time_range[0][:10]} | {alt} km map altitude')
# gen = asis.animate_map_gen(overwrite=True, ax=ax, min_elevation=10, color_bounds=color_bounds)
# for _guide_time, _asi_times, _asi_images, _ in gen:
#     if '_plot_time' in locals():
#         _plot_time.remove()
#     _plot_time = ax.text(
#         0.05, 0.95, f'{_guide_time:%H:%M:%S}', va='top', transform=ax.transAxes, fontsize=15
#         )
asis2 = asilib.Imagers(
    [asilib.asi.themis(location_code=location_code, time=time_range[0], alt=alt) 
    for location_code in location_codes]
    )
asis2.plot_map(ax=ax, color_bounds=color_bounds)
plt.show()