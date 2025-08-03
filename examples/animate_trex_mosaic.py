import asilib
import asilib.asi

time_range = ('2021-11-04T06:55', '2021-11-04T07:05')
asis = asilib.Imagers(
    [asilib.asi.trex_rgb(location_code, time_range=time_range)
    for location_code in ['LUCK', 'PINA', 'GILL', 'RABB']]
    )
asis.animate_map(lon_bounds=(-115, -85), lat_bounds=(43, 63), overwrite=True)