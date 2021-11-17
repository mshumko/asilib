import numpy as np
import matplotlib.colors as colors


def get_color_bounds(image):
    lower, upper = np.nanquantile(image, (0.25, 0.98))
    color_bounds = [lower, np.min([upper, lower * 10])]
    return color_bounds

def get_color_map(asi_array_code, color_map):
    if (color_map == 'auto') and (asi_array_code.lower() == 'themis'):
        color_map = 'Greys_r'
    elif (color_map == 'auto') and (asi_array_code.lower() == 'rego'):
        color_map = colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    else:
        raise NotImplementedError('color_map == "auto" but the asi_array_code is unsupported')
    return color_map

def get_color_norm(color_norm, color_bounds):
    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')
    return norm