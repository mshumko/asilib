import numpy as np
import matplotlib.colors as colors

# TODO: Make all of the other plot functions call these ones.


def get_color_bounds(image):
    """
    A decent default for the minimum and maximum colorbar values for aurora images. This way
    bright objects like the moon don't saturate the image while preserving enough dynamic range.
    """
    lower, upper = np.nanquantile(image, (0.25, 0.98))
    color_bounds = [lower, np.min([upper, lower * 10])]
    return color_bounds


def get_color_map(asi_array_code, color_map):
    """
    Color maps for the THEMIS and REGO ASIs.
    """
    if (color_map == 'auto') and (asi_array_code.lower() == 'themis'):
        color_map = 'Greys_r'
    elif (color_map == 'auto') and (asi_array_code.lower() == 'rego'):
        color_map = colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r'])
    else:
        raise NotImplementedError('color_map == "auto" but the asi_array_code is unsupported')
    return color_map


def get_color_norm(color_norm, color_bounds):
    """
    Sets the normalization of the images to linear or logarithmic. You will typically call
    get_color_bounds() to get the color_bounds for this function.
    """
    if color_norm == 'log':
        norm = colors.LogNorm(vmin=color_bounds[0], vmax=color_bounds[1])
    elif color_norm == 'lin':
        norm = colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    else:
        raise ValueError('color_norm must be either "log" or "lin".')
    return norm
