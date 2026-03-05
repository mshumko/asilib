from .rego import rego, rego_info, rego_skymap
from .themis import themis, themis_info, themis_skymap, themis_available, plot_themis_available
from .trex import trex_nir, trex_nir_skymap, trex_nir_info
from .trex import trex_rgb, trex_rgb_info, trex_rgb_skymap, trex_rgb_available, plot_trex_rgb_available
from .psa_project import psa_project, psa_project_info, psa_project_skymap, psa_project_lamp
from .mango import mango, mango_info

__all__ = [
    'themis', 
    'themis_info', 
    'themis_skymap',
    'themis_available',
    'plot_themis_available',
    'rego',
    'rego_info', 
    'rego_skymap',
    'trex_rgb', 
    'trex_rgb_info', 
    'trex_rgb_skymap',
    'trex_nir', 
    'trex_nir_skymap', 
    'trex_nir_info',
    'trex_rgb_available', 
    'plot_trex_rgb_available',
    'mango',
    'mango_info',
    'psa_project',
    'psa_project_info',
    'psa_project_lamp'
]  # So Sphinx shortens the name. See https://stackoverflow.com/a/31594545
