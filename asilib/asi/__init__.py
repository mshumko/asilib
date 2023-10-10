from .rego import rego, rego_info, rego_skymap
from .themis import themis, themis_info, themis_skymap
from .trex import trex_nir, trex_nir_skymap, trex_nir_info
from .trex import trex_rgb, trex_rgb_info, trex_rgb_skymap

__all__ = [
    'themis', 
    'themis_info', 
    'themis_skymap',
    'rego',
    'rego_info', 
    'rego_skymap',
    'trex_rgb', 
    'trex_rgb_info', 
    'trex_rgb_skymap',
    'trex_nir', 
    'trex_nir_skymap', 
    'trex_nir_info',
]  # So Sphinx shortens the name. See https://stackoverflow.com/a/31594545
