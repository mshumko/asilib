import pathlib
import configparser

__version__ = '0.18.0'

# Load the configuration settings.
HERE = pathlib.Path(__file__).parent.resolve()
settings = configparser.ConfigParser()
settings.read(HERE / 'config.ini')

try:
    ASI_DATA_DIR = settings['Paths'].get('ASI_DATA_DIR', pathlib.Path.home() / 'asilib-data')
    ASI_DATA_DIR = pathlib.Path(ASI_DATA_DIR)
except KeyError:  # Raised if config.ini does not have Paths.
    ASI_DATA_DIR = pathlib.Path.home() / 'asilib-data'

config = {'ASILIB_DIR': HERE, 'ASI_DATA_DIR': ASI_DATA_DIR}

# Import download programs.
from asilib.io.download import download_image
from asilib.io.download import download_skymap

# Import the loading functions.
from asilib.io.load import load_skymap
from asilib.io.load import load_image
from asilib.io.load import load_image_generator

# Import the plotting and animating functions.
from asilib.plot.plot_fisheye import plot_fisheye
from asilib.plot.plot_map import plot_map
from asilib.plot.plot_map import make_map
from asilib.plot.plot_keogram import plot_keogram
from asilib.plot.animate_fisheye import animate_fisheye
from asilib.plot.animate_fisheye import animate_fisheye_generator
from asilib.plot.animate_map import animate_map
from asilib.plot.animate_map import animate_map_generator

# Import the analysis functions.
from asilib.analysis.map import lla2azel
from asilib.analysis.map import lla2footprint
from asilib.analysis.keogram import keogram
from asilib.analysis.equal_area import equal_area

# Imager implementation functions and classes.
from asilib.imager import Imager
from asilib.imagers import Imagers
from asilib.conjunction import Conjunction
from asilib.asi.themis import themis, themis_info, themis_skymap
from asilib.asi.rego import rego, rego_info, rego_skymap

__all__ = [
    'Imager',
    'Conjunction',
]  # So Sphinx shortens the name. See https://stackoverflow.com/a/66743762
