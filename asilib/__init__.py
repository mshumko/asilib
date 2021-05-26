import warnings
import pathlib
import importlib.util
import configparser

__version__ = '0.3.1'

# Load the configuration settings.
HERE = pathlib.Path(__file__).parent.resolve()
settings = configparser.ConfigParser()
settings.read(HERE / 'config.ini')

try:
    ASI_DATA_DIR = settings['Paths'].get('ASI_DATA_DIR', 
        pathlib.Path.home() / 'asilib-data')
except KeyError:  # Raised if config.ini does not have Paths.
    ASI_DATA_DIR = pathlib.Path.home() / 'asilib-data'

try:
    IRBEM_WARNING = settings['Warnings'].getboolean('IRBEM')
except KeyError: # Raised if config.ini does not have Warnings.
    IRBEM_WARNING = True

config = {
    'ASILIB_DIR': HERE,
    'ASI_DATA_DIR': ASI_DATA_DIR, 
    'IRBEM_WARNING':IRBEM_WARNING
}

# Import download programs.
from asilib.io.download_rego import download_rego_img, download_rego_cal
from asilib.io.download_themis import download_themis_img, download_themis_cal

# Import the loading functions.
from asilib.io.load import load_img
from asilib.io.load import load_cal
from asilib.io.load import get_frame
from asilib.io.load import get_frames

# Import the plotting and animating functions.
from asilib.plot.plot_frame import plot_frame
from asilib.plot.plot_movie import plot_movie, plot_movie_generator
from asilib.plot.plot_keogram import plot_keogram

# Import the analysis functions.
from asilib.analysis.project_lla_to_skyfield import lla_to_skyfield
from asilib.analysis.keogram import keogram
from asilib.analysis.equal_area import equal_area

# Import the equal_area function.
from asilib.analysis.equal_area import equal_area

# If the IRBEM module exists, import map_along_magnetic_field.
# This is better than a try-except block because the ImportError
# exception can be raised from another map_along_magnetic_field 
# dependency but we want to specifically check for IRBEM and let
# it crash if something else is wrong.
if importlib.util.find_spec('IRBEM'):
    from asilib.analysis.map_along_magnetic_field import map_along_magnetic_field
else:
    if config['IRBEM_WARNING']:
        warnings.warn(
            "The IRBEM magnetic field library is not installed and is "
            "a dependency of asilib.map_along_magnetic_field()."
        )
