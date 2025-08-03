import pathlib
import configparser

__version__ = '0.26.5'

# Load the configuration settings.
HERE = pathlib.Path(__file__).parent.resolve()
settings = configparser.ConfigParser()
settings.read(HERE / 'config.ini')

try:
    ASI_DATA_DIR = settings['Paths'].get('ASI_DATA_DIR', pathlib.Path.home() / 'asilib-data')
    ASI_DATA_DIR = pathlib.Path(ASI_DATA_DIR)
except KeyError:
    ASI_DATA_DIR = pathlib.Path.home() / 'asilib-data'

config = {'ASILIB_DIR': HERE, 'ASI_DATA_DIR': ASI_DATA_DIR, 'ACKNOWLEDGED_ASIS':[]}

# Imager implementation functions and classes.
from asilib.imager import Imager
from asilib.imagers import Imagers
from asilib.conjunction import Conjunction

__all__ = [
    'Imager',
    'Imagers',
    'Conjunction',
]  # So Sphinx shortens the name. See https://stackoverflow.com/a/31594545
