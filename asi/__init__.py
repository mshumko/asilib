# Currently the download scripts can be accessed by 
# asi.download_themis.download_asi_frames,
# but I want to just have asi.download_themis() function.
# from .download import download_themis
import warnings
import pathlib

# Check that the package has been configured and config.py file exists.
here = pathlib.Path(__file__).parent.resolve()
if not pathlib.Path(here / 'config.py').is_file():
    raise ImportError('config.py file with the ASI data directory not found. '
                'Did you run "python3 -m aurora_asi init"?')

from asi.download.download_wrapper import download # Import download scrips.
from asi.download import download_rego
# from . import plot_frame
# from . import plot_movie