# Currently the download scripts can be accessed by 
# asi.download_themis.download_asi_frames,
# but I want to just have asi.download_themis() function.
# from .download import download_themis

from asi.download.download_wrapper import download # Import download scrips.
from asi.download import download_rego
# from . import plot_frame
# from . import plot_movie