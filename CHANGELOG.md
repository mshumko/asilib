# Version 0.9.4
- Renamed and deprecated `plot_movie` and `plot_movie_generator` for `animate_fisheye` and `animate_fisheye_generator` functions. This change is necessary for consistency with new functions such as `animate_map` and `animate_map_generator` functions.

# Version 0.9.3
- Renamed the `_make_map` function to `create_cartopy_map()` so users can use this function to create maps from now on.
- Updated the information in `CONTRIBUTE.md`.

# Version 0.9.2
- Fixed a bug where the `color_bounds` in `plot_movie_generator()` were static after the first image.
- Renamed and deprecated `plot_image` for `plot_fisheye`.

# Version 0.9.0
- Rotated the 2- and 3-D `skymap` arrays and fixed a bug in `keogram.py` where the skymap latitudes were a few pixels off.

# Version 0.8.0 
- Swapped the order of most functions to `asi_array_code, location_code, time`. This is a major API change that is not backwards-compatible. 
- Many edits the the docstrings.
- Renamed the example scripts and added example scripts for `plot_map`.
- Finished a first complete draft of tutorial.ipynb.

# Version 0.7.5
- Changed all instances of the word `station` and replaced it with `location_code`.

# Version 0.7.4
- The main change is the `ignore_missing_files` parameter that is passed to the download functions. When True, the `download_themis_img` and `download_rego_img` will not raise a missing file error when data from that hour does not exist.

# Version 0.7.3
- The biggest API change is the parameter order for `download_rego_img` and `download_themis_img`. Now it is `(location code (i.e. station), time, and time_range)`. Beware that now the parameter order API is inconsistent across all of the functions---I will standardize it to (`asi_array_code`, `location code (i.e. station)`, `time`, and `time_range`) in the next minor (0.X.0 release).

# Version 0.7.2
- For consistency, I removed most instances of the word "frame" and changed them to "image". This propagated to the following function renaming (deprecation of the old name).
- Deprecated the get_frame and get_frames functions for load_image. It is a wrapper for _load_image and _load_images functions that were once get_frame and get_frames. I added this function to standardize the load/download names. It returns either one or multiple images, depending on if the time or time_range keyword arguments are given; it will raise an error unless time or time_range is passed (not both).
- Renamed the plot_frame function to `plot_image`; plot_frame is now deprecated.

# Version 0.7.1
- Removed deprecated functions