# Version 0.7.2
- For consistency, I removed most instances of the word "frame" and changed them to "image". This propagated to the following function renaming (deprecation of the old name).
- Deprecated the get_frame and get_frames functions for load_image. It is a wrapper for _load_image and _load_images functions that were once get_frame and get_frames. I added this function to standardize the load/download names. It returns either one or multiple images, depending on if the time or time_range keyword arguments are given; it will raise an error unless time or time_range is passed (not both).
- Added a plot_image function and deprecated plot_frame.


# Version 0.7.1
- Removed deprecated functions