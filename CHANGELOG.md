# Version 0.7.2
- Deprecated the get_frame and get_frames functions for load_image; load_image returns either one or multiple images, depending on if the time or time_range keyword arguments are given.
- Added the load_image function. It is a wrapper for _get_image and _get_images functions that were once get_frame and get_frames. I added this function to standardize the load/download names.
- For consistency, I removed most instances of the word "frame" and changed them to "image". 

# Version 0.7.1
- Removed deprecated functions