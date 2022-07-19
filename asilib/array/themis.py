"""
This module contains the THEMIS ASI data loader for the asilib.Imager. 

Public Functions
----------------
themis
    Loads the THEMIS ASI data and returns an asilib.Imager object.
themis_info
    Loads a table of THEMIS ASI locations, and location codes.
"""

from datetime import datetime, timedelta
from multiprocessing import Pool
import re
import warnings
import pathlib
import copy
import gzip
import bz2
import signal
import dateutil.parser
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io

import asilib
import asilib.utils as utils
import asilib.io.download as download
import asilib.io.manager as manager


pgm_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/stream0/'
skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/THEMIS/asi/skymaps/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'themis'

# pgm loader variables
THEMIS_IMAGE_SIZE_BYTES = 131072
THEMIS_DT = np.dtype("uint16")
THEMIS_DT = THEMIS_DT.newbyteorder('>')  # force big endian byte ordering


def themis(location_code: str, time: utils._time_type=None, 
        time_range: utils._time_range_type=None, alt: int=110, 
        overwrite: bool=False, missing_ok: bool=True, load_images: bool=True,
        imager=asilib.Imager)->asilib.Imager:
    """
    Create an Imager instance using the THEMIS ASI images and skymaps.

    Parameters
    ----------
    location_code: str
        The ASI's location code (four letters).
    time: str or datetime.datetime
        A time to look for the ASI data at. Either time or time_range
        must be specified (not both or neither).
    time_range: list of str or datetime.datetime
        A length 2 list of string-formatted times or datetimes to bracket
        the ASI data time interval.
    alt: int
        The reference skymap altitude, in kilometers.
    overwrite: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    load_images: bool
        Create an Imager object without images. This is useful if you need to
        calculate conjunctions and don't need to download or load unnecessary data.
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.
    """

    if time is not None:
         time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    if load_images:
        # Download and find image data
        file_paths = _get_pgm_files(location_code, time, time_range, overwrite, missing_ok)

        if time is not None:
            # Find and load the nearest time stamp
            _times, _images = _load_pgm(file_paths[0])
            image_index = np.argmin(np.abs(
                    [(time - t_i).total_seconds() for t_i in _times]
                    ))
            if np.abs((time-_times[image_index]).total_seconds()) > 3:
                raise IndexError(f'Cannot find a time stamp within 3 seconds of '
                                    f'{time}. Closest time stamp is {_times[image_index]}.')
            data = {'time':_times[image_index], 'image':_images[image_index]}

        elif time_range is not None:
            start_times = len(file_paths)*[None]
            end_times = len(file_paths)*[None]
            for i, file_path in enumerate(file_paths):
                date_match = re.search(r'\d{8}_\d{4}', file_path.name)
                start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d_%H%M')
                end_times[i] = start_times[i] + timedelta(minutes=1) 
            data = {
                'path':file_paths, 'start_time':start_times, 'time_range':time_range,
                'end_time':end_times, 'loader':_load_pgm
                }
    else:
        data = {'time':time, 'image':None}

    # Download and find the appropriate skymap
    if time is not None:
        _time = time 
    else:
        _time = time_range[0]
    _skymap = load_skymap(location_code, _time, overwrite)
    alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0]
    assert len(alt_index) == 1, (
        f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km.'
    )
    alt_index = alt_index[0]
    skymap = {
        'lat':_skymap['FULL_MAP_LATITUDE'][alt_index, :, :],
        'lon':_skymap['FULL_MAP_LONGITUDE'][alt_index, :, :],
        'alt':_skymap['FULL_MAP_ALTITUDE'][alt_index],
        'el':_skymap['FULL_ELEVATION'],
        'az':_skymap['FULL_AZIMUTH'],
        'path':_skymap['PATH']
        }
    
    meta = {
        'array':'THEMIS', 'location':location_code.upper(),
        'lat':float(_skymap['SITE_MAP_LATITUDE']),
        'lon':float(_skymap['SITE_MAP_LONGITUDE']),
        'alt':float(_skymap['SITE_MAP_ALTITUDE']),
        'cadence':3,
        'resolution':(256, 256)
        }
    return imager(data, meta, skymap)

def themis_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the THEMIS ASI names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'THEMIS'] 
    return df.reset_index(drop=True)

def _get_pgm_files(location_code, time, time_range, overwrite, missing_ok):
    """
    """
    if (time is None) and (time_range is None):
        raise ValueError('time or time_range must be specified.')
    elif (time is not None) and (time_range is not None):
        raise ValueError('both time and time_range can not be simultaneously specified.')

    local_dir = local_base_dir / 'images' / location_code.lower()
    _manager = manager.File_Manager(local_dir, base_url=pgm_base_url)

    # Find one image file.
    if time is not None:
        time = utils.validate_time(time)
        url_subdirectories = [
            str(time.year), 
            str(time.month).zfill(2), 
            str(time.day).zfill(2), 
            f'{location_code.lower()}*', 
            f'ut{str(time.hour).zfill(2)}', 
            ]
        filename = time.strftime(f'%Y%m%d_%H%M_{location_code.lower()}*.pgm.gz')
        # file_paths will only contain one path.
        file_paths = [_manager.find_file(filename, subdirectories=url_subdirectories, overwrite=overwrite)]
        
    # Find multiple image files.
    if time_range is not None:
        time_range = utils.validate_time_range(time_range)
        file_times = utils.get_filename_times(time_range, dt='minutes')
        file_paths = []

        for file_time in file_times:
            url_subdirectories = [
                str(file_time.year), 
                str(file_time.month).zfill(2), 
                str(file_time.day).zfill(2), 
                f'{location_code.lower()}*', 
                f'ut{str(file_time.hour).zfill(2)}', 
                ]
            filename = file_time.strftime(f'%Y%m%d_%H%M_{location_code.lower()}*.pgm.gz')
            try:
                file_paths.append(
                    _manager.find_file(filename, subdirectories=url_subdirectories, overwrite=overwrite)
                )
            except (FileNotFoundError, AssertionError) as err:
                if (missing_ok and 
                    (
                        ('does not contain any hyper references containing' in str(err)) or
                        ('Only one href is allowed' in str(err))
                    )):
                    continue
                else:
                    raise
    return file_paths


def load_skymap(location_code, time, overwrite):
    """
    Load a THEMIS ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-fomatted time string or datetime object. Must be in UT time. 
    overwrite: bool
        Redownload the file.
    """
    time = utils.validate_time(time)
    local_dir = local_base_dir / 'skymaps' / location_code.lower()
    skymap_top_url = skymap_base_url + location_code.lower()

    # Check if the skymaps are already downloaded.
    local_skymaps = list(pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav'))
    #TODO: Add a check to periodically redownload the skymap data, maybe once a month?
    if (len(local_skymaps) == 0) or overwrite:
        # Download the skymaps.
        local_skymaps = []
        parent_folders = download.Downloader(skymap_top_url)
        skymap_dirs = parent_folders.find_url(filename=f'{location_code.lower()}_*')
        for skymap_dir in skymap_dirs:
            d = download.Downloader(skymap_dir)
            d.find_url(filename=f'*skymap_{location_code.lower()}*.sav')
            d.download(local_dir)
            local_skymaps.append(d.save_path[0])

    skymap_filenames = [skymap_url.name for skymap_url in local_skymaps]
    skymap_file_dates = []
    for skymap_filename in skymap_filenames:
        date_match = re.search(r'\d{8}', skymap_filename)
        skymap_file_dates.append(datetime.strptime(date_match.group(), '%Y%m%d'))

    # Find the skymap_date that is closest and before time.
    # For reference: dt > 0 when time is after skymap_date.
    dt = np.array([(time - skymap_date).total_seconds() for skymap_date in skymap_file_dates])
    dt[dt < 0] = np.inf  # Mask out all skymap_dates after time.
    if np.all(~np.isfinite(dt)):
        # Edge case when time is before the first skymap_date.
        closest_index = 0
        warnings.warn(
            f'The requested skymap time={time} for THEMIS-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymaps[closest_index]
    skymap = _load_skymap(skymap_path)
    return skymap

def _load_skymap(skymap_path):
    """
    A helper function to load a THEMIS skymap and transform it.
    """
    # Load the skymap file and convert it to a dictionary.
    skymap_file = scipy.io.readsav(str(skymap_path), python_dict=True)['skymap']
    skymap_dict = {key: copy.copy(skymap_file[key][0]) for key in skymap_file.dtype.names}

    skymap_dict = _tranform_longitude_to_180(skymap_dict)
    skymap_dict = _flip_skymap(skymap_dict)
    skymap_dict['PATH'] = skymap_path
    return skymap_dict

def _flip_skymap(skymap):
    """
    IDL is a column-major language while Python is row-major. This function
    tranposes the 2- and 3-D arrays to make them compatable with the images
    that are saved in row-major.
    """
    for key in skymap:
        if hasattr(skymap[key], 'shape'):
            shape = skymap[key].shape
            if (len(shape) == 2) and (shape[0] == shape[1]):
                skymap[key] = skymap[key][::-1, :]  # For Az/El maps.
            elif (len(shape) == 3) and (shape[1] == shape[2]):
                skymap[key] = skymap[key][:, ::-1, :]  # For lat/lon maps
    return skymap


def _tranform_longitude_to_180(skymap):
    """
    Transform the SITE_MAP_LONGITUDE and FULL_MAP_LONGITUDE arrays from
    (0 -> 360) to (-180 -> 180).
    """
    skymap['SITE_MAP_LONGITUDE'] = np.mod(skymap['SITE_MAP_LONGITUDE'] + 180, 360) - 180

    # Don't take the modulus of NaNs
    valid_val_idx = np.where(~np.isnan(skymap['FULL_MAP_LONGITUDE']))
    skymap['FULL_MAP_LONGITUDE'][valid_val_idx] = (
        np.mod(skymap['FULL_MAP_LONGITUDE'][valid_val_idx] + 180, 360) - 180
    )
    return skymap

def _load_pgm(file_list, workers=1) -> Tuple[np.array, np.array]:
    """
    Read in a single PGM file or set of PGM files. The original author
    is Lukas Vollmerhaus.

    Parameters
    ----------
    file_list: str or pathlib.Path
        The file path(s).
    workers: int
        How many CPU cores to use to process multiple image files.

    Returns
    -------
    times
        A 1d numpy array of time stamps.
    images
        A 3d numpy array of images with the first dimension corresponding to
        time. Images are oriented such that pixel located at (0, 0) is in the 
        southeast corner.
    """
    # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes=workers)
    signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler

    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str):
        file_list = [file_list]
    elif isinstance(file_list, pathlib.Path):
        file_list = [str(file_list)]

    # call readfile function, run each iteration with a single input file from file_list
    # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
    data = []
    try:
        data = pool.map(__themis_readfile_worker, file_list)
    except KeyboardInterrupt:
        pool.terminate()  # gracefully kill children
        return np.empty((0, 0, 0), dtype=THEMIS_DT), [], []
    else:
        pool.close()

    # derive number of frames to prepare for
    total_num_frames = 0
    for i in range(0, len(data)):
        if (data[i][2] is True):
            continue
        total_num_frames += data[i][0].shape[2]

    # pre-allocate array sizes
    images = np.empty([256, 256, total_num_frames], dtype=THEMIS_DT)
    metadata_dict_list = [{}] * total_num_frames
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(data)):
        # check if file was problematic
        if (data[i][2] is True):
            problematic_file_list.append({
                "filename": data[i][3],
                "error_message": data[i][4],
            })
            continue

        # check if any data was read in
        if (len(data[i][1]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        real_num_frames = data[i][0].shape[2]

        # metadata dictionary list at data[][1]
        metadata_dict_list[list_position:list_position + real_num_frames] = data[i][1]
        images[:, :, list_position:list_position + real_num_frames] = data[i][0]  # image arrays at data[][0]
        list_position = list_position + real_num_frames  # advance list position

    # trim unused elements from predicted array sizes
    metadata_dict_list = metadata_dict_list[0:list_position]
    images = np.delete(images, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as uint16
    images = images.astype(np.uint16)
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, :]  # Flip north-south.
    times = np.array(
        [dateutil.parser.parse(dict_i['Image request start']).replace(tzinfo=None)
        for dict_i in metadata_dict_list]
    )
    return times, images


def __themis_readfile_worker(file):
    """
    Parse one THEMIS pgm file.
    """
    images = np.array([])
    metadata_dict_list = []
    first_frame = True
    metadata_dict = {}
    site_uid = ""
    device_uid = ""
    problematic = False
    error_message = ""

    # check file extension to see if it's gzipped or not
    try:
        if file.endswith("pgm.gz"):
            unzipped = gzip.open(file, mode='rb')
        elif file.endswith("pgm"):
            unzipped = open(file, mode='rb')
        elif file.endswith("pgm.bz2"):
            unzipped = bz2.open(file, mode='rb')
        else:
            print("Unrecognized file type: %s" % (file))
            problematic = True
            error_message = "Unrecognized file type"
            return images, metadata_dict_list, problematic, file, error_message
    except Exception as e:
        print("Failed to open file '%s' " % (file))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        return images, metadata_dict_list, problematic, file, error_message

    # read the file
    while True:
        # read a line
        try:
            line = unzipped.readline()
        except Exception as e:
            print("Error reading before image data in file '%s'" % (file))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            return images, metadata_dict_list, problematic, file, error_message

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):
            continue

        # process line
        if (line.startswith(b'#"')):
            # metadata lines start with #"<key>"
            try:
                line_decoded = line.decode("ascii")
            except Exception as e:
                # skip metadata line if it can't be decoded, likely corrupt file but don't mark it as one yet
                print("Warning: issue decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file))
                continue

            # split the key and value out of the metadata line
            line_decoded_split = line_decoded.split('"')
            if (len(line_decoded_split) != 3):
                print("Warning: issue splitting metadata line (line='%s', file='%s')" % (line_decoded, file))
                continue
            key = line_decoded_split[1]
            value = line_decoded_split[2].strip()

            # add entry to dictionary
            metadata_dict[key] = value

            # set the site/device uids, or inject the site and device UIDs if they are missing
            if ("Site unique ID" not in metadata_dict):
                metadata_dict["Site unique ID"] = site_uid
            else:
                site_uid = metadata_dict["Site unique ID"]
            if ("Imager unique ID" not in metadata_dict):
                metadata_dict["Imager unique ID"] = device_uid
            else:
                device_uid = metadata_dict["Imager unique ID"]

            # split dictionaries up per frame, exposure plus initial readout is
            # always the end of metadata for frame
            if (key.startswith("Exposure plus initial readout") or key.startswith("Exposure duration plus readout")):
                metadata_dict_list.append(metadata_dict)
                metadata_dict = {}
        elif line == b'65535\n':
            # there are 2 lines between "exposure plus read out" and the image
            # data, the first is b'256 256\n' and the second is b'65535\n'
            #
            # read image
            try:
                # read the image size in bytes from the file
                image_bytes = unzipped.read(THEMIS_IMAGE_SIZE_BYTES)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                image_np = np.frombuffer(image_bytes, dtype=THEMIS_DT)

                # change 1d numpy array into 256x256 matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (256, 256, 1))
            except Exception as e:
                print("Failed reading image data frame: %s" % (str(e)))
                metadata_dict_list.pop()  # remove corresponding metadata entry
                problematic = True
                error_message = "image data read failure: %s" % (str(e))
                continue  # skip to next frame

            # initialize image stack
            if first_frame:
                images = image_matrix
                first_frame = False
            else:
                images = np.dstack([images, image_matrix])  # depth stack images (on 3rd axis)

    # close gzip file
    unzipped.close()

    # check to see if the image is empty
    if (images.size == 0):
        print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file, error_message