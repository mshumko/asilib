"""
As part of the Canadian Space Agency's Geospace Observatory (GO) Canada initiative, the Auroral Imaging Group operates an array of redline auroral all-sky imagers designed by the AIG for the RISR-C CFI project. These systems are specifically designed to capture image data from only the 630.0nm wavelength (red coloured aurora). The Redline Geospace Observatory (REGO) array consists of 9 systems and are co-located with other auroral instrumentation that we operate to maximize the scientific impact. These imaging systems were deployed in late 2014 and continue to operate.
"""

from datetime import datetime, timedelta
import functools
import re
import warnings
import pathlib
import copy
import os
import dateutil.parser
import gzip
import signal
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.colors

import asilib
import asilib.asi.themis as themis
import asilib.utils as utils
import asilib.download as download
import asilib.skymap


pgm_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/'
skymap_base_url = 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/skymap/'
local_base_dir = asilib.config['ASI_DATA_DIR'] / 'rego'


def rego(
    location_code: str,
    time: utils._time_type = None,
    time_range: utils._time_range_type = None,
    alt: int = 110,
    custom_alt: bool = False,
    redownload: bool = False,
    missing_ok: bool = True,
    load_images: bool = True,
    acknowledge: bool = True,
    dark: bool = False,
    imager=asilib.Imager,
) -> asilib.Imager:
    """
    Create an Imager instance with the REGO ASI images and skymaps.

    Redline Emission Geospace Observatory (REGO) data is courtesy of Space Environment Canada (space-environment.ca). Use of the data must adhere to the rules of the road for that dataset. Please see below for the required data acknowledgement. Any questions about the REGO instrumentation or data should be directed to the University of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan (edonovan@ucalgary.ca).

    “The Redline Emission Geospace Observatory (REGO) is a joint Canada Foundation for Innovation and Canadian Space Agency project developed by the University of Calgary. REGO is operated and maintained by Space Environment Canada with the support of the Canadian Space Agency (CSA) [23SUGOSEC].”

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
    custom_alt: bool
        If True, asilib will calculate (lat, lon) skymaps assuming a spherical Earth. Otherwise, it will use the official skymaps (Courtesy of University of Calgary).

        .. note::
        
            The spherical model of Earth's surface is less accurate than the oblate spheroid geometrical representation. Therefore, there will be a small difference between these and the official skymaps.
    redownload: bool
        If True, will download the data from the internet, regardless of
        wether or not the data exists locally (useful if the data becomes
        corrupted).
    missing_ok: bool
        Wether to allow missing data files inside time_range (after searching
        for them locally and online).
    load_images: bool
        Create an Imager object without images. This is useful if you need to
        calculate conjunctions and don't need to download or load unnecessary data.
    acknowledge: bool
        If True, prints the acknowledgment statement for REGO. 
    imager: asilib.Imager
        Controls what Imager instance to return, asilib.Imager by default. This
        parameter is useful if you need to subclass asilib.Imager.

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        The THEMIS Imager instance.

    Examples
    --------
    >>> # Plot a single image.
    >>> from datetime import datetime
    >>> import matplotlib.pyplot as plt
    >>> import asilib.asi
    >>> import asilib.map
    >>> location_code = 'RANK'
    >>> time = datetime(2017, 9, 15, 2, 34, 0)
    >>> alt_km = 110
    >>> fig = plt.figure(figsize=(10, 6))
    >>> ax = fig.add_subplot(121)
    >>> bx = asilib.map.create_map(fig_ax=(fig, 122), lon_bounds = (-102, -82), lat_bounds = (58, 68))
    >>> asi = asilib.asi.rego(location_code, time=time, alt=alt_km)
    >>> asi.plot_fisheye(ax=ax)
    >>> asi.plot_map(ax=bx)
    >>> plt.tight_layout()
    >>> plt.show()

    Returns
    -------
    :py:meth:`~asilib.imager.Imager`
        A REGO ASI instance with the time stamps, images, skymaps, and metadata.
    """
    if time is not None:
        time = utils.validate_time(time)
    else:
        time_range = utils.validate_time_range(time_range)

    local_pgm_dir = local_base_dir / 'images' / location_code.lower()
    
    file_search_str = functools.partial(_glob_filename, dark=dark)

    if load_images:
        # Download and find image data
        file_paths = themis._get_pgm_files(
            'rego',
            location_code,
            time,
            time_range,
            pgm_base_url,
            local_pgm_dir,
            redownload,
            missing_ok,
            file_search_str=file_search_str
        )

        start_times = len(file_paths) * [None]
        end_times = len(file_paths) * [None]
        for i, file_path in enumerate(file_paths):
            date_match = re.search(r'\d{8}_\d{4}', file_path.name)
            start_times[i] = datetime.strptime(date_match.group(), '%Y%m%d_%H%M')
            end_times[i] = start_times[i] + timedelta(minutes=1)
        file_info = {
            'path': file_paths,
            'start_time': start_times,
            'end_time': end_times,
            'loader': _load_rego_pgm,
        }
    else:
        file_info = {
            'path': [],
            'start_time': [],
            'end_time': [],
            'loader': [],
        }
    if time_range is not None:
        file_info['time_range'] = time_range
    else:
        file_info['time'] = time

    # Download and find the appropriate skymap
    if time is not None:
        _time = time
    else:
        _time = time_range[0]
    _skymap = rego_skymap(location_code, _time, redownload=redownload)
    
    if custom_alt==False:
        alt_index = np.where(_skymap['FULL_MAP_ALTITUDE'] / 1000 == alt)[0] #Compares the altitudes versus the ones provided by default and chooses the correct index that correlates to the chosen alitudes
        assert (
            len(alt_index) == 1
        ), f'{alt} km is not in the valid skymap altitudes: {_skymap["FULL_MAP_ALTITUDE"]/1000} km. If you want a custom altitude with less percision, please use the custom_alt keyword'
        alt_index = alt_index[0]
        lat=_skymap['FULL_MAP_LATITUDE'][alt_index, :, :] #selects lat lon coordinates from data provided in skymap
        lon=_skymap['FULL_MAP_LONGITUDE'][alt_index, :, :]
    else:
        lat,lon = asilib.skymap.geodetic_skymap( #Spherical projection for lat lon coordinates
            (float(_skymap['SITE_MAP_LATITUDE']), float(_skymap['SITE_MAP_LONGITUDE']), float(_skymap['SITE_MAP_ALTITUDE']) / 1e3),
            _skymap['FULL_AZIMUTH'],
            _skymap['FULL_ELEVATION'],
            alt
            )

    skymap = {
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'el': _skymap['FULL_ELEVATION'],
        'az': _skymap['FULL_AZIMUTH'],
        'path': _skymap['PATH'],
    }

    meta = {
        'array': 'REGO',
        'location': location_code.upper(),
        'lat': float(_skymap['SITE_MAP_LATITUDE']),
        'lon': float(_skymap['SITE_MAP_LONGITUDE']),
        'alt': float(_skymap['SITE_MAP_ALTITUDE']) / 1e3,
        'cadence': 3,
        'resolution': (512, 512),
        'acknowledgment':(
            'Redline Emission Geospace Observatory (REGO) data is courtesy of Space Environment '
            'Canada (space-environment.ca). Use of the data must adhere to the rules of the road '
            'for that dataset. Please see below for the required data acknowledgement. Any '
            'questions about the REGO instrumentation or data should be directed to the University '
            'of Calgary, Emma Spanswick (elspansw@ucalgary.ca) and/or Eric Donovan '
            '(edonovan@ucalgary.ca).\n\n"The Redline Emission Geospace Observatory (REGO) is a joint '
            'Canada Foundation for Innovation and Canadian Space Agency project developed by the '
            'University of Calgary. REGO is operated and maintained by Space Environment Canada '
            'with the support of the Canadian Space Agency (CSA) [23SUGOSEC]."'
            )
    }
    plot_settings = {
        'color_map': matplotlib.colors.LinearSegmentedColormap.from_list('black_to_red', ['k', 'r']),
        'color_bounds':(300, 1000)
    }

    if acknowledge and ('rego' not in asilib.config['ACKNOWLEDGED_ASIS']):
        print(meta['acknowledgment'])
        asilib.config['ACKNOWLEDGED_ASIS'].append('rego')   
    return imager(file_info, meta, skymap, plot_settings=plot_settings)


def rego_info() -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the REGO ASI names and locations.

    Returns
    -------
    pd.DataFrame
        A table of REGO imager names and locations.
    """
    path = pathlib.Path(asilib.__file__).parent / 'data' / 'asi_locations.csv'
    df = pd.read_csv(path)
    df = df[df['array'] == 'REGO']
    return df.reset_index(drop=True)


def rego_skymap(location_code: str, time: utils._time_type, redownload: bool = False) -> dict:
    """
    Load a REGO ASI skymap file.

    Parameters
    ----------
    location_code: str
        The four character location name.
    time: str or datetime.datetime
        A ISO-formatted time string or datetime object. Must be in UT time.
    redownload: bool
        Redownload the file.

    Returns
    -------
    dict
        The skymap.
    """
    time = utils.validate_time(time)
    local_dir = local_base_dir / 'skymaps' / location_code.lower()
    local_dir.mkdir(parents=True, exist_ok=True)
    skymap_top_url = skymap_base_url + location_code.lower() + '/'

    if redownload:
        # Delete any existing skymap files.
        local_skymap_paths = pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        for local_skymap_path in local_skymap_paths:
            os.unlink(local_skymap_path)
        local_skymap_paths = _download_all_skymaps(
            location_code, skymap_top_url, local_dir, redownload=redownload
        )

    else:
        local_skymap_paths = sorted(
            pathlib.Path(local_dir).rglob(f'*skymap_{location_code.lower()}*.sav')
        )
        # TODO: Add a check to periodically redownload the skymap data, maybe once a month?
        if len(local_skymap_paths) == 0:
            local_skymap_paths = _download_all_skymaps(
                location_code, skymap_top_url, local_dir, redownload=redownload
            )

    skymap_filenames = [local_skymap_path.name for local_skymap_path in local_skymap_paths]
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
            f'The requested skymap time={time} for REGO-{location_code.upper()} is before first '
            f'skymap file dated: {skymap_file_dates[0]}. This skymap file will be used.'
        )
    else:
        closest_index = np.nanargmin(dt)
    skymap_path = local_skymap_paths[closest_index]
    skymap = _load_skymap(skymap_path)
    return skymap


def _download_all_skymaps(location_code, url, save_dir, redownload):
    d = download.Downloader(url, headers={'User-Agent':'asilib'})
    # Find the dated subdirectories
    ds = d.ls(f'{location_code.lower()}')

    save_paths = []
    for d_i in ds:
        ds = d_i.ls(f'*skymap_{location_code.lower()}*.sav')
        for ds_j in ds:
            save_paths.append(ds_j.download(save_dir, redownload=redownload))
    return save_paths


def _load_skymap(skymap_path):
    """
    A helper function to load a REGO skymap and transform it.
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
    tranposes the 2- and 3-D arrays to make them compatible with the images
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


def _load_rego_pgm(path):
    images, meta, problematic_file_list = read(str(path))
    if len(problematic_file_list):
        raise ValueError(f'A problematic PGM file: {problematic_file_list[0]}')
    images = np.moveaxis(images, 2, 0)
    images = images[:, ::-1, :]  # Flip so north is up and .
    times = np.array(
        [
            dateutil.parser.parse(dict_i['Image request start']).replace(tzinfo=None)
            for dict_i in meta
        ]
    )
    return times, images

def _glob_filename(time, location_code, _, dark=False):
    """
    Return a file search string to pass into asilib.Downloader.
    """
    if dark:
        return f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}_rego*6300_dark.pgm.gz'
    else:
        return f'{time.strftime("%Y%m%d_%H%M")}_{location_code.lower()}_rego*6300.pgm.gz'


"""
TThe data reader below was developed by the University of Calgary under the 
MIT License. You can find the original source files at:  
https://github.com/ucalgary-srs/rego-imager-readfile

MIT License

Copyright (c) 2020 University of Calgary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# globals
REGO_EXPECTED_HEIGHT = 512
REGO_EXPECTED_WIDTH = 512
REGO_DT = np.dtype("uint16")
REGO_DT = REGO_DT.newbyteorder('>')  # force big endian byte ordering


def read(file_list, workers=1, first_frame=False, no_metadata=False, quiet=False):
    """
    Read in a single PGM file or set of PGM files

    :param file_list: filename or list of filenames
    :type file_list: str
    :param workers: number of worker processes to spawn, defaults to 1
    :type workers: int, optional
    :param first_frame: only read the first frame for each file, defaults to False
    :type first_frame: bool, optional
    :param no_metadata: exclude reading of metadata (performance optimization if
                        the metadata is not needed), defaults to False
    :type no_metadata: bool, optional
    :param quiet: reduce output while reading data
    :type quiet: bool, optional

    :return: images, metadata dictionaries, and problematic files
    :rtype: numpy.ndarray, list[dict], list[dict]
    """
    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str):
        file_list = [file_list]

    # check workers
    if (workers > 1):
        try:
            # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(processes=workers)
            signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler
        except ValueError:
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=workers)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        data = []
        try:
            data = pool.map(functools.partial(
                __rego_readfile_worker,
                first_frame=first_frame,
                no_metadata=no_metadata,
                quiet=quiet,
            ), file_list)
        except KeyboardInterrupt:
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0), dtype=REGO_DT), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(__rego_readfile_worker(
                f,
                first_frame=first_frame,
                no_metadata=no_metadata,
                quiet=quiet,
            ))

    # derive number of frames to prepare for
    total_num_frames = 0
    image_height = REGO_EXPECTED_HEIGHT
    image_width = REGO_EXPECTED_WIDTH
    for i in range(0, len(data)):
        if (data[i][2] is True):
            continue
        total_num_frames += data[i][0].shape[2]
        image_height = data[i][0].shape[0]
        image_width = data[i][0].shape[1]

    # pre-allocate array sizes
    images = np.empty([image_height, image_width, total_num_frames], dtype=REGO_DT)
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

    # return
    data = None
    return images, metadata_dict_list, problematic_file_list


def __rego_readfile_worker(file, first_frame=False, no_metadata=False, quiet=False):
    # init
    images = np.array([])
    metadata_dict_list = []
    is_first = True
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
        else:
            if (quiet is False):
                print("Unrecognized file type: %s" % (file))
            problematic = True
            error_message = "Unrecognized file type"
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file, error_message
    except Exception as e:
        if (quiet is False):
            print("Failed to open file '%s' " % (file))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        try:
            unzipped.close()
        except Exception:
            pass
        return images, metadata_dict_list, problematic, file, error_message

    # read the file
    prev_line = None
    line = None
    while True:
        # break out depending on first_frame param
        if (first_frame is True and is_first is False):
            break

        # read a line
        try:
            prev_line = line
            line = unzipped.readline()
        except Exception as e:
            if (quiet is False):
                print("Error reading before image data in file '%s'" % (file))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            try:
                unzipped.close()
            except Exception:
                pass
            return images, metadata_dict_list, problematic, file, error_message

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):
            continue

        # process line
        if (line.startswith(b'#"')):
            if (no_metadata is True):
                metadata_dict = {}
                metadata_dict_list.append(metadata_dict)
            else:
                # metadata lines start with #"<key>"
                try:
                    line_decoded = line.decode("ascii")
                except Exception as e:
                    # skip metadata line if it can't be decoded, likely corrupt file
                    if (quiet is False):
                        print("Error decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file))
                    problematic = True
                    error_message = "error decoding metadata line: %s" % (str(e))
                    continue

                # split the key and value out of the metadata line
                line_decoded_split = line_decoded.split('"')
                if (len(line_decoded_split) != 3):
                    if (quiet is False):
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
                if (key.startswith("Exposure plus readout")):
                    metadata_dict_list.append(metadata_dict)
                    metadata_dict = {}
        elif line == b'65535\n':
            # there are 2 lines between "exposure plus read out" and the image
            # data, the first is the image dimensions and the second is the max
            # value
            #
            # check the previous line to get the dimensions of the image
            prev_line_split = prev_line.decode("ascii").strip().split()
            image_width = int(prev_line_split[0])
            image_height = int(prev_line_split[1])
            bytes_to_read = image_width * image_height * 2  # 16-bit image depth

            # read image
            try:
                # read the image size in bytes from the file
                image_bytes = unzipped.read(bytes_to_read)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                image_np = np.frombuffer(image_bytes, dtype=REGO_DT)

                # change 1d numpy array into matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))
            except Exception as e:
                if (quiet is False):
                    print("Failed reading image data frame: %s" % (str(e)))
                metadata_dict_list.pop()  # remove corresponding metadata entry
                problematic = True
                error_message = "image data read failure: %s" % (str(e))
                continue  # skip to next frame

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                images = np.dstack([images, image_matrix])  # depth stack images (on 3rd axis)

    # close gzip file
    unzipped.close()

    # check to see if the image is empty
    if (images.size == 0):
        if (quiet is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file, error_message