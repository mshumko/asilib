# Original header and licence is copied below from supermag-api.py
# ================
# Author S. Antunes, based on supermag-api.pro by R.J.Barnes

# (c) 2021    The Johns Hopkins University Applied Physics Laboratory
# LLC. All Rights Reserved. 

# This material may be only be used, modified, or reproduced by or for
# the U.S. Government pursuant to the license rights granted under the 
# clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other
# permission, please contact the Office of Technology Transfer at JHU/APL.

# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED "AS IS."
# JHU/APL MAKES NO REPRESENTATION OR WARRANTY WITH RESPECT TO THE
# PERFORMANCE OF THE MATERIALS, INCLUDING THEIR SAFETY, EFFECTIVENESS, 
# OR COMMERCIAL VIABILITY, AND DISCLAIMS ALL WARRANTIES IN THE
# MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT LIMITED TO)
# ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF
# INTELLECTUAL PROPERTY OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE
# MATERIAL ASSUMES THE ENTIRERISK AND LIABILITY FOR USING THE
# MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY USER OF THE
# MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
# DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL.
# INCLUDING, BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS. 

import json
import re
import datetime
import dateutil.parser
import importlib
# the 'certifi' library is required at APL and other sites that require SSL certs for web fetches.
# If you need this, install certifi (python -m pip install certifi)
if importlib.util.find_spec("certifi") is not None:
    import certifi

# Sandy's important comment: pd Dataframes are awesome! To manipulate, just pull out what you need.
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates

baseurl = "https://supermag.jhuapl.edu/"


class SuperMAG():
    """
    Download, load, and plot SuperMAG magnetometer data from APL's API.

    This code is adapted from the supermag-api.py (https://supermag.jhuapl.edu/mag/?tab=api) which
    is credited to S. Antunes, which in turn was credited to R.J. Barnes.

    Methods
    -------
    location_codes()
        Download a list of SuperMAG magnetometer location code names for a given time range.
    mag_data(station, query_parameters)
        Download and return SuperMAG magnetometer data for a given station.
    indices(query_parameters)
        Download the SuperMAG indices data.

    Attributes
    ----------
    userid : str
        Your SuperMAG logon name.
    time_range : list of datetime
        A list containing the start and end times for the data request.
    api_url: str
        The last API URL used to fetch data.
    
    
    """
    def __init__(self, userid, time_range):
        self.userid = userid
        self.time_range = time_range
        for i, t_i in enumerate(self.time_range):
            if isinstance(t_i, (str)):
                self.time_range[i] = dateutil.parser.isoparse(t_i)
        self.api_url = ''  # initialize api_url attribute so it exists.
        return
    
    def location_codes(self):
        """
        Download a list of SuperMAG magnetometer location code names.

        Returns
        -------
        stations : list
            A list of SuperMAG magnetometer location code names.

        Example
        -------
        # Grab a day's worth of station codes. The output list should be 184 stations
        >>> stations = supermag_location_codes('myname', '2019-11-15T10:40', 86400)
        """

        # construct URL             
        urlstr = self._base_url('inventory.php')

        # get the string array of stations
        stations = self._get_url_data(urlstr,'raw')

        # first data item is how many stations were found
        numstations = int(stations[0])
        if numstations > 0: 
            return stations[1:]
        else: 
            return []


    def indices(self, query_parameters=''):
        """
        Download the SuperMAG indices data.

        Parameters
        ----------
        query_parameters : str
            A comma-separated string of data options to include. See the `SuperMAG API
            documentation <https://supermag.jhuapl.edu/mag/lib/content/api/supermag_doc_python.pdf>`_
            for details.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> import supermag
        >>>
        >>> userid=input("Enter your SuperMAG userid: ")
        >>> time_range = ['2022-11-03T00:00','2022-11-05T00:00']
        >>> 
        >>> sm = supermag.SuperMAG(userid, time_range)
        >>> indices = sm.indices('sml,smr,baseline=yearly')
        >>> print(indices.keys())  # Index(['SML', 'smr'], dtype='object')
        >>> 
        >>> _, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        >>> ax[0].plot(indices.index, indices.SML)
        >>> ax[1].plot(indices.index, indices.smr)
        >>> ax[0].set(
        >>>     ylabel='SML [nT]', 
        >>>     ylim=(-1600, 0), 
        >>>     title=f'SuperMAG indices: {sm.time_range[0]} to {sm.time_range[1]}'
        >>>     )
        >>> ax[1].set(
        >>>     ylabel='SMR [nT]', 
        >>>     xlabel='Time', 
        >>>     ylim=(-80, 0)
        >>>     )
        >>> 
        >>> supermag.format_time_axis(ax[-1])
        >>> plt.tight_layout()
        >>> plt.show()
        """
        urlstr = self._base_url('indices.php')
        indices = self._sm_keycheck_indices(query_parameters)
        urlstr += indices

        # get the string array of JSON data         
        data_list = self._get_url_data(urlstr,'json')

        # default, converts the json 'list of dictionaries' into a dataframe
        data_df = pd.DataFrame(data_list)
        data_df.index = pd.to_datetime(data_df['tval'],unit='s')
        data_df.drop(columns=['tval'],inplace=True)
        return data_df


    def mag_data(self, station, query_parameters):
        """
        Download and return SuperMAG magnetometer data for a given station.

        Parameters
        ----------
        station : str
            The SuperMAG location code name for the magnetometer station.
        query_parameters : str
            A comma-separated string of data options to include. See the `SuperMAG API
            documentation <https://supermag.jhuapl.edu/mag/lib/content/api/supermag_doc_python.pdf>`_
            for details.
        
        Returns
        -------
        data_df : pandas.DataFrame
            A DataFrame containing the magnetometer data from the specified station.
        """        
        urlstr = self._base_url('data-api.php')
        indices = self._sm_keycheck_data(query_parameters)
        urlstr += indices
        urlstr += '&station='+station.upper()

        data_list = self._get_url_data(urlstr,'json')

        data_df = pd.DataFrame(data_list)
        data_df.index = pd.to_datetime(data_df['tval'],unit='s')
        data_df.drop(columns=['tval'],inplace=True)
        return data_df
    
    def _get_start_extent(self):
        # compute start string and extent in seconds from time_range
        extent = (self.time_range[1] - self.time_range[0]).total_seconds()
        return self.time_range[0].strftime("%Y-%m-%dT%H:%M"), int(extent)
    

    def _base_url(self, page):
        """
        Create the base URL for SuperMAG API calls.
        """
        start, extent = self._get_start_extent()

        urlstr = baseurl + 'services/'+page+'?python&nohead'
        urlstr+='&start='+start
        urlstr += '&logon='+self.userid
        urlstr+='&extent='+ ("%12.12d" % extent)
        return(urlstr)
    
    def _sm_keycheck_data(self, query_parameters):
        # internal helper function
        toggles=['mlt','mag','geo','decl','sza','delta=start','baseline=yearly','baseline=none']

        myflags=''
        flags=[x.strip() for x in query_parameters.split(',')]

        for i in range(0,len(flags)):
            chk=flags[i]
            chk=chk.lower()
            # check for the '*all', also individual keys, and assemble url flags
            if chk == 'all': myflags += '&mlt&mag&geo&decl&sza'
            for ikey in range(0,len(toggles)):
                if chk == toggles[ikey]: myflags += '&'+toggles[ikey]

        return(myflags)


    def _sm_keycheck_indices(self, query_parameters):
        # internal helper function
        # For category='indices', always returns:
        #                tval
        # additional flags to return data include:
        #                indicesall (or its alias: all)
        #    (or any of)
        #                baseall, sunall, darkall, regionalall, plusall
        #    (or specify individual items to include, from the sets below)
        #                
        basekeys=["sme","sml","smu","mlat","mlt","glat","glon","stid","num"]
        # sunkeys: alias allowed of SUN___ -> ___s
        sunkeys=["smes","smls","smus","mlats","mlts","glats","glons","stids","nums"]
        # darkkeys: alias allowed of DARK___ -> ___d
        darkkeys=["smed","smld","smud","mlatd","mltd","glatd","glond","stidd","num"]
        # regkeys: alias allowed of REGIONAL___ -> ___r
        regkeys=["smer","smlr","smur","mlatr","mltr","glatr","glonr","stidr","numr"]
        pluskeys=["smr","ltsmr","ltnum","nsmr"]
        indiceskeys = basekeys + sunkeys + darkkeys + regkeys + pluskeys
        # 'all' means all the above                                                                                                 

        imfkeys=["bgse","bgsm","vgse","vgsm"] # or imfall for all these                        
        swikeys=["pdyn","epsilon","newell","clockgse","clockgsm","density"] # % or swiall for all these                                                                                                                         
        myflags=''
        indices='&indices='
        swi='&swi='
        imf='&imf='

        flags=[x.strip() for x in query_parameters.split(',')]

        for i in range(0,len(flags)):
            chk=flags[i]
            chk=chk.lower()
            
            # check for the '*all', also individual keys, and assemble url flags
            if chk == 'all': indices += 'all,'
            if chk == 'indicesall': indices += 'all,'
            if chk == 'imfall': imf += 'all,'
            if chk == 'swiall': swi += 'all,'
            # available keywords, we allow both the url version and the
            # aliases of "SUN___ -> ___s", "DARK___ -> ___d", "REGIONAL___ -> ___r"

            for ikey in range(0,len(indiceskeys)):
                mykey=indiceskeys[ikey]
                sunkey="sun"+mykey # allow alias
                darkkey="dark"+mykey # allow alias
                regkey1="regional"+mykey # allow alias
                regkey2="reg"+mykey # allow alias
                if chk == mykey:
                    indices += mykey+','    # base key is correct
                elif sunkey == mykey:
                    indices += mykey+'s,'    # alias, so base key + 's'
                elif darkkey == mykey:
                    indices += mykey+'d,'    # alias, so base key + 'd'
                elif regkey1 == mykey or regkey2 == mykey:
                    indices += mykey+'r,'    # alias, so base key + 'r'

            for ikey in range(0,len(swikeys)):
                if chk == swikeys[ikey]: swi += swikeys[ikey] + ','
            
            for ikey in range(0,len(imfkeys)):
                if chk == imfkeys[ikey]: imf += imfkeys[ikey] + ','
        
            # more aliases to the user
            if chk == 'baseall': indices += ','.join(basekeys)
            if chk == 'sunall': indices += ','.join(sunkeys)
            if chk == 'darkall': indices += ','.join(darkkeys)
            if chk == 'regionalall' or chk == 'regall': indices += ','.join(regkeys)
            if chk == 'plusall': indices += ','.join(pluskeys)

        # clean it up a bit by removing extraneous tags/characters
        if indices == "&indices=": indices=""
        if swi == "&swi=": swi=""
        if imf == "&imf=": imf=""
        # add them together
        myflags = indices + swi + imf
        # a little more cleaning for tidiness, removes extraneous commas
        myflags = re.sub(',&','&',myflags)
        myflags = re.sub(',$','',myflags)

        return(myflags)

    def _get_url_data(self, api_url,fetch='raw'):
        """
        Given a fetchurl, fetch the data from SuperMAG API.

        Parameters
        ----------
        fetchurl : str
            The complete URL to fetch data from.
        fetch : str, optional
            The format of data to fetch. Options are 'raw' for raw text data
            or 'json' for JSON formatted data.
        """
        self.api_url = api_url
        try:
            cafile=certifi.where()
        except:
            cafile=''
        
        response = requests.get(api_url, verify=cafile)
        response.raise_for_status()
        longstring = response.text
        mydata = longstring.split('\n')

        assert 'ERROR: Invalid username' != mydata[0], (
            f'The user "{self.userid}" is an invalid SuperMAG username.'
            )
    
        if fetch == 'json':
            if len(longstring) > 3:
                mydata = json.loads(longstring)
            else:
                mydata=[] # just the word 'OK', no data, so return no data
        elif fetch == 'raw':
            if f'ERROR' in mydata[0]: 
                raise ValueError(f'ERROR returned from SuperMAG API: {mydata[0]}')
        else:
            raise ValueError(f'{fetch=} is not recognized, must be "raw" or "json".')
    
        return mydata

def format_time_axis(ax):
    """
    Format the x-axis of a time series plot to include date only on the first tick and at midnight.
    Include the HH:MM on all ticks.
    """
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_xaxis))
    return

def _format_xaxis(tick_val, tick_pos):
    """
    The tick magic to include the date only on the first tick, and ticks at midnight.
    """
    tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
    if (tick_pos==0) or ((tick_time.hour == 0) and (tick_time.minute == 0)):
        ticks = tick_time.strftime('%H:%M') + '\n' + tick_time.strftime('%Y-%m-%d')
    else:
        ticks = tick_time.strftime('%H:%M')
    return ticks

def _sm_grabme(dataf,key,subkey):
    # syntactical sugar to grab nested subitems from a dataframe
    data = dataf[key]
    subdata = [temp[subkey] for temp in data]
    return(subdata)
    
# Unlike IDL, which returns as Array or Struct,
# we return as List (of dictionaries) or DataFrame

def sm_microtest(choice,userid):
    # 3 simple unit tests to verify the core fetches work
    

    start='2019-11-15T10:40'

    if choice == 1 or choice == 4:
        stations = supermag_location_codes(userid,start,3600)
        print(stations)

    if choice == 2 or choice == 4:
        data = SuperMAGGetData(userid,start,3600,'all,delta=start,baseline=yearly','HBK')
        print(data)
        print(data.keys())
        
        tval=data.index
        mlt=data.mlt
        ### Python way
        N_nez = [temp['nez'] for temp in data.N]
        N_geo = [temp['geo'] for temp in data.N]
        ### or, supermag helper shorthand way
        N_nez = _sm_grabme(data,'N','nez')
        N_geo = _sm_grabme(data,'N','geo')
        #
        plt.plot(tval,N_nez)
        plt.plot(tval,N_geo)
        plt.ylabel('N_geo vs N_nez')
        plt.xlabel('date')
        plt.show()

    if choice == 3 or choice == 4:
        idxdata = SuperMAGGetIndices(userid,start,3600,'swiall,density,darkall,regall,smes')

        idxdata.keys()
        tval=idxdata.index
        hours=list(range(24))
        y=idxdata.SMLr
        for i in range(len(tval)-1):
            plt.plot( hours, y[i] )
            plt.ylabel('SMLr')
            plt.xlabel('hour')
            plt.title('SMLr variation by hour, for successive days')
        plt.show()

def supermag_testing(userid):

    start='2019-11-15T10:40'

    stations = supermag_location_codes(userid,start,3600)


    # DATA fetches
    # BARE CALL, dataframe returned
    mydata1a = SuperMAGGetData(userid,start,3600,'','HBK')
    mydata1a                # is 1440 rows x 6 columns dataframe
    mydata1a.keys() # Index(['tval', 'ext', 'iaga', 'N', 'E', 'Z'], dtype='object')

    # CALL with ALLINDICES, dataframe returned
    mydata1a = SuperMAGGetData(userid,start,3600,'all','HBK')
    mydata1a                # is 1440 rows x 12 columns dataframe
    mydata1a.keys() # Index(['tval', 'ext', 'iaga', 'glon', 'glat', 'mlt', 'mcolat', 'decl', 'sza', 'N', 'E', 'Z'], dtype='object')

    # BARE CALL, list returned
    mydata1b = SuperMAGGetData(userid,start,3600,'','HBK',FORMAT='list')
    len(mydata1b)    # is 1440 rows of dicts (key-value pairs)
    mydata1b[0:1]    # {'tval': 1572726240.0, 'ext': 60.0, 'iaga': 'DOB', 'N': {'nez': -3.942651, 'geo': -5.964826}, 'E': {'nez': 4.492887, 'geo': 0.389075}, 'Z': {'nez': 7.608168, 'geo': 7.608168}}

    # CALL with ALLINDICES, list returned
    mydata1b = SuperMAGGetData(userid,start,3600,'all','HBK',FORMAT='list')
    mydata1b                # is 1440 rows of dicts (key-value pairs)
    mydata1b[0:1]    # {'tval': 1572726240.0, 'ext': 60.0, 'iaga': 'DOB', 'glon': 9.11, 'glat': 62.07, 'mlt': 21.694675, 'mcolat': 30.361519, 'decl': 3.067929, 'sza': 124.698227, 'N': {'nez': -3.942651, 'geo': -5.964826}, 'E': {'nez': 4.492887, 'geo': 0.389075}, 'Z': {'nez': 7.608168, 'geo': 7.608168}}
    
    ####################
    # INDICES fetches
    idxdata = SuperMAGGetIndices(userid,start,3600)
    idxdata    # empty!

    idxdata = SuperMAGGetIndices(userid,start,3600,'all,swiall,imfall')
    idxdata    # 1440 rows x 77 columns dataframe
    idxdata.keys() # Index(['tval', 'SME', 'SML', 'SMLmlat', 'SMLmlt', 'SMLglat', 'SMLglon', 'SMLstid', 'SMU', 'SMUmlat', 'SMUmlt', 'SMUglat', 'SMUglon', 'SMUstid', 'SMEnum', 'SMEs', 'SMLs', 'SMLsmlat', 'SMLsmlt', 'SMLsglat', 'SMLsglon', 'SMLsstid', 'SMUs', 'SMUsmlat', 'SMUsmlt', 'SMUsglat', 'SMUsglon', 'SMUsstid', 'SMEsnum', 'SMEd', 'SMLd', 'SMLdmlat', 'SMLdmlt', 'SMLdglat', 'SMLdglon', 'SMLdstid', 'SMUd', 'SMUdmlat', 'SMUdmlt', 'SMUdglat', 'SMUdglon', 'SMUdstid', 'SMEdnum', 'SMEr', 'SMLr', 'SMLrmlat', 'SMLrmlt', 'SMLrglat', 'SMLrglon', 'SMLrstid', 'SMUr', 'SMUrmlat', 'SMUrmlt', 'SMUrglat', 'SMUrglon', 'SMUrstid', 'SMErnum', 'smr', 'smr00', 'smr06', 'smr12', 'smr18', 'smrnum', 'smrnum00', 'smrnum06', 'smrnum12', 'smrnum18', 'bgse', 'bgsm', 'vgse', 'vgsm', 'clockgse', 'clockgsm', 'density', 'dynpres', 'epsilon', 'newell'], dtype='object')
    #
    # just INDICESALL = 67 columns, above 'tval' through 'smrnum18'
    # just IMFALL = 5 columns, Index(['tval', 'bgse', 'bgsm', 'vgse', 'vgsm'], dtype='object')
    # just SWIALL = 7 columns, Index(['tval', 'clockgse', 'clockgsm', 'density', 'dynpres', 'epsilon', 'newell'], dtype='object')
    #
    tval = idxdata.tval
    density = idxdata.density
    vgse = idxdata.vgse
    # or all as 1 line of code
    tval, density, vgse = idxdata.tval, idxdata.density, idxdata.vgse
    # note that vgse is itself a dictionary of values for X/Y/Z, so you can get subitems from it like this
    vgse_x = [d.get('X') for d in idxdata.vgse]

    # to save the data, there are many formats.    Here is how to save as csv
    idxdata.to_csv('mydata.csv')

    # to read it back in later
    mydata2b=pd.read_csv('mydata.csv',index_col=0) # you can read it into any variable name, we just used 'mydata2b' as an example
    # now you can do all the above items again, with one exception: each line of the CVS file got split into a dict (key-value pairs) but items like 'vsge' are part of the pandas structure
    # the 'd.get()' approach will _not_ work once read from csv
    stationlist = mydata2b.SMLrstid # item is a pandas series (not python list)
    print(stationlist[0]) # prints a list of stations as a string, but cannot easily access a single item because it is a pandas series
    # so you can convert a pandas series to a list
    stationlist2=sm_csvitem_to_list(mydata2b.SMLrstid) # goal is a list of stations
    slist = stationlist2[0] # grabs a list of stations for row 0
    s1 = stationlist2[0][0] # grabs the first station for row 0

    vgse=sm_csvitem_to_dict(mydata2b.vgse) # goal is a dict of coords or other values
    x = vgse[0]['X'] # grab just the 'X' value for the 1st row of data
    vgse_x = [mydat['X'] for mydat in vgse] # grab all the 'X' values as a new list
    vgse_xyz = [(mydat['X'],mydat['Y'],mydat['Z']) for mydat in vgse] # grab all 3

    # We also offer a list format, for users who prefer to work in python lists
    mydata2c = SuperMAGGetIndices(userid,start,3600,'all,swiall,imfall',FORMAT='list')
    len(mydata2c)    # is 1440 rows of dicts (key-value pairs)
    mydata2c[0:1] # {'tval': 1572726240.0, 'SME': 58.887299, 'SML': -27.709055, 'SMLmlat': 73.529922, 'SMLmlt': 23.321493, 'SMLglat': 76.510002, 'SMLglon': 25.01, 'SMLstid': 'HOP', 'SMU': 31.178246, 'SMUmlat': 74.702339, 'SMUmlt': 2.090216, 'SMUglat': 79.480003, 'SMUglon': 76.980003, 'SMUstid': 'VIZ', 'SMEnum': 118, 'SMEs': 34.451469, 'SMLs': -16.599854, 'SMLsmlat': 62.368008, 'SMLsmlt': 9.399416, 'SMLsglat': 62.299999, 'SMLsglon': 209.800003, 'SMLsstid': 'T39', 'SMUs': 17.851616, 'SMUsmlat': 73.989975, 'SMUsmlt': 18.228394, 'SMUsglat': 67.93, 'SMUsglon': 306.429993, 'SMUsstid': 'ATU', 'SMEsnum': 54, 'SMEd': 58.887299, 'SMLd': -27.709055, 'SMLdmlat': 73.529922, 'SMLdmlt': 23.321493, 'SMLdglat': 76.510002, 'SMLdglon': 25.01, 'SMLdstid': 'HOP', 'SMUd': 31.178246, 'SMUdmlat': 74.702339, 'SMUdmlt': 2.090216, 'SMUdglat': 79.480003, 'SMUdglon': 76.980003, 'SMUdstid': 'VIZ', 'SMEdnum': 64, 'SMEr': [29.685059, 29.857538, 31.387127, 41.707573, 10.320444, 10.885443, 9.604616, 13.479583, 15.471248, 15.471248, 15.714731, 5.434914, 12.13654, 11.156847, 9.62884, 14.752592, 14.752592, 24.204388, 21.41181, 21.41181, 27.121195, 46.345322, 51.403328, 51.403328], 'SMLr': [-27.709055, 1.320708, -0.208882, -10.529325, -10.529325, -10.529325, -9.248499, -13.123466, -16.599854, -16.599854, -16.599854, -5.449972, -5.449972, -4.470279, -2.942272, -6.352773, -6.352773, -6.352773, -3.560194, -3.560194, -7.514064, -22.651047, -27.709055, -27.709055], 'SMLrmlat': [73.529922, 51.264774, 47.791527, 66.696564, 66.696564, 66.696564, 41.771515, 70.602707, 62.368008, 62.368008, 62.368008, 67.471809, 67.471809, 60.639145, 68.500282, 72.20977, 72.20977, 72.20977, 75.762718, 75.762718, 77.33667, 71.889503, 73.529922, 73.529922], 'SMLrmlt': [23.321493, 2.119074, 3.578985, 4.929673, 4.929673, 4.929673, 5.414416, 8.57761, 9.399416, 9.399416, 9.399416, 11.35623, 11.35623, 12.266475, 13.977451, 16.720993, 16.720993, 16.720993, 19.65963, 19.65963, 21.307804, 22.863134, 23.321493, 23.321493], 'SMLrglat': [76.510002, 55.029999, 52.169998, 71.580002, 71.580002, 71.580002, 47.799999, 71.300003, 62.299999, 62.299999, 62.299999, 61.756001, 61.756001, 53.351002, 58.763, 63.75, 63.75, 63.75, 72.300003, 72.300003, 76.769997, 74.5, 76.510002, 76.510002], 'SMLrglon': [25.01, 82.900002, 104.449997, 129.0, 129.0, 129.0, 132.414001, 203.25, 209.800003, 209.800003, 209.800003, 238.770004, 238.770004, 247.026001, 265.920013, 291.480011, 291.480011, 291.480011, 321.700012, 321.700012, 341.369995, 19.200001, 25.01, 25.01], 'SMLrstid': ['HOP', 'NVS', 'IRT', 'TIK', 'TIK', 'TIK', 'BRN', 'BRW', 'T39', 'T39', 'T39', 'FSP', 'FSP', 'C06', 'FCC', 'IQA', 'IQA', 'IQA', 'SUM', 'SUM', 'DMH', 'BJN', 'HOP', 'HOP'], 'SMUr': [1.976003, 31.178246, 31.178246, 31.178246, -0.208882, 0.356117, 0.356117, 0.356117, -1.128606, -1.128606, -0.885122, -0.015059, 6.686568, 6.686568, 6.686568, 8.399819, 8.399819, 17.851616, 17.851616, 17.851616, 19.60713, 23.694275, 23.694275, 23.694275], 'SMUrmlat': [52.904049, 74.702339, 74.702339, 74.702339, 47.791527, 54.29908, 54.29908, 54.29908, 66.244217, 66.244217, 57.76614, 54.597057, 55.715378, 55.715378, 55.715378, 57.829525, 57.829525, 73.989975, 73.989975, 73.989975, 70.473801, 68.194489, 68.194489, 68.194489], 'SMUrmlt': [0.510692, 2.090216, 2.090216, 2.090216, 3.578985, 6.394085, 6.394085, 6.394085, 9.99274, 9.99274, 11.729218, 12.269058, 13.969843, 13.969843, 13.969843, 16.160952, 16.160952, 18.228394, 18.228394, 18.228394, 21.200783, 22.967857, 22.967857, 22.967857], 'SMUrglat': [56.432999, 79.480003, 79.480003, 79.480003, 52.169998, 59.970001, 59.970001, 59.970001, 64.047997, 64.047997, 51.882999, 47.664001, 45.870998, 45.870998, 45.870998, 48.650002, 48.650002, 67.93, 67.93, 67.93, 70.900002, 71.089996, 71.089996, 71.089996], 'SMUrglon': [58.567001, 76.980003, 76.980003, 76.980003, 104.449997, 150.860001, 150.860001, 150.860001, 220.889999, 220.889999, 239.973999, 245.791, 264.916992, 264.916992, 264.916992, 287.549988, 287.549988, 306.429993, 306.429993, 306.429993, 351.299988, 25.790001, 25.790001, 25.790001], 'SMUrstid': ['ARS', 'VIZ', 'VIZ', 'VIZ', 'IRT', 'MGD', 'MGD', 'MGD', 'DAW', 'DAW', 'C13', 'C10', 'C08', 'C08', 'C08', 'T50', 'T50', 'ATU', 'ATU', 'ATU', 'JAN', 'NOR', 'NOR', 'NOR'], 'SMErnum': [5, 3, 3, 4, 5, 6, 6, 4, 8, 9, 12, 13, 20, 17, 17, 11, 12, 14, 12, 14, 22, 51, 51, 35], 'smr': 0.252399, 'smr00': -0.531382, 'smr06': 0.885406, 'smr12': 1.051192, 'smr18': -0.395618, 'smrnum': 72, 'smrnum00': 26, 'smrnum06': 23, 'smrnum12': 6, 'smrnum18': 17, 'bgse': {'X': 1.07, 'Y': -3.75, 'Z': -0.74}, 'bgsm': {'X': 1.07, 'Y': -3.82, 'Z': -0.06}, 'vgse': {'X': -351.100006, 'Y': -5.5, 'Z': -4.0}, 'vgsm': {'X': 351.100006, 'Y': 6.128625, 'Z': -2.947879}, 'clockgse': 258.340698, 'clockgsm': 268.664337, 'density': 5.03, 'dynpres': 1.25, 'epsilon': 29.468521, 'newell': 2504.155029}
    # sample accessing
    print(mydata2c[0]['tval'],mydata2c[0]['density'])    # single element
    result=[ (myeach['tval'],myeach['density']) for myeach in mydata2c] # pull out pairs e.g. 'tval, density')
    # two-line method for extracting any variable set from this
    pairsets= [ (myeach['tval'],myeach['density'],myeach['vgse']) for myeach in mydata2c] # same, pull out pairs, only assign e.g. x=tval, y=density
    tval, density, vgse = [ [z[i] for z in pairsets] for i in (0,1,2)]
    # since 'vgse' is itself an dict of 3 values X/Y/Z, you can pull out nested items like this
    pairsets= [ (myeach['tval'],myeach['density'],myeach['vgse']['X']) for myeach in mydata2c] # same, pull out pairs, only assign e.g. x=tval, y=density
    tval, density, vgse_x = [ [z[i] for z in pairsets] for i in (0,1,2)]
    # the above methods are extensible to any number of variables, just update the (0,1,2) to reflect now many you have
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import supermag

    userid=input("Enter your SuperMAG userid: ")
    time_range = ['2022-11-03T00:00','2022-11-05T00:00']
    # data = SuperMAGGetData(userid,start,3600,'all,baseline=yearly','HBK')
    # print(data)

    sm = supermag.SuperMAG(userid, time_range)
    indices = sm.indices('sml,smr,baseline=yearly')

    print(indices.keys())

    _, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax[0].plot(indices.index, indices.SML)
    ax[1].plot(indices.index, indices.smr)
    ax[0].set(
        ylabel='SML [nT]', 
        ylim=(-1600, 0), 
        title=f'SuperMAG indices: {sm.time_range[0]} to {sm.time_range[1]}'
        )
    ax[1].set(
        ylabel='SMR [nT]', 
        xlabel='Time', 
        ylim=(-80, 0)
        )

    supermag.format_time_axis(ax[-1])
    plt.tight_layout()
    plt.show()