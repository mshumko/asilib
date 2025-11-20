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
        Download a list of SuperMAG magnetometer location code names for a given time range.

        Returns
        -------
        stations : list
            A list of SuperMAG magnetometer location code names.

        Example
        -------
        >>> import clowncar
        >>> 
        >>> userid=input("Enter your SuperMAG userid: ")
        >>> time_range = ['2022-11-04T06:40','2022-11-04T07:20']
        >>> sm = clowncar.SuperMAG(userid, time_range)
        >>> print(sm.location_codes())
        """

        # construct URL             
        urlstr = self._base_url('inventory.php')

        # get the string array of stations
        stations = self._get_url_data(urlstr,'raw')

        # first data item is how many stations were found
        numstations = int(stations[0])
        if numstations > 0: 
            return stations[1:-1]
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
        >>> import clowncar
        >>>
        >>> userid=input("Enter your SuperMAG userid: ")
        >>> time_range = ['2022-11-03T00:00','2022-11-05T00:00']
        >>> 
        >>> sm = clowncar.SuperMAG(userid, time_range)
        >>> indices = sm.indices('sml,smr,baseline=yearly')
        >>> print(indices.keys())  # Index(['SML', 'smr'], dtype='object')
        >>> 
        >>> _, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        >>> ax[0].plot(indices.index, indices.SML)
        >>> ax[1].plot(indices.index, indices.smr)
        >>> ax[0].set(
        >>>     ylabel='SML [nT]', 
        >>>     title=f'SuperMAG indices: {sm.time_range[0]} to {sm.time_range[1]}'
        >>>     )
        >>> ax[1].set(
        >>>     ylabel='SMR [nT]', 
        >>>     xlabel='Time',
        >>>     )
        >>> 
        >>> clowncar.supermag.format_time_axis(ax[-1])
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


    def mag_data(self, station, query_parameters=''):
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

        Example
        -------
        >>> import clowncar
        >>> userid=input("Enter your SuperMAG userid: ")
        Enter your SuperMAG userid: ...
        >>> time_range = ['2022-11-04T06:40','2022-11-04T07:20']
        >>> sm = clowncar.SuperMAG(userid, time_range)
        >>> df = sm.mag_data('HBK')
        >>> df.head()
                            ext iaga      mag_n      mag_e     mag_z      geo_n      geo_e     geo_z
        tval                                                                                          
        2022-11-04 06:40:00  60.0  HBK -20.106401  25.399405 -2.382750 -10.571674  30.620856 -2.382750
        2022-11-04 06:41:00  60.0  HBK -19.933125  24.740351 -1.831235 -10.626166  29.941593 -1.831235
        2022-11-04 06:42:00  60.0  HBK -19.469585  24.020023 -1.568676 -10.427026  29.108477 -1.568676
        2022-11-04 06:43:00  60.0  HBK -18.896301  23.956665 -1.867642  -9.906982  28.859032 -1.867642
        2022-11-04 06:44:00  60.0  HBK -18.863857  23.847208 -1.805935  -9.912575  28.745005 -1.805935
        """        
        urlstr = self._base_url('data-api.php')
        indices = self._sm_keycheck_data(query_parameters)
        urlstr += indices
        urlstr += '&station='+station.upper()

        data_list = self._get_url_data(urlstr,'json')

        data_df = pd.DataFrame(data_list)
        data_df.index = pd.to_datetime(data_df['tval'],unit='s')
        data_df.drop(columns=['tval'],inplace=True)
        data_df = self._flatten_nested_columns(data_df)
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
        myflags = myflags.replace(',&','&')  
        myflags = myflags.replace(',$','')
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
    
    def _flatten_nested_columns(self, df):
        """
        Flatten the columns with dictionary vector compoonents into separate columns.
        """
        for column in ['N', 'E', 'Z']:
            df[f'mag_{column.lower()}'] = df[column].apply(lambda x: x['nez'])
            df[f'geo_{column.lower()}'] = df[column].apply(lambda x: x['geo'])
            df.drop(columns=[column], inplace=True)

        # The above for-loop scrambles the column order, so we reorder them here.
        other_keys = [column for column in df.columns if ('mag_' not in column) and ('geo_' not in column)]
        mag_keys = [column for column in df.columns if ('mag_' in column)]
        geo_keys = [column for column in df.columns if ('geo_' in column)]
        return df[other_keys + mag_keys + geo_keys]


def format_time_axis(ax):
    """
    Format the x-axis of a time series plot to include date only on the first tick and at midnight.
    Include the HH:MM on all ticks.
    """
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_xaxis))
    return

def _format_xaxis(tick_val, tick_pos):
    """
    The tick magic that is a matplotlib callback.
    """
    tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
    if (tick_pos==0) or ((tick_time.hour == 0) and (tick_time.minute == 0)):
        ticks = tick_time.strftime('%H:%M') + '\n' + tick_time.strftime('%Y-%m-%d')
    else:
        ticks = tick_time.strftime('%H:%M')
    return ticks