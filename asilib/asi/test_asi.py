"""
An ASI for testing asilib.Imager.
"""
import pandas as pd
import numpy as np

def asi(location, time, time_range):


    return

def asi_info()->pd.DataFrame:
    """
    The test ASI has three locations, a, b, and c.
    """
    locations = pd.DataFrame(data=np.array([
        ['GILL', 56.3494, -94.7056, 500],
        ['ATHA', 54.7213, -113.285, 1000],
        ['TPAS', 53.8255, -101.2427, 0],
        ]), columns=['name', 'lat', 'lon', 'alt'])
    return locations

if __name__ == '__main__':
    print(asi_info())