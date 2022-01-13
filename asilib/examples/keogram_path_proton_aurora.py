# Make a keogram of a proton aurora observed by The Pas along a custom path 
# (the Gillam meridional scanning photometer).
import pandas as pd
import matplotlib.pyplot as plt

import asilib

msp_url = ('https://github.com/mshumko/aurora-asi-lib/raw/'
    '73ef9bd5220b781436aea3281c70da0f5b08ac05/asilib/data/GILL_MSP_coords.csv')
msp_df = pd.read_csv(msp_url)
print(msp_df.head())