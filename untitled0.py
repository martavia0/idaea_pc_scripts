# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:26:52 2023

@author: Marta Via
"""

import pandas as pd
import os as os
import numpy as np
#%%
path="D:/Dades ACSM/ACSM_PalauReial_2022_2023/Treatment"
os.chdir(path)
df1 = pd.read_csv('Waves_raw.txt', sep='\t', dayfirst=True)
df2 = pd.read_csv('Time_desired.txt', sep='\t', dayfirst=True)
df1=df1.drop(df1.index[:20])
#%%
df1['dt']=pd.to_datetime(df1['acsm_utc_time'], dayfirst=True)
df1['dt']=df1['dt'].values.astype('<M8[m]')
df2['dt']=pd.to_datetime(df2['acsm_utc_time_new'], dayfirst=True)
#%%
mask = df1['dt'].isin(df2['dt'])
filtered_df1 = df1[mask]
filtered_df1.to_csv('Reviewed_waves.txt', sep='\t')
