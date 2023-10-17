# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:11:00 2022

@author: Marta Via
"""
#%% Importing libraries
import pandas as pd
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
#%% Importing Spec matrix and organising data
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/"
os.chdir(path)
specs=pd.read_csv('Specs_w_CrCoCdTi.txt', sep='\t', dayfirst=True)
dt_in=pd.to_datetime(specs['date_in'],dayfirst=True)
dt_out=pd.to_datetime(specs['date_out'],dayfirst=True)
dt_hr, dt_in_lr, dt_out_lr=dt_in[:17604], dt_in[17604:], dt_out[17604:]
amus=specs.columns[2:]
amus=pd.Series(amus)
amus_hr, amus_lr=amus[:99], amus[99:]
#%% Sums
hr_sum=specs.iloc[:17604,2:101].sum(axis=1)
lr_sum=specs.iloc[17604:,101:].sum(axis=1)
OA_sum=specs.iloc[:17604,2:95].sum(axis=1)
#%%Repeating LR data towards HR datetimes
PM1_lr=pd.Series([np.nan]*len(dt_hr))
for j in range(0,len(dt_in_lr)):
    for i in range(0,len(dt_hr)):
        if (dt_hr.iloc[i] > dt_in_lr.iloc[j]) and (dt_hr.iloc[i] <= dt_out_lr.iloc[j]):
            # print('hi',dt_hr.iloc[i], dt_in_lr.iloc[j])
            print(i,j)
            PM1_lr.iloc[i]=(lr_sum.iloc[j])
        
            # continue
#%% Plotting for inspection
PM1_lr.plot()
#%% Product calculation
PM1_all=pd.DataFrame()
PM1_all['HR'], PM1_all['LR']=hr_sum, PM1_lr
pm1=PM1_all.sum(axis=1)
pm1.index=dt_hr
pm1.plot()
OA_sum.index=dt_hr
#%% Exportation
pm1.to_csv('PM1_TS_w_CrCoCdTi.txt', sep='\t')
# OA_sum.to_csv('OA_TS.txt', sep='\t')
# amus.to_csv('amus_txt.txt', sep='\t')
# dt_hr.to_csv('dt.txt', sep='\t')
