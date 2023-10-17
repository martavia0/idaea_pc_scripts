# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:30:25 2022

@author: Marta Via
"""

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
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Input_Matrices/"
os.chdir(path)
specs=pd.read_csv('30m.txt', sep='\t', dayfirst=True)
error=pd.read_csv('30m_errors.txt', sep='\t', dayfirst=True, header=None)
date=pd.read_csv('30m_date.txt', sep='\t', dayfirst=True)
amus=pd.read_csv('amus.txt',sep='\t', header=None)
dt_in=pd.to_datetime(date['date_in'],dayfirst=True)
dt_out=pd.to_datetime(date['date_out'],dayfirst=True)
dt_hr, dt_in_lr, dt_out_lr=dt_in[:17604], dt_in[17604:], dt_out[17604:]
amus_hr, amus_lr=amus[:78], amus[78:]
del specs['date']
#%% Sums
hr_sum=specs.iloc[:17604,1:79]#.sum(axis=1)
lr_sum=specs.iloc[17604:,79:]#.sum(axis=1)
OA_sum=specs.iloc[:17604,2:72]#.sum(axis=1)
#%%Repeating LR data towards HR datetimes
specs_kd_li,errors_kd_li,date_kd_li=[],[],[]
for j in range(0,len(dt_in_lr)):
    print(j)
    for i in range(0,len(dt_hr)):
        if (dt_hr.iloc[i] > dt_in_lr.iloc[j]) and (dt_hr.iloc[i] <= dt_out_lr.iloc[j]):
            # print('hi',dt_hr.iloc[i], dt_in_lr.iloc[j])
            #print(i,j)
            specs_kd_li.append(specs.iloc[i])
            errors_kd_li.append(error.iloc[i])
            date_kd_li.append(date.iloc[i])
            # continue
#%% into dataframes
specs_kd=pd.DataFrame(specs_kd_li)
errors_kd=pd.DataFrame(errors_kd_li)
date_kd=pd.DataFrame(date_kd_li)
#%% Exportation
specs_kd.to_csv('30m_Specs_KD.txt', sep='\t')
errors_kd.to_csv('30m_Errors_KD.txt', sep='\t')
date_kd.to_csv('30m_Date_KD.txt', sep='\t')
