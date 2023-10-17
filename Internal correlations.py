# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:33:40 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Input_Matrices/')
li_sum=[]
#%% 30m
thirty_min=pd.read_csv('30m.txt', sep='\t', dayfirst=True)
thirty_min_hr=thirty_min.iloc[:-83, :80]
del thirty_min_hr['date']
cm_30m=(thirty_min_hr.corr())**2.0
li_sum.append((cm_30m.sum(axis=0)).sum())
#%% 1h
one_h=pd.read_csv('1h.txt', sep='\t', dayfirst=True)
one_h_hr=one_h.iloc[:-83, :80]
del one_h_hr['date']
cm_1h=(one_h_hr.corr())**2.0
li_sum.append((cm_1h.sum(axis=0)).sum())
#%% 2h
two_h=pd.read_csv('2h.txt', sep='\t', dayfirst=True)
two_h_hr=two_h.iloc[:-83, :80]
del two_h_hr['date']
cm_2h=(two_h_hr.corr())**2.0
li_sum.append((cm_2h.sum(axis=0)).sum())
#%% 3h
three_h=pd.read_csv('3h.txt', sep='\t', dayfirst=True)
three_h_hr=three_h.iloc[:-83, :80]
del three_h_hr['date']
cm_3h=(three_h_hr.corr())**2.0
li_sum.append((cm_3h.sum(axis=0)).sum())
#%% 6h
six_h=pd.read_csv('6h.txt', sep='\t', dayfirst=True)
six_h_hr=six_h.iloc[:-83, :80]
del six_h_hr['date']
cm_6h=(six_h_hr.corr())**2.0
li_sum.append((cm_6h.sum(axis=0)).sum())
#%% 12h
twelve_h=pd.read_csv('12h.txt', sep='\t', dayfirst=True)
twelve_h_hr=twelve_h.iloc[:-83, :80]
del twelve_h_hr['date']
cm_12h=(twelve_h_hr.corr())**2.0
li_sum.append((cm_12h.sum(axis=0)).sum())
#%% 24h
twfour_h=pd.read_csv('24h.txt', sep='\t', dayfirst=True)
twfour_h_hr=twfour_h.iloc[:-83, :80]
del twfour_h_hr['date']
cm_24h=(twfour_h_hr.corr())**2.0
li_sum.append((cm_24h.sum(axis=0)).sum())
#%%
corr_r1=pd.Series(li_sum)
corr_r1.index=['30m', '1h', '2h','3h','6h','12h','24h']
fig, axs=plt.subplots(figsize=(4,4))
corr_r1.plot(marker='o', color='grey', grid=True, ax=axs)
axs.set_ylabel('Sum of internal correlations (adim.)')
axs.set_xlabel('HR dataset resolution')
#%%
ic=pd.DataFrame({'30 m': cm_30m.sum(axis=1), '1 h': cm_1h.sum(axis=1), '2 h': cm_2h.sum(axis=1), '3 h': cm_3h.sum(axis=1),
                 '6 h': cm_6h.sum(axis=1),'12 h': cm_12h.sum(axis=1), '24 h': cm_24h.sum(axis=1),})
fig, axs=plt.subplots(nrows=7)
for i in range(0,7):
    axs[i].bar(ic.index, ic.iloc[:,i], color='grey')
#%%
fig, axs=plt.subplots()
ic_T=ic.T
ic_T.plot(ax=axs, legend=False)
#%%
fig, axs=plt.subplots(figsize=(15,3))
ic_std=ic.std(axis=1)/ic.mean(axis=1)
ic_std.iloc[1:].plot.bar(color='grey', ax=axs)
axs.set_ylabel('Standard dev. / mean')
axs.set_xlabel('Species')
# axs.bar(ic.index, ic, color='grey')

















