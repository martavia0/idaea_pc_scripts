# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:09:11 2023

@author: Marta Via
"""

import pandas as pd
import matplotlib.pyplot as plt
import os as os
import glob as glob
#%%
num_f=8
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/Manual DISP/DISP_1_5_50runs/"
os.chdir(path) #changing the path for each combination
all_files = glob.glob(path+'Profile_run_*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
for filename in all_files: 
    df_i=pd.read_csv(filename, sep='\t')
    if df_i.empty or len(df_i)<=2:
        df=pd.concat([df,pd.Series(np.nan)],axis=1)
    else:
        dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        dfi=dfi.astype(float)
        df=pd.concat([df, dfi],axis=1)
df_orig=df.copy(deep=True)
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/"
os.chdir(path) 
mz=pd.read_csv('amus_txt.txt', sep='\t')
#%%
li_factors=[]
for j in range(0,num_f):
    f1=pd.DataFrame()
    for i in range(0+j,len(df.columns), num_f):
        f1=pd.concat([f1,df.iloc[:,i]], axis=1)
        f1['mz']=mz['lab']
    a=f1.set_index('mz', inplace=True)
    li_factors.append(f1.T)
#%%
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
labels=['AS + Heavy oil\n combustion', 'AN + ACl', 'Aged SOA', 'Traffic', 
              'Biomass \nBurning', 'Fresh SOA +\n Road dust',  'COA', 'Industry']
#%%
fig,axs =plt.subplots(nrows=num_f, ncols=2, figsize=(22,15), gridspec_kw={'width_ratios': [1.7, 1]}, sharex='col')
for i in range(0, num_f):
    factor=li_factors[i]
    factor.loc[:,:'100'].boxplot(ax=axs[i,0], showfliers=False, boxprops=boxprops, 
                                 medianprops=medianprops, whiskerprops=whiskerprops)
    
    factor.loc[:,'SO4':].boxplot(ax=axs[i,1], showfliers=False, boxprops=boxprops, 
                                 medianprops=medianprops, whiskerprops=whiskerprops)
    axs[i,1].set_yscale('log')
    axs[i,0].set_ylabel(labels[i], fontsize=18)
    axs[i,0].tick_params(labelrotation=90, labelsize=15)
    axs[i,1].tick_params(labelrotation=90, labelsize=16)
    axs[i,0].set_ylim([0,0.2])
    axs[i,1].set_ylim([0.00000001, 10])
    factor.median().to_csv('factor_'+str(i)+'_medianDISP.txt', sep='\t')
    factor.std().to_csv('factor_'+str(i)+'_stdDISP.txt', sep='\t')
#%%
