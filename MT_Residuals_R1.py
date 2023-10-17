# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:32:34 2022

@author: Marta Via
"""
#%% Importing libraries
import pandas as pd
import os as os
import datetime as dt
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
#%% This cell imports the scaled residuals of all runs, concatenates them and separates them into HR, Lr
''' By Cs '''
run_name='1h_Sweep'#'30m_KD_Sweep'
n_HR=7070 #7070 for 1h, 3623 for 30m_KD
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/' 
os.chdir(path)
all_files = glob.glob(path+'Res_run*') 
all_files.sort()
df, hr, lr=pd.DataFrame(),pd.DataFrame(), pd.DataFrame()
for filename in all_files:
    dfi=pd.read_csv(filename, sep='\t',header=0)
    if dfi.empty or len(dfi.columns)==2:
        df=pd.concat([df, pd.Series(np.nan, name='2', index=range(0,n_HR))],axis=1)
        hr=pd.concat([hr, pd.Series(np.nan, name='2')],axis=1)
        lr=pd.concat([lr,pd.Series(np.nan, name='2')],axis=1)
        continue
    else:
        dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        dfi=dfi.astype(float)
        dfi=dfi.reset_index(drop=True)
        df=pd.concat([df, dfi.iloc[:,2]],axis=1)
        hr=pd.concat([hr, dfi.iloc[:n_HR,2]],axis=1)
        lr=pd.concat([lr, dfi.iloc[n_HR:,2]],axis=1)
df_orig=df.copy(deep=True)
#%% We concat by number of factors. 
n3=int(len(df.columns)/3)
df.columns, hr.columns, lr.columns =range(0,len(df.columns)),range(0,len(df.columns)),range(0,len(df.columns))
#%%
df_c=pd.concat([df.iloc[:,5:20], df.iloc[:,15:20]], ignore_index=True)
#%%
df_c=pd.concat([df.iloc[:,0:140], df.iloc[:,140:280], df.iloc[:, 280:]], ignore_index=True)
hr_c=pd.concat([hr.iloc[:,0:n3], hr.iloc[:,n3:n3*2], hr.iloc[:, n3*2:]])
lr_c=pd.concat([lr.iloc[:,0:n3], lr.iloc[:,n3:n3*2], lr.iloc[:, n3*2:]])
#%% We separate by C values used
# c=['1, 1', '0.1, 1', '1, 0.1', '10, 1', '1, 10'] for KD
c=['1000,1', '100,1', '10,1', '1,1', '0.1, 1', '0.01, 1', '0.001, 1', '1, 1000', '1, 100', '1, 10', '1,1', '1, 0.1', '1,0.01', '1, 0.001']
df_cs,df_cs_i,hr_cs,hr_cs_i,lr_cs,lr_cs_i=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
for j in range(0,len(c)):
    for i in range(0,10):
        df_cs_i=pd.concat([df_c.iloc[:,j*10+i]])
        hr_cs_i=pd.concat([hr_c.iloc[:,j*10+i]])
        lr_cs_i=pd.concat([lr_c.iloc[:,j*10+i]])
    df_cs=pd.concat([df_cs, df_cs_i], axis=1)
    hr_cs=pd.concat([hr_cs, hr_cs_i], axis=1)
    lr_cs=pd.concat([lr_cs, lr_cs_i], axis=1)
df_cs.columns, hr_cs.columns, lr_cs.columns=c,c,c
#%%
df_cs.plot()
#%%
for i in range(0,len(c)):
    cs.append(hr_cs.iloc[:,i])
    cs.append(lr_cs.iloc[:,i])
# cs=[hr_cs.iloc[:,0].reset_index(),lr_cs.iloc[:,0].reset_index(), hr_cs.iloc[:,1].reset_index(), lr_cs.iloc[:,1].reset_index(),
#     hr_cs.iloc[:,2].reset_index(), lr_cs.iloc[:,2].reset_index(),hr_cs.iloc[:,3].reset_index(),lr_cs.iloc[:,3].reset_index(),
#     hr_cs.iloc[:,4].reset_index(),lr_cs.iloc[:,4].reset_index()]
df_cs=pd.concat(cs, axis=1, ignore_index=True)
for i in list(range(0,20,2)):
    del df_cs[i]
#%%
c_hrlr=[]
for i in c:
    c_hrlr.append(i + ': HR')
    c_hrlr.append(i + ': LR')
df_cs.columns=c_hrlr
#%% Plotting
fig, axs=plt.subplots(figsize=(10,10))
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
df_cs.boxplot(ax=axs, showfliers=False, boxprops=boxprops, medianprops=medianprops, showmeans=True, meanprops=meanprops, whiskerprops=whiskerprops)
# axs.set_yscale('log')
# axs.set_ylim(0.1,10000000)
whiskerprops = dict(linestyle='-', linewidth=1, color='k')

#%%
lr.iloc[:,1].hist()













    
    
    