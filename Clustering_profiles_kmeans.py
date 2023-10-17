# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:16:00 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/Manual DISP/DISP_100_runs/"
os.chdir(path) #changing the path for each combination

all_files = glob.glob(path+'Profiles_run_*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
for filename in all_files: #We import all runs for the given combination
    dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    dfi=dfi.astype(float)
    df=pd.concat([df, dfi],axis=1)
df_orig=df.copy(deep=True)
path_orig="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/Manual DISP/"
os.chdir(path_orig) #changing the path for each combination
df_orig=pd.read_csv('Profiles_Def.txt', sep='\t')
mz=pd.DataFrame({'lab':df_orig['lab']})
del df_orig['lab']
#%%
from scipy.stats import linregress
from scipy import stats
def R2(a, b):
    c = pd.DataFrame({"a": a, "b": b})
    cm = c.corr(method='pearson')
    r = cm.iloc[0, 1]
    return (r**2).round(2)
def slope(b, a):
    c = pd.DataFrame({"a": a, "b": b})
    mask = ~np.isnan(a) & ~np.isnan(b)
    a1 = a[mask]
    b1 = b[mask]
    if (a1.empty) or (b1.empty):
        s = np.nan
    else:
        s, intercept, r_value, p_value, std_err = linregress(a1, b1)
    return s.round(2), intercept.round(2)
#%%
nf=8
nb_runs=len(all_files)
std_df= df.std(axis=1)
li_R2, li_slope, li_dif, li_reldif=[],[],[],[]
for j in range(0,nf):
    li_R2_i,li_slope_i, li_dif_i, li_reldif_i=[],[],[],[]
    subset=df[[j]]
    std_df= subset.std(axis=1)
    for i in range(0,nb_runs):
        li_R2_i.append(R2(df_orig.iloc[:,j], subset.iloc[:,i]))
        li_slope_i.append(slope(df_orig.iloc[:,j], subset.iloc[:,i])[0])
        li_dif_i.append((df_orig.iloc[:,j]-subset.iloc[:,i]))#/std_df
        li_reldif_i.append((df_orig.iloc[:,j]-subset.iloc[:,i])/df_orig.iloc[:,j])#/std_df
    li_R2.append(li_R2_i)
    li_slope.append(li_slope_i)
    li_dif.append(li_dif_i)
    li_reldif.append(li_reldif_i)

#%%    
R2=pd.DataFrame(li_R2)
slopes=pd.DataFrame(li_slope)
#%%
li_dif_2=[]
li_reldif_2=[]
for k,l in zip(li_dif, li_reldif):
    li_dif_2.append(pd.DataFrame(k).mean())
    li_reldif_2.append(pd.DataFrame(l).mean())
dif=pd.DataFrame(li_dif_2).T
reldif=pd.DataFrame(li_reldif_2).T

#%%
Profiles_std(df_orig, dif, dif, 'hola',8)

#%%
def Profiles_std(df,df_std, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col',gridspec_kw={'width_ratios': [3, 2]})
        for i in range(0,nf):
            org_prop=str(100.0*((df.iloc[:,i][:73].sum()/df.iloc[:,i].sum())).round(1))
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            axes[i,0].errorbar(mz['lab'][:73], df.iloc[:,i][:73], yerr=df_std.iloc[:,i][:73],  fmt='.', color='k')#, uplims=True, lolims=True)
            # ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            # ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,0].text(x=65, y=max(df.iloc[:,i][:73])-0.1*max(df.iloc[:,i][:73]), s='OA(%) = '+org_prop+'%', fontsize=20)
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].errorbar(mz['lab'][73:], df.iloc[:,i][73:], yerr=df_std.iloc[:,i][73:],  fmt='.', color='k')#, uplims=True, lolims=True)
            axes[i,1].set_yscale('log')
            # ax4=axes[i,1].twinx()
            # ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            axes[i,1].grid(axis='x')
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            # ax4.tick_params(labelrotation=0, labelsize=16)
            # ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)
            axes[i,0].set_ylabel(df_orig.columns[i], fontsize=17)
            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)    
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/Manual DISP/DISP_100_runs/"
os.chdir(path) #changing the path for each combination

all_files_ts = glob.glob(path+'TS_run_*') #we import all files which start by Res in the basecase folder (10 runs)
ts=pd.DataFrame()
for filename in all_files_ts: #We import all runs for the given combination
    tsi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    tsi=tsi.astype(float)
    ts=pd.concat([ts,tsi],axis=1)
path_orig="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/BS_DISP/Manual DISP/"
os.chdir(path_orig) #changing the path for each combination
ts_orig=pd.read_csv('Def_TS.txt', sep='\t')
del ts_orig['Unnamed: 0']
#%%
nf=8
nb_runs=len(all_files_ts)
std_ts= ts.std(axis=1)
li_R2, li_slope, li_dif, li_reldif, li_std=[],[],[],[],[]
for j in range(0,nf):
    li_R2_i,li_slope_i, li_dif_i, li_reldif_i=[],[],[],[]
    subset=ts[[j]]
    std_ts= subset.std(axis=1)
    for i in range(0,nb_runs):
        li_R2_i.append(R2(ts_orig.iloc[:,j], subset.iloc[:,i]))
        li_slope_i.append(slope(ts_orig.iloc[:,j], subset.iloc[:,i])[0])
        li_dif_i.append((ts_orig.iloc[:,j]-subset.iloc[:,i]))#/std_ts
        li_reldif_i.append((ts_orig.iloc[:,j]-subset.iloc[:,i])/ts_orig.iloc[:,j])#/std_ts
    li_R2.append(li_R2_i)
    li_slope.append(li_slope_i)
    li_dif.append(li_dif_i)
    li_reldif.append(li_reldif_i)
    li_std.append(std_ts)
#%%
R2=pd.DataFrame(li_R2)
slopes=pd.DataFrame(li_slope)
std=pd.DataFrame(li_std).T
#%%
li_dif_2, li_dif_m, li_dif_M=[],[],[]
li_reldif_2=[]
for k,l in zip(li_dif, li_reldif):
    li_dif_2.append(pd.DataFrame(k).mean())
    li_dif_m.append(pd.DataFrame(k).min())
    li_dif_M.append(pd.DataFrame(k).max())
    li_reldif_2.append(pd.DataFrame(l).mean())
dif=pd.DataFrame(li_dif_2).T
dif_m=pd.DataFrame(li_dif_m).T
dif_M=pd.DataFrame(li_dif_M).T
reldif=pd.DataFrame(li_reldif_2).T
#%%
date=pd.read_csv('date.txt', sep="\t")
dt=pd.to_datetime(date['d'],dayfirst=True)
#%%
ts_orig['std_0']=std.iloc[:,0]
#%%
fig, axs=plt.subplots(nf,1, figsize=(10,10), sharex=True)
for i in range(0,nf):
    axs[i].plot(dt, ts_orig.iloc[:,i], color='k')
    axs[i].fill_between(x=dt, y1=ts_orig[:,i]-std[i], y2=ts_orig[:,i]+std[i], interpolate=True, color="grey", alpha=0.4)
    axs[i].set_ylabel(ts_orig.columns[i])
#%%
dif_m.plot()