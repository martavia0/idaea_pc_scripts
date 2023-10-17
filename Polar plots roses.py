# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:50:39 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
# import seaborn as sns
import matplotlib.pyplot as plt
# import windrose as wr
#%% Importing HR data
path='C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/HR only/Output/Plots/Solution/'
os.chdir(path)
df=pd.read_csv('HR_meteo.txt', sep='\t', dayfirst=True)
df['dt']=pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
dt=pd.to_datetime(df.dt,errors='coerce')
#%%
path='C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Solutions_20220909_w_CrCdCoTi/SA/'
os.chdir(path)
df=pd.read_csv('TS_8F_meteo.txt', sep='\t', dayfirst=True)
df['dt']=pd.to_datetime(df['d'], dayfirst=True, errors='coerce')
dt=pd.to_datetime(df.dt,errors='coerce')
# %% Windrose only
from windrose import WindroseAxes
plt.rcParams.update({'font.size': 22})
ax = WindroseAxes.from_ax()
ax.bar(df['wd'], df['ws'], normed=True, opening=0.8, edgecolor='white')
ax.legend(fontsize=12)

#%% WINDROSE
ax = WindroseAxes.from_ax()
ax.contourf(df['wd'], df['ws'], bins=np.arange(0, 8, 1))
ax.legend(fontsize=12)

print(df.mean())
#%% POLLUTION ROSES
factor=df.columns[8]
theta = np.linspace(0,360,17)
r=np.linspace(0,10,11)
li_theta=[]
for i in range(0,len(theta)-1):
    mask=(df['wd']>= theta[i]) &  (df['wd']<theta[i+1])
    df1=df[mask]
    li_r=[]
    for j in range(0,len(r)-1):
        mask2=(df1['ws']>r[j])&(df1['ws']<r[j+1])
        df2=df1[mask2]
        li_r.append(df2[factor].mean())
    li_theta.append(li_r)
theta_rad=(theta*np.pi/180.0).round(5)
poll=pd.DataFrame(li_theta)#, columns=r[:-1],index=theta_rad[:-1])   

fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(8,8))
R, Theta=np.meshgrid(r,theta*np.pi/180.0)
plot=ax.pcolormesh(Theta, R, poll, cmap='Greys', vmin=-1, vmax=2.5)
# ax.contourf(Theta, R,poll)
# ax.set_xticks(np.pi/180. * np.linspace(0,360, 16 , endpoint=False))
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")  # theta=0 at the top
ax.set_rticks(r)  # Less radial ticks
ax.grid()
ax.set_title(factor+' $(\mu g·m^{-3})$', fontsize=25)
cb=fig.colorbar(plot, ax=ax, orientation='horizontal')
# cb.set_clim(0,2.5)
#%%






































