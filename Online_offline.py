# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:38:44 2023

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
path_py="C:/Users/maria/Documents/Marta Via/1. PhD/F. Scripts/Python Scripts"
os.chdir(path_py)
import sys
# sys.path.insert(0,“..”)
from Treatment import *
#%%
zero=0
trt = Treatment_class(zero)
trt.Hello()
print(trt.x)

#%% Set path
path_online='C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/NRT_PR/2023_06/SoFi/SA/'
path_offline='C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/NRT_PR/2023_06/Comp/'
os.chdir(path_online)
#%% Import and concatenate all SA results.
all_files = glob.glob(path_online+'Palau*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
for filename in all_files: #We import all runs for the given combination
    dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    df=pd.concat([df, dfi],axis=0)
df=df.iloc[:,:6]
df.columns=['date', 'HOA', 'COA', 'BBOA', 'Amine_OA', 'OOA']
df['datetime']=pd.to_datetime(df['date'], dayfirst=True)
del df['date']
#df=df.reset_index(df['datetime'], inplace=True)
#%%
df_on=trt.SelectDates(df, '01/06/2023', '01/08/2023')
df_on['datetime']=df_on['datetime'].dt.floor('Min')
#%% Import offline data
os.chdir(path_offline)
df_off=pd.read_csv('TS_factors.txt', sep='\t',keep_default_na=True,na_values='np.nan')
df_off['datetime']=pd.to_datetime(df_off['date'], dayfirst=True)

#%% Retouches to homogenise both
df_offline =trt.Data_Matcher(df_on, df_off, 'datetime', 'datetime')    
df_online = df_on.copy( deep=True)
df_online.columns=['HOA', 'COA', 'BBOA', 'Amine-OA','OOA', 'datetime']
df_offline['OOA'] = df_offline['LO']+df_offline['MO']
df_offline.columns=['date', 'HOA', 'COA', 'Amine-OA', 'LO-OOA', 'MO-OOA','OOA', 'datetime']
df_online.index=range(0,len(df_online))
#%% Plotting Scatterplot
os.chdir(path_offline)
factor='OOA'
colors=pd.DataFrame(data={'HOA':['grey'], 'COA': ['darkkhaki'], 'Amine-OA': ['skyblue'],  'OOA': ['darkgreen']})
fig, ax=plt.subplots(figsize=(4,4))
ax.scatter(x=df_offline[factor], y=df_online[factor], color=colors[factor])
ax.set_ylabel('Online (μg·m$^{-3}$)', fontsize=16)
ax.set_xlabel('Offline (μg·m$^{-3}$)', fontsize=16)
x_reg=x=list(np.arange(0.0,max(df_offline[factor]),0.05))
y_reg=[i * slope(df_online[factor], df_offline[factor])[0] + slope(df_online[factor], df_offline[factor])[1] for i in x_reg]
ax.plot(x_reg, y_reg, color='k')
ax.text(x=0,y=max(df_online[factor])-0.1, s= 'y='+str(slope(df_online[factor], df_offline[factor])[0])+'·x+'+ str(slope(df_online[factor], df_offline[factor])[1])+
        '\nR$^2$='+str(R2(df_online[factor], df_offline[factor])))
plt.title('April 2023')
fig.suptitle(factor, fontsize=20, y=1.02)
plt.savefig(factor+'.jpeg', bbox_inches='tight')
#%%Plotting TS
os.chdir(path_offline+'/Comp_offline_online/')
factor='OOA'
colors=pd.DataFrame(data={'HOA':['grey'], 'COA': ['darkkhaki'], 'Amine-OA': ['skyblue'], 'LO-OOA': ['chartreuse'], 'MO-OOA': ['darkgreen']})
fig, ax=plt.subplots(figsize=(10,2))
df_offline=df_offline.set_index(df_offline['datetime'])
df_online=df_online.set_index(df_online['datetime'])
ax.plot(df_online[factor], color='k')
ax.plot(df_offline[factor], color='darkgreen')
ax.legend(['Online', 'Offline'])
ax.set_ylabel(factor+' (μg·m$^{-3}$)', fontsize=12)
plt.savefig(factor+'_TS.jpeg', bbox_inches='tight')

#%%
df_online.index.is_unique
df_online.index.duplicated()
# df_offline.loc[:, ~df.columns.duplicated()]
#%%
# df_offline[factor+'_off']=df_offline[factor]
# c=pd.DataFrame({'a':df_online[factor], 'b': df_offline['HOA_off']})

print(type(slope(df_offline[factor+'_off'], df_online[factor])[0]))


