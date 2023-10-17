#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:44:19 2020

@author: martaviagonzalez
"""

import pandas as pd
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
import math
#%%
path = r"C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/ACSM_PalauReial_2021_05_ToF/" # use your path
all_files = glob.glob(path + "/*.txt")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

f = pd.concat(li, axis=0, ignore_index=True)
#%%
""" FOR ONE FILE ONLY
os.chdir("/Users/martaviagonzalez/Documents/IDAEA-CSIC/1. PhD/ToF-ACSM/First data/Native")
f=pd.read_csv("2020_02_22_ACSM_nativeData.txt", sep=",", parse_dates=True, 
              infer_datetime_format=True)
print(f.head())
"""
#%%     
#%%
""" AB and Q corrections are already applied in these .txt files!
    Nevertheless, CE application is set on a constant number of 0.5.
    Here it goes the CE as a function of phase correction """
#NH4 Detection limit definition:    
 
f['Org_CE1'],f['SO4_CE1'],f['NO3_CE1'],f['NH4_CE1'],f['Chl_CE1']= f['Org']*f['CE'],f['SO4']*f['CE'],f['NO3']*f['CE'],f['NH4']*f['CE'],f['Chl']*f['CE']
Org_CE1, SO4_CE1, NO3_CE1, NH4_CE1, Chl_CE1, =f['Org']*f['CE'],f['SO4']*f['CE'],f['NO3']*f['CE'],f['NH4']*f['CE'],f['Chl']*f['CE']

NH4_DL=3.0*math.sqrt(60.0/600.0)*f['NH4'].std()
CE_lowNH4 = 0.5
l=range(0,len(Org_CE1))
CE_dry=pd.Series(index=l)
CE_fPhase=pd.Series(index=l) 

PredNH4_CE1 = 18.0+SO4_CE1/96.0*2.0+NO3_CE1/62.0 + Chl_CE1/35.45
NH4_MeasToPredict = NH4_CE1/PredNH4_CE1
ANMF = (80.0/62.0)*NO3_CE1/(NO3_CE1+SO4_CE1+NH4_CE1+Org_CE1+Chl_CE1)
#ANMF=ANMF.tolist()
for i in range(0,len(NH4_MeasToPredict)):
    if NH4_MeasToPredict.iloc[i]<0.0:
        ANMF.iloc[i]= np.nan
    if ANMF.iloc[i]<0.0:
        ANMF.iloc[i] = np.nan
    elif ANMF.iloc[i]>1.0:
        ANMF.iloc[i] = np.nan
    if PredNH4_CE1.iloc[i]<NH4_DL:
        CE_dry.iloc[i]=CE_low_NH4
    elif NH4_MeasToPredict.iloc[i]>=0.75:
        CE_dry.iloc[i]=0.0833+0.9167*ANMF.iloc[i]
    elif NH4_MeasToPredict.iloc[i]<0.75:
        CE_dry.iloc[i]=1-0.73*NH4_MeasToPredict.iloc[i]
k=0
for j in CE_dry:
    CE_fPhase.iloc[k]=(min(1,max(CE_lowNH4,j)))
    k=k+1
    
f['Org_fP']=f['Org']/CE_fPhase      
f['SO4_fP']=f['SO4']/CE_fPhase      
f['NO3_fP']=f['NO3']/CE_fPhase      
f['NH4_fP']=f['NH4']/CE_fPhase      
f['Chl_fP']=f['Chl']/CE_fPhase      
  
cols_c=['Org','Org_fP', 'SO4', 'SO4_fP']
ax_c = f[cols_c].plot(linewidth=1, alpha=0.5,
        figsize=(9, 6))

#%% 
"""WITH CE AS A FUNCTION OF PHASE APPLIED"""
#=f.sort_values(by=pd.to_datetime(f['Stop_DOY'], unit='D', origin='01-01-2021'))
d = {'Org': f['Org_fP'],'SO4': f['SO4_fP'],'NO3': f['NO3_fP'],'NH4': f['NH4_fP'],'Chl': f['Chl_fP'],'Year': f['Year']}
species=pd.DataFrame(data=d)
species['d_End']=pd.to_datetime(f.Stop_DOY, unit='D', origin='01-01-2021')

#%%
"""WITH CE AS A FUNCTION OF PHASE NOT APPLIED"""
f=f.sort_values(by=['Stop_DOY'])
d = {'Org': f['Org'],'SO4': f['SO4'],'NO3': f['NO3'],
     'NH4': f['NH4'],'Chl': f['Chl'],'Year': f['Year']}
#%%
#%%
species['d_End2']=pd.to_datetime(species['d_End'], unit='D',origin='01-01-2020')
#%%
species['d_Stop']=pd.to_datetime(f.Stop_DOY, unit='D',origin=pd.Timestamp('2020-12-31'))
species=species.set_index('d_Stop')
f['d_Stop']=pd.to_datetime(f.Stop_DOY, unit='D',origin=pd.Timestamp('2020-12-01'))
f=f.set_index('d_Stop') 
species['ind']=range(0,len(species))
#%%
os.chdir( r"C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/ACSM_PalauReial_2021_05_ToF/")
w1=species.to_csv("Tof.txt", sep="\t", decimal=".")
#%%
""" TO FIX """
a=dt.datetime(2020, 3, 2)
aa=pd.to_datetime('20200221', format='%Y%m%d', errors='ignore')
start = species.index.searchsorted((aa))
bb=pd.to_datetime('20200331', format='%Y%m%d', errors='ignore')
#start = species.index.searchsorted(3742)
#start=species.index.get_loc(dt.datetime(2020,3,2))
#species.truncate(before='2020-03-01')
end = species.index.searchsorted(bb)
#end = species.index.searchsorted(3924)
#%%
#species_cut=species[start:end]
species_cut=species[start:end]
species_cut['Org'].plot()
f_cut=f[start:end]
w2=species_cut.to_csv("Tof1.txt", sep="\t", decimal=".")

#%%
""" TIME PLOT ALL SPECIES"""
#fig = plt.figure(figsize = (10,7))
colours=['chartreuse','red','blue','orange','fuchsia']
sns.set(rc={'figure.figsize':(30, 10)})
sns.set_style("whitegrid")
cols_plot = ['Org', 'SO4', 'NO3', "NH4", "Chl"]
axes = species_cut[cols_plot].plot(marker='.', alpha=0.5, linewidth=1,
        figsize=(11, 3), color=colours)
#%%
""" TIME PLOT ALL SPECIES NOT CUT"""
#fig = plt.figure(figsize = (10,7))
colours=['chartreuse','red','blue','orange','fuchsia']
sns.set(rc={'figure.figsize':(30, 10)})
sns.set_style("whitegrid")
cols_plot = ['Org', 'SO4', 'NO3', "NH4", "Chl"]
axes = species[cols_plot].plot(marker='.', alpha=0.5, linewidth=1,
        figsize=(11, 3), color=colours)
#%%
""" INDIVIDUAL TIME PLOTS """
answer = input('Do you want to plot the TS cut? (Yes/No) ')
cols_plot = ['Org', 'SO4', 'NO3', "NH4", "Chl"]
labels_plot= ['OA (μg/m³)','SO4 (μg/m³)','NO3 (μg/m³)', 'NH4 (μg/m³)', 'Chl (μg/m³)']
df=pd.DataFrame()

if answer=="Yes":
    df=species_cut
if answer=="No":
    df=species
axes = df[cols_plot].plot(linewidth=1, alpha=0.5,
                  figsize=(11, 6), subplots=True, color=colours)
i=0
for ax in axes:
    ax.set_ylabel(labels_plot[i])
    i=i+1
#ax.set_xlabel(species.strftime("%m/%d/%Y, %H:%M:%S"))    
#%%    
""" Daily means"""    
hour=pd.DataFrame()
cols_plot = ['Org', 'SO4', 'NO3', "NH4", "Chl"]
hour['Org']=species_cut['Org'].groupby(species_cut.index.hour).mean()
hour['SO4']=species_cut['SO4'].groupby(species_cut.index.hour).mean()
hour['NO3']=species_cut['NO3'].groupby(species_cut.index.hour).mean()
hour['NH4']=species_cut['NH4'].groupby(species_cut.index.hour).mean()
hour['Chl']=species_cut['Chl'].groupby(species_cut.index.hour).mean()
axes2 = hour[cols_plot].plot(linewidth=1, alpha=0.5,
        figsize=(6, 4), subplots=True, color=colours, legend='left')
#more axes style tips on: https://seaborn.pydata.org/tutorial/aesthetics.html
#%%
""" Weekly means"""    
dow=pd.DataFrame()
cols_plot = ['Org', 'SO4', 'NO3', "NH4", "Chl"]
dow['Org']=species_cut['Org'].groupby(species_cut.index.weekday_name).mean()
dow['SO4']=species_cut['SO4'].groupby(species_cut.index.weekday_name).mean()
dow['NO3']=species_cut['NO3'].groupby(species_cut.index.weekday_name).mean()
dow['NH4']=species_cut['NH4'].groupby(species_cut.index.weekday_name).mean()
dow['Chl']=species_cut['Chl'].groupby(species_cut.index.weekday_name).mean()
dow=dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
axes3 = dow[cols_plot].plot(linewidth=1, alpha=0.5,
        figsize=(5, 4), subplots=True, color=colours, legend='left')

#%%
""" Diagnostics plot"""
cols_diag_plot = ['AB_total','Flow_ccs','HB','Lens', 'Press_inlet']
lab = ['AB_total','Flow_ccs','HB','Lens','Press_inlet']
axes3 = f_cut[cols_diag_plot].plot(linewidth=1, alpha=0.5,
        figsize=(11, 6), subplots=True)
i=0
for ax in axes:
    ax.set_ylabel(lab[i])
    i=i+1
#%%
"""Turbo pressure"""
cols_pres_plot = ['Press_inlet','Turbo_speed','Turbo_power','Press_ioniser']
lab = ['Press_inlet','Turbo_speed','Turbo_power','Press_ioniser']
axes3 = f_cut[cols_pres_plot].plot(linewidth=1, alpha=0.5,
        figsize=(11, 6), subplots=True)
i=0
for ax in axes:
    ax.set_ylabel(lab[i])
    i=i+1




