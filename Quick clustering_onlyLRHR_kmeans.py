# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:07:52 2022

@author: Marta Via
"""
#%%

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
run_name='LR only'
num_f=7
nf=3 #3 for filters, 5 for HR
# path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/Results/"
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/LR/Solutions/F_20220930/"
os.chdir(path) #changing the path for each combination
all_files = glob.glob(path+'Profile_run_'+str(num_f-nf)+'*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
for filename in all_files: #We import all runs for the given combination
    dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    dfi=dfi.astype(float)
    df=pd.concat([df, dfi],axis=1)
df_orig=df.copy(deep=True)
relf=pd.DataFrame()
ts=pd.DataFrame()
namef=os.listdir(path)
all_rel_files = glob.glob(path+'Rel_profile_run_'+str(num_f-nf)+'*') #4 if filters, 5 if HR
for filename in all_rel_files:
    rel_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    rel_i=rel_i.astype(float)
    relf=pd.concat([relf, rel_i],axis=1)
relf_orig=relf.copy(deep=True)
namef=os.listdir(path)
all_ts_files = glob.glob(path+'TS_run_'+str(num_f-nf)+'*') #4 if filters, 5 if HR
for filename in all_ts_files:
    ts_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    ts_i=ts_i.astype(float)
    ts=pd.concat([ts, ts_i],axis=1)
ts_orig=ts.copy(deep=True)
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_res_files = glob.glob(path+'Res_run_'+str(num_f-nf)+'*') #4 if filters, 5 if HR
res=pd.DataFrame()
for filename in all_res_files:
    res_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    res_i=res_i.iloc[:,2].astype(float)
    res=pd.concat([res, res_i],axis=1)
res_orig=res.copy(deep=True)
#%% Clustering! k-means
from scipy.cluster.vq import vq, kmeans, whiten
data=df.T
centroids, clusters = kmeans(data, num_f)
codebook=centroids
clx,_ = vq(data,centroids)
print(len(centroids), len(centroids[0]), clusters)
centroids=pd.DataFrame(centroids).T
centroids/=centroids.sum(axis=1)
centroids.plot.bar(legend=False,subplots=True)
#%%
df2=df.T
df2['Cluster']=clx
df_std=df2.groupby(by= ['Cluster']).std().T
#%%
whitened=np.array(data)
plt.scatter(whitened[:, 17], whitened[:, 18])
plt.scatter(codebook[:, 17], codebook[:, 18], c='r')
#%% Calculation of stars (rel prof)
rel_f2=relf.T
rel_f2['Cluster']=clx
rel_f3=rel_f2.groupby(by=['Cluster']).mean().T
# rel_f3/=rel_f3.sum(axis=1)
rel_f3.iloc[:,:] = rel_f3.iloc[:,:].div(rel_f3.sum(axis=1), axis=0)

#%% Calculation of time series (TS)
ts_f2=ts
ts_f3=ts_f2.T
ts_f3['Cluster']=clx
ts_f3_std=ts_f3.groupby(by=['Cluster']).std().T
ts_f3=ts_f3.groupby(by=['Cluster']).mean().T
ts_f3.plot(subplots=True)

#%%Capturing TS and mzs
if run_name=='LR only':
    os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/")
    amus_txt=pd.read_csv("amus_F.txt", sep='\t', header=0)
    mz=amus_txt
    dt_F=pd.read_csv('dt_F.txt', sep='\t', dayfirst=True)
    date=pd.to_datetime(dt_F['date_out'], dayfirst=True)
if run_name=='HR only':
    os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/")
    dt=pd.read_csv("datetime.txt", sep='\t', dayfirst=True )
    date=pd.to_datetime(dt['date'],dayfirst=True)
    pm1=pd.read_csv('PM1.txt', sep='\t', dayfirst=True)
#%%
if run_name=='LR only':
    Profiles_F(centroids, rel_f3, str(num_f), num_f)
if run_name=='HR only':
    Profiles_OA(centroids, rel_f3, str(num_f), num_f)
#%%
df_TS=ts_f3.copy(deep=True)
df_TS['d']=date
df_TS.set_index('d', inplace=True)
#%%
pie=df_TS.mean(axis=0)
print(pie)
pie_relf = pd.Series([i/pie.sum() for i in pie])
labels_pie=['P'+str(i) for i in range(1,len(pie)+1)]
# labels_pie=['Aged SOA + AS'+'\n', 'Heavy oil combustion', 'Biomass burning + fresh SOA'+'\n', 'Traffic', 'Industry'+'\n', 
#             'Fresh SOA +AN', 'Cd outbreak + Aged Aerosol'+'\n']
pie_relf.plot.pie(autopct='%1.1f%%', fontsize=10, labels=labels_pie,colors=['white', 'gainsboro','lightgrey',  'silver', 'dimgray', 'slategrey', '#424242'])#'white','gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey', '#424242'])#'white',#,'slategrey'])
plt.ylabel("")
df_TS.plot(subplots=True, figsize=(10,15),  grid=True, legend=False, title=labels_pie, color='grey',lw=3, fontsize=14)#marker='o'
# pd.DataFrame(df_TS).to_csv('TS_5F_Results.txt', sep='\t')
#%%
res_mean=res.mean(axis=1)
#%%Exporting
if run_name=='HR only':
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/Plots/')
if run_name=='LR only':
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/LR/Solutions/F_20220930/SA/')
    
centroids.to_csv('Profiles_'+str(num_f)+'.txt', sep='\t')
rel_f3.to_csv('RelProf_'+str(num_f)+'.txt', sep='\t')
df_TS.to_csv('TS_'+str(num_f)+'.txt', sep='\t')
res_mean.to_csv('Res_'+str(num_f)+'.txt', sep='\t')
ts_f3_std.to_csv('TS_std_'+str(num_f)+'.txt', sep='\t')
df_std.to_csv('PR_std_'+str(num_f)+'.txt', sep='\t')

#%% Monthly
df_TS['d']=df_TS.index
df_TS['Month']=df_TS['d'].dt.month
Month=df_TS.groupby(df_TS['Month']).mean()
Month.plot(subplots=True, figsize=(10,10), marker='o', color='grey', legend=False, grid=True)
del df_TS['Month']
#%%Diel
df_TS['d']=df_TS.index
df_TS['Hour']=df_TS['d'].dt.hour
Month=df_TS.groupby(df_TS['Hour']).mean()
Month.plot(subplots=True, figsize=(6,10), marker='o', color='grey', legend=False, grid=True)
del df_TS['Hour']

#%% MASS CLOSURE
ts_sum=df_TS[:17604].sum(axis=1)
col=dt
fig, axs=plt.subplots(figsize=(5,5))
pm1_plot=plt.scatter(x=pm1['PM1'], y=ts_sum, c=date.dt.month, cmap='Greys')
axs.set_xlabel('$PM_1 (\mu g·m^{-3})$')
axs.set_ylabel('Apportioned ' +'$PM_1 (\mu g·m^{-3})$')
axs.set_xlim(-1,75)
axs.set_ylim(-1,75)
a=plt.colorbar()
a.set_label('Month')
# fig.suptitle(run_name + '  '+str(num_f) + 'F')
pm1['PM1'].reset_index(drop=True, inplace=True)
ts_sum.reset_index(drop=True, inplace=True)
fig.text(x=0.15,y=0.75,s='Linear regression \n'+
         '$R^2 = $ '+str(R2(pm1['PM1'], ts_sum))+'\n'+'y='+
         str(slope(pm1['PM1'],ts_sum)[0])+'x+'+str(slope(pm1['PM1'],ts_sum)[1]))
#%%
def Profiles_F(df, relf, name, nf):
    fig,axes=plt.subplots(nrows=nf, figsize=(10,10), sharex='col')
    for i in range(0,nf):
        axes[i].bar(amus_txt['lab'], df.iloc[:,i], color='gray')
        ax2=axes[i].twinx()
        axes[i].tick_params(labelrotation=90)
        ax2.plot(amus_txt['lab'], relf.iloc[:,i], marker='o', linewidth=False,color='black')
        axes[i].grid()
        axes[i].set_yscale('log')
    # fig.suptitle(name, fontsize=16)
    axes[2].set_ylabel('Concentration (μg·m$^{-3}$)', fontsize=14)# \t\t\t\t
    axes[num_f-1].set_xlabel('Species', fontsize=14)        
#%%
def Profiles_OA(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col',gridspec_kw=dict(width_ratios=[3, 1]))
        for i in range(0,nf):
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,1].bar(mz['lab'][93:], df.iloc[:,i][93:], color='grey')
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[93:], relf.iloc[:,i][93:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            axes[i,1].grid(axis='x')
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[93:], fontsize=17)
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)

            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)
#%% Input/Output Pie
fig, ax = plt.subplots(figsize =(5,5),
                       )
input_pie=pd.Series([3.94,1.52,1.38,1.02,0.06,1.24,0.20])
labels=['OA', 'SO4', 'NO3', 'NH4', 'Chl', 'BC$_{ff}$', 'BC$_{wb}$']
colors=['chartreuse', 'r','blue', 'sandybrown', 'fuchsia', 'dimgrey', 'brown']
pie_relf.plot.pie(autopct='%1.1f%%', labels=labels,colors=colors, startangle=90,counterclock=False,
                  labeldistance=1.1,pctdistance=0.65, textprops = dict(color ="black", fontsize=14,weight='bold') )
plt.ylabel('')
