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
#%% Importing
run_name='30m_1_1'
num_f=6
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Solutions/"+run_name+'/'
# path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Base Case/BC/"
os.chdir(path) #changing the path for each combination
all_files = glob.glob(path+'Profile_run_'+str(num_f-6)+'*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
# all_files=all_files[:100]
for filename in all_files: #We import all runs for the given combination
    dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    dfi=dfi.astype(float)
    df=pd.concat([df, dfi],axis=1)
df_orig=df.copy(deep=True)
relf=pd.DataFrame()
ts=pd.DataFrame()
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_rel_files = glob.glob(path+'Rel_profile_run_'+str(num_f-6)+'*')
for filename in all_rel_files:
    rel_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    rel_i=rel_i.astype(float)
    relf=pd.concat([relf, rel_i],axis=1)
relf_orig=relf.copy(deep=True)
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_ts_files = glob.glob(path+'TS_run_'+str(num_f-6)+'*')
for filename in all_ts_files:
    ts_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    ts_i=ts_i.astype(float)
    ts=pd.concat([ts, ts_i],axis=1)
ts_orig=ts.copy(deep=True)
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_res_files = glob.glob(path+'Res_run_'+str(num_f-6)+'*')
res=pd.DataFrame()
for filename in all_res_files:
    res_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    res_i=res_i.iloc[:,2].astype(float)
    res=pd.concat([res, res_i],axis=1)
res_orig=res.copy(deep=True)
#%% Filtering by scaled residuals mean, P25, 75
res.columns, ts.columns, df.columns=range(0,len(res.columns)), range(0,len(ts.columns)), range(0,len(df.columns))
res2, ts2, df2=res, ts, df
for i in range(0,len(res.columns)):
    sr=res.iloc[:,i]
    # print(sr_p75)
    sr_mean=sr.mean().round(2)
    sr_p25=sr.quantile(0.25).round(2)
    sr_p75=sr.quantile(0.75).round(2)
    if (sr_mean>3.0) or (sr_mean<-3.0) or (sr_p75>10.0) or (sr_p25<-7.0):
        print(i, sr_mean, sr_p25, sr_p75)
        res2=res2.drop(i, axis=1)
        num_start=num_f*i
        ts2=ts2.drop(ts2.loc[:, num_start:num_start+num_f], axis=1)
        df2=df2.drop(df2.loc[:, num_start:num_start+num_f], axis=1)
#%% Filtering out those overshooting runs. Final number indicates the % of accepted runs.
ts2.columns=range(0,len(ts2.columns))
df2.columns=range(0,len(df2.columns))
res2.columns=range(0,len(res2.columns))
pr_f=pd.DataFrame()
ts_f=pd.DataFrame()
res_f=pd.DataFrame()
for i in range(0, len(ts2.columns)):
    # print(i, ts.iloc[:,i].mean(axis=0).round(2), ts.iloc[:,i].median(axis=0).round(2))
    if (ts2.iloc[:,i].mean(axis=0).round(2))>=20.0:
        print(ts.columns[i], 'drop')
    else:
        ts_f=pd.concat([ts_f, ts2.iloc[:,i]], axis=1)
        pr_f=pd.concat([pr_f, df2.iloc[:,i]], axis=1)
        # res_f=pd.concat([res_f, res2.iloc[:,i]], axis=1)
print(len(ts_f.columns)/ len(ts.columns))
#%%
pr_f.columns=range(0,len(pr_f.columns))
#%%
del pr_f[33]
#%% Print the mean of profiles
pr_f.mean(axis=1).plot.bar(legend=False)
#%% Clustering!
from scipy.cluster.hierarchy import dendrogram, linkage

df_orig=df.copy(deep=True)
linked=linkage(pr_f.T, 'average')
link_copy=pd.DataFrame(linked)  
# for i in linked
plt.figure(figsize=(40, 7))
plt.rcParams.update({'font.size':22})
dendr=dendrogram(linked, p=num_f, truncate_mode=None, distance_sort='descending',show_leaf_counts=True , show_contracted=True)
ax=plt.gca()
ax.tick_params(axis='x', which='major', labelsize=15)  
ax.set_ylim(0,20)

#%% Clustering
C1=pd.Series([49,4,31,25,0,51,26,52,59,32,63,61,17,18,7,62,44,1,16])
C2=pd.Series([37,8,58,43,57,67,15])
C3=pd.Series([42,12,35,23,45,4,13])
C4=pd.Series([28,9,29, 30,68])
C5=pd.Series([60,21,27,3])
C6=pd.Series([22,13,46])
C7=pd.Series([50,39])



C1=pd.Series([2,10,22,61,64, ])
C2=pd.Series([8,20,38,44,46,])
C3=pd.Series([1,3,9,15,16,18,25,28,31,36])
C4=pd.Series([0,12,17,24,62,68,])
C5=pd.Series([3, 27, 33,42,51,56,59,67,])
C6=pd.Series([6,21,47, ])
C7=pd.Series([13,19,23,30, 32,55,60,])
# C8min=pd.Series([58])
# C8=pd.Series([58,0,91,75,9])
# C9=pd.Series([65,42,49,37,27,6,80])
# C10=pd.Series([85,5,57,35,71,50,22])

clus_prof=pd.DataFrame()
clus_rel=pd.DataFrame()
clus_ts=pd.DataFrame()
clus_res=pd.DataFrame()

clus_prof['P1'], clus_prof['P2'],clus_prof['P3'] = df.iloc[:, C1].mean(axis=1), df.iloc[:, C2].mean(axis=1),  df.iloc[:, C3].mean(axis=1)
clus_prof['P4'], clus_prof['P5'], =df.iloc[:, C4].mean(axis=1), df.iloc[:, C5].mean(axis=1)
clus_prof['P6']=df.iloc[:,C6].mean(axis=1)
clus_prof['P7']=df.iloc[:,C7].mean(axis=1)
# clus_prof['P8']=df.iloc[:,C8].mean(axis=1)
# clus_prof['P9']=df.iloc[:,C9].mean(axis=1)

clus_ts['P1'], clus_ts['P2'] =ts_f.iloc[:, C1].mean(axis=1), ts_f.iloc[:, C2].mean(axis=1)
clus_ts['P3'], clus_ts['P4'] = ts_f.iloc[:, C3].mean(axis=1), ts_f.iloc[:, C4].mean(axis=1)
clus_ts['P5']= ts_f.iloc[:, C5].mean(axis=1) 
clus_ts['P6']=ts_f.iloc[:, C6].mean(axis=1)
clus_ts['P7']=ts_f.iloc[:, C7].mean(axis=1)
# # clus_ts['P8']=ts_f.iloc[:, C8].mean(axis=1)
# clus_ts['P9']=ts_f.iloc[:, C9].mean(axis=1)

#%%Individual profile plotting
for i in range(60,70):
    num=i
    Profile(df.iloc[:,num], relf.iloc[:,num], str(num))
    ts_run=pd.DataFrame(ts.iloc[:,num])
    ts_run['d']=dt
    ts_run.set_index('d', inplace=True)
# ts_p=ts_run.plot(subplots=True, figsize=(20,10),  grid=True)#marker='o'

#%%
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06_Solutions/')
mz=pd.read_csv("amus_txt.txt", sep='\t', header=0)
time=pd.read_csv("TS_12h.txt", sep='\t', header=0)
daet=pd.to_datetime(time['ts'],dayfirst=True)
#%%clus_re_prof calculation
clus_pr_sum=clus_prof.sum(axis=1)
clus_relf=pd.DataFrame()
for i in range(0,len(clus_prof.columns)):
    clus_relf=pd.concat([clus_relf, clus_prof.iloc[:,i]/clus_pr_sum], axis=1)
clus_relf.columns=range(0,len(clus_relf.columns))
#%%
Profiles(clus_prof, clus_relf, run_name,num_f)
#%%
df_TS=clus_ts.copy(deep=True)
df_TS['d']=dt
df_TS.set_index('d', inplace=True)
#%%
pie=df_TS.mean(axis=0)
print(pie)
pie_relf = pd.Series([i/pie.sum() for i in pie])
labels_pie=['P'+str(i) for i in range(1,len(pie)+1)]
# labels_pie=['Aged SOA + AS'+'\n', 'Heavy oil combustion', 'Biomass burning + fresh SOA'+'\n', 'Traffic', 'Industry'+'\n', 
#             'Fresh SOA +AN', 'Cd outbreak + Aged Aerosol'+'\n']
pie_relf.plot.pie(autopct='%1.1f%%', fontsize=10, labels=labels_pie,colors=[ 'white','gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey'])#'white',#,'slategrey'])
plt.ylabel("")
#%%
# labels_pie=['Aged SOA + AS', 'Heavy oil combustion', 'Biomass burning + fresh SOA', 
#             'Traffic', 'Industry', 'Fresh SOA +AN', 'Cd outbreak + Aged Aerosol']
ts_p=df_TS.plot(subplots=True, figsize=(20,25),  grid=True, legend=False, title=labels_pie, color='grey',lw=3)#marker='o'
pd.DataFrame(df_TS).to_csv('TS_5F_Results.txt', sep='\t')
#%%
df_TS['d']=df_TS.index
df_TS['Week']=df_TS['d'].dt.dayofweek
del df_TS['d']
Week=df_TS.groupby(df_TS['Week']).mean()
Week.plot(subplots=True, figsize=(6,10), marker='o', color='grey', legend=False, grid=True)
del df_TS['Week']
#%%
df_TS['d']=df_TS.index
df_TS['Month']=df_TS['d'].dt.month
Month=df_TS.groupby(df_TS['Month']).mean()
Month.plot(subplots=True, figsize=(10,10), marker='o', color='grey', legend=False, grid=True)
del df_TS['Month']
#%% MASS CLOSURE
path='C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Results/12h_C_1_1/'
os.chdir(path)
pm1=pd.read_csv('PM1_12h.txt', sep='\t', dayfirst=True,)
pm1.set_index(pm1['dt'], inplace=True)
# del pm1['dt']
ts_sum=df_TS.sum(axis=1)
col=dt
fig, axs=plt.subplots(figsize=(10,10))
pm1_plot=plt.scatter(x=pm1['PM1'], y=ts_sum, c=daet.dt.month, cmap='viridis')
axs.set_xlabel('$PM_1 (\mu g·m^{-3})$')
axs.set_ylabel('Apportioned ' +'$PM_1 (\mu g·m^{-3})$')
axs.set_xlim(-1,40)
axs.set_ylim(-1,40)
a=plt.colorbar()
a.set_label('Month')
fig.suptitle(run_name + '  '+str(num_f) + 'F')
pm1['PM1'].reset_index(drop=True, inplace=True)
ts_sum.reset_index(drop=True, inplace=True)
fig.text(x=0.2,y=0.7,s='Linear regression \n'+
         '$R^2 = $ '+str(R2(pm1['PM1'], ts_sum))+'\n'+'y='+
         str(slope(pm1['PM1'],ts_sum)[0])+'x+'+str(slope(pm1['PM1'],ts_sum)[1]))
#%%
pm1['PM1'].plot()
ts_sum.plot()
#%% MASS CLOSURE FILTERED
pm_in, pm_out, ts_out=[],[],[]
for i in range(0,len(pm1.PM1)):
    a=pm1.PM1.iloc[i]
    b=ts_sum.iloc[i]
    c=df_TS.iloc[i]
    if (b < a+0.5*a) and (b> a-0.5*a):
        print(i,'in', a,b)
        pm_in.append(a)
        pm_out.append(b)
        ts_out.append(c)        
       
pm1_in=pd.Series(pm_in)
pm1_out=pd.Series(pm_out)
ts_out=pd.DataFrame(ts_out)
# del ts_out['d']
#%%
fig, axs=plt.subplots(figsize=(10,10))
pm1_plot=plt.scatter(x=pm1_in, y=pm1_out, color='grey')
axs.set_xlabel('$PM_1 (\mu g·m^{-3})$')
axs.set_ylabel('Apportioned ' +'$PM_1 (\mu g·m^{-3})$')
axs.set_xlim(-1,35)
axs.set_ylim(-1,35)

fig.suptitle(run_name + '  '+str(num_f) + 'F')
pm1['PM1'].reset_index(drop=True, inplace=True)
ts_sum.reset_index(drop=True, inplace=True)
fig.text(x=0.2,y=0.7,s='Linear regression \n'+
         '$R^2 = $ '+str(R2(pm1_in, pm1_out))+'\n'+'y='+str(slope(pm1_in, pm1_out)[0])+
         'x+'+str(slope(pm1_in, pm1_out)[1]))
#%%TS Filtered
pie=ts_out.mean(axis=0)
print(pie)
pie_relf = pd.Series([i/pie.sum() for i in pie])
labels_pie=['P'+str(i) for i in range(1,len(pie)+1)]

pie_relf.plot.pie(autopct='%1.1f%%', fontsize=10, colors=['gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey'])#'white',#,'slategrey'])
plt.ylabel("")
#%%
ts_p=ts_out.plot(subplots=True, figsize=(20,25),  grid=True, legend=False, title=labels_pie, color='grey',lw=3)#marker='o'

pd.DataFrame(ts_out).to_csv('TS_5F_results_filtered.txt', sep='\t')
#%%
def Profiles(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col')
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
            axes[i,1].tick_params(labelrotation=90)
            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            fig.suptitle(name)
#%%
def Profile(df, relf, name):
        fig,axes=plt.subplots(nrows=1,ncols=2, figsize=(35,5), sharex='col')
        axes[0].bar(mz['lab'][:73], df.iloc[:73])
        ax2=axes[0].twinx()
        axes[0].tick_params(labelrotation=90)
        ax2.plot(mz.lab[:73], relf.iloc[:73], marker='o', linewidth=False,color='black')
        axes[0].grid(axis='x')
        axes[1].bar(mz['lab'][93:], df.iloc[93:])
        axes[1].set_yscale('log')
        ax4=axes[1].twinx()
        ax4.plot(mz.lab[93:], relf.iloc[93:], marker='o', linewidth=False,color='black')
        # ax4.set_yscale('log')
        axes[1].grid(axis='x')
        axes[1].tick_params(labelrotation=90)
        fig.suptitle(name)
#%%

for i in range(0,len(res.columns)):
    res_hr=res.iloc[:99,i]#99 for 24h #604 for 12h
    res_lr=res.iloc[99:,i]
    fig, ax=plt.subplots()
    ax.hist(res_hr, bins=100)
    ax.hist(res_lr, bins=100)
#%%
num_ts=604##99 for 24h, 604 for 12h
residuals_HR, residuals_LR=pd.Series(), pd.Series()
residuals_HR=res.iloc[:num_ts]#.median(axis=1)
residuals_LR=res.iloc[num_ts:]#.median(axis=1)
fig, axs=plt.subplots()
residuals_HR.hist(bins=100, alpha=0.5, density=True)
residuals_LR.hist(bins=100,alpha=0.5, density=True)
#%%
fig, axs=plt.subplots(figsize=(5,5))
residuals_HR.plot.kde()
residuals_LR.plot.kde()
axs.legend(['HR', 'LR'])
axs.grid(True)
fig.suptitle(str(num_f)+'F'+'  '+run_name)
# residuals['HR'].hist(bins=100,ax=axs)
# residuals['LR'].hist(bins=100, ax=axs)
#%%

values_a, bins_a, patches_a = plt.hist(residuals_HR, bins=100, density=True)
values_b, bins_b, patches_b = plt.hist(residuals_LR, bins=100, density=True)
print(histogram_intersection(values_a, values_b))
# residuals.boxplot(showmeans=True, showfliers=False)
#%%
def histogram_intersection(h1, h2):
    sm = 0
    sM=0
    nbins=100
    for i in range(nbins):
        sm += min(h1[i], h2[i])
        sM+=max(h1[i], h2[i])
    return sm*100/sM
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Base Case/BC/Plots/"
os.chdir(path) #changing the path for each combination
df_TS=pd.read_csv('7F_TS_depurated.txt', sep='\t')
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
pm1['PM1'].reset_index(drop=True, inplace=True)
print(R2(pm1['PM1'], ts_sum))
#%%
def Profiles(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col')
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