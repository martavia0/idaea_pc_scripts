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
#%% Importing by number of factors
run_name='30m_1_5'
num_f=8
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/'
os.chdir(path) #changing the path for each combination
all_files = glob.glob(path+'Profile_run_*') #we import all files which start by Res in the basecase folder (10 runs)
divider=int(len(all_files)/3)
# all_files=all_files[divider*(num_f-6):divider*(num_f-6+1)]
# all_files = glob.glob(path+'Profile_run_17'+'*') #we import all files which start by Res in the basecase folder (10 runs)
df=pd.DataFrame()
# all_files=all_files[:100]
for filename in all_files: #We import all runs for the given combination
    df_i=pd.read_csv(filename, sep='\t')
    if df_i.empty or len(df_i)<=2:
        df=pd.concat([df,pd.Series(np.nan)],axis=1)
    else:
        dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        dfi=dfi.astype(float)
        df=pd.concat([df, dfi],axis=1)
df_orig=df.copy(deep=True)
relf=pd.DataFrame()
ts=pd.DataFrame()
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_rel_files = glob.glob(path+'Rel_profile_run_*')
# all_rel_files=all_rel_files[divider*(num_f-6):divider*(num_f-6+1)]
for filename in all_rel_files:
    rel_i=pd.read_csv(filename, sep='\t')
    if rel_i.empty or len(rel_i)<=2:
        relf=pd.concat([relf,pd.Series(np.nan)],axis=1)
    else:
        rel_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        rel_i=rel_i.astype(float)
        relf=pd.concat([relf, rel_i],axis=1)
relf_orig=relf.copy(deep=True)
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_ts_files = glob.glob(path+'TS_run_*')
# all_ts_files=all_ts_files[divider*(num_f-6):divider*(num_f-6+1)]
for filename in all_ts_files:
    ts_i=pd.read_csv(filename, sep='\t')
    if ts_i.empty or len(ts_i)<=2:
        ts=pd.concat([ts,pd.Series(np.nan)],axis=1)
    else:
        ts_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        ts_i=ts_i.astype(float)
        ts=pd.concat([ts, ts_i],axis=1)
ts_orig=ts.copy(deep=True)
os.chdir(path) #changing the path for each combination
namef=os.listdir(path)
all_res_files = glob.glob(path+'Res_run_*')
# all_res_files=all_res_files[divider*(num_f-6):divider*(num_f-6+1)]

res=pd.DataFrame()
for filename in all_res_files:
    res_i=pd.read_csv(filename, sep='\t')
    if res_i.empty or len(res_i)<=2:
        res=pd.concat([res,pd.Series(np.nan)],axis=1)
    else:    
        res_i=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        res_i=res_i.iloc[:,2].astype(float)
        res=pd.concat([res, res_i],axis=1)
res_orig=res.copy(deep=True)
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/SA/'
res_mean=res.mean(axis=1)
res_mean.to_csv('Residuals_8F.txt', sep='\t')
#%% Filtering by scaled residuals mean, P25, 75
res.columns, ts.columns, df.columns, relf.columns=range(0,len(res.columns)), range(0,len(ts.columns)), range(0,len(df.columns)), range(0,len(relf.columns))
res2, ts2, df2, relf2=res, ts, df, relf
for i in range(0,len(res.columns)):
    sr=res.iloc[:,i]
    # print(sr_p75)  b
    sr_mean=sr.mean()#.round(2)
    sr_p25=sr.quantile(0.25).round(2)
    sr_p75=sr.quantile(0.75).round(2)
    if (sr_p75>10.0) or (sr_p25<-10.0):#(sr_mean>3.0) or (sr_mean<-3.0)or 
        print(i, sr_mean, sr_p25, sr_p75)
        res2=res2.drop(i, axis=1)
        num_start=num_f*i
        ts2=ts2.drop(ts2.loc[:, num_start:num_start+num_f], axis=1)
        df2=df2.drop(df2.loc[:, num_start:num_start+num_f], axis=1)
        relf2=relf2.drop(relf2.loc[:, num_start:num_start+num_f], axis=1)
#%% Filtering out those overshooting runs. Final number indicates the % of accepted runs.
ts2.columns=range(0,len(ts2.columns))
df2.columns=range(0,len(df2.columns))
relf2.columns=range(0,len(relf2.columns))
res2.columns=range(0,len(res2.columns))
pr_f=pd.DataFrame()
ts_f=pd.DataFrame()
res_f=pd.DataFrame()
rel_f=pd.DataFrame()
for i in range(0, len(ts2.columns)):
    # print(i, ts.iloc[:,i].mean(axis=0).round(2))#, ts.iloc[:,i].median(axis=0).round(2))
    if (ts2.iloc[:,i].mean(axis=0))>=15.0:
        print(ts.columns[i], 'drop')
        
    else:
        ts_f=pd.concat([ts_f, ts2.iloc[:,i]], axis=1)
        pr_f=pd.concat([pr_f, df2.iloc[:,i]], axis=1)
        rel_f=pd.concat([rel_f, relf2.iloc[:,i]], axis=1)
        # res_f=pd.concat([res_f, res2.iloc[:,i]], axis=1)
print(len(ts_f.columns)/ len(ts.columns))
pr_f.columns=range(0,len(pr_f.columns))
#%% Print the mean of profiles
pr_f.mean(axis=1).plot.bar(legend=False)
print(pr_f.mean())
pr_f2=pr_f/pr_f.sum()
# ts_f2=ts_f/pr_f.sum()
# ts_f2=ts_f2.iloc[:,:1975]
print(pr_f2.sum())
pr_f2.dropna(inplace=True, axis=1)
rel_f.dropna(inplace=True, axis=1)
ts_f.dropna(inplace=True, axis=1)

#%% Clustering! k-means
from scipy.cluster.vq import vq, kmeans, whiten
data=pr_f2.T
centroids, clusters = kmeans(data, num_f)
codebook=centroids
clx,_ = vq(data,centroids)
print(len(centroids), len(centroids[0]), clusters)
centroids=pd.DataFrame(centroids).T
centroids/=centroids.sum(axis=1)
centroids.plot.bar(legend=False,subplots=True)
#%%
whitened=np.array(data)
plt.scatter(whitened[:, 17], whitened[:, 18])
plt.scatter(codebook[:, 17], codebook[:, 18], c='r')
#%% Calculation of stars (rel prof)
rel_f2=rel_f.T
rel_f2['Cluster']=clx
df2=pr_f2.T
df2['Cluster']=clx
df_std=df2.groupby(by= ['Cluster']).std().T
rel_f3=rel_f2.groupby(by=['Cluster']).mean().T
rel_f3.iloc[:,:] = rel_f3.iloc[:,:].div(rel_f3.sum(axis=1), axis=0)
#%%
df_std.to_csv('Profiles_centroids_std.txt', sep='\t')
centroids.to_csv('Profiles_centroids.txt', sep='\t')
#%% Calculation of time series (TS)
ts_f2=ts_f
ts_f3=ts_f2.T
ts_f3['Cluster']=clx
ts_f3_mean=ts_f3.groupby(by=['Cluster']).mean().T
# ts_f3['Cluster']=clx
ts_f3_std=ts_f3.groupby(by=['Cluster']).std().T
# ts_f3.plot(subplots=True)
ts_f4=ts_f3.copy(deep=False)
#%%
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/')
mz=pd.read_csv("amus_txt.txt", sep='\t', header=0) 
time=pd.read_csv("date_30m.txt", sep='\t', header=0,dayfirst=True )
date=pd.to_datetime(time['date_out'],dayfirst=True)
df_TS=ts_f3_mean.copy(deep=True)
df_TS['d']=date
df_TS.set_index('d', inplace=True)
#%% Plotting
centroids_norm=centroids/centroids.sum()
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/'+run_name+'/SA/')
Profiles(centroids_norm, rel_f3, str(num_f)+' factors',num_f)
plt.savefig('Profiles'+str(num_f)+'_F_PR.png')
Profiles_std(centroids_norm, df_std,rel_f3, str(num_f)+' factors',num_f)
plt.savefig('Profiles_std'+str(num_f)+'_F_PR.png')
#%%Exporting
centroids_norm.to_csv('Profiles_'+str(num_f)+'.txt', sep='\t')
rel_f3.to_csv('RelProf_'+str(num_f)+'.txt', sep='\t')
df_TS.to_csv('TS_'+str(num_f)+'.txt', sep='\t')
df_std.to_csv('Profile_std.txt', sep='\t')
ts_f3_std.to_csv('TS_std.txt', sep='\t')
res2=res.mean()
res2.to_csv('Res_'+str(num_f)+'.txt', sep='\t')
#%%
pie=df_TS.mean(axis=0).T
print(pie)
pie_relf = pd.Series([i/pie.sum() for i in pie])
labels_pie=['P'+str(i) for i in range(1,len(pie)+1)]
# labels_pie=['Aged SOA + AS'+'\n', 'Heavy oil combustion', 'Biomass burning + fresh SOA'+'\n', 'Traffic', 'Industry'+'\n', 
#             'Fresh SOA +AN', 'Cd outbreak + Aged Aerosol'+'\n']
pie_relf.plot.pie(autopct='%1.0f%%', fontsize=10, labels=labels_pie,colors=[ 'white','gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey', '#424242'])#'white',#,'slategrey'])
plt.ylabel("")
plt.savefig(str(num_f)+'_F_Pie.png')
#%%
# labels_pie=['Aged SOA + AS', 'Heavy oil combustion', 'Biomass burning + fresh SOA', 
#             'Traffic', 'Industry', 'Fresh SOA +AN', 'Cd outbreak + Aged Aerosol']
# del df_TS['d']
df_TS.plot(subplots=True, figsize=(20,25),  grid=True, legend=False, title=labels_pie, color='grey',lw=3, fontsize=14)#marker='o'
# pd.DataFrame(df_TS).to_csv('TS_5F_Results.txt', sep='\t')
plt.savefig(str(num_f)+'_F_TS.png')

#%% Month
df_TS['d']=df_TS.index
df_TS['Month']=df_TS['d'].dt.month
Month=df_TS.groupby(df_TS['Month']).mean()
Month.plot(subplots=True, figsize=(6,8), marker='o', color='grey', legend=False, grid=True)
del df_TS['Month']
plt.savefig(str(num_f)+'_F_Month.png')
#%% Week
df_TS['d']=df_TS.index
df_TS['Week']=df_TS['d'].dt.dayofweek
del df_TS['d']
Week=df_TS.groupby(df_TS['Week']).mean()
Week.plot(subplots=True, figsize=(6,10), marker='o', color='grey', legend=False, grid=True)
del df_TS['Week']
#%%Diel
df_TS['d']=df_TS.index
df_TS['Hour']=df_TS['d'].dt.hour
Month=df_TS.groupby(df_TS['Hour']).mean()
Month.plot(subplots=True, figsize=(6,8), marker='o', color='grey', legend=False, grid=True)
del df_TS['Hour']
plt.savefig(str(num_f)+'_F_Diel.png')
#%% MASS CLOSURE
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/'+run_name+'/')
os.chdir(path)
pm1=pd.read_csv('PM1.txt', sep='\t', dayfirst=True,)
pm1.set_index(pm1['date_in'], inplace=True)
# del pm1['dt']
ts_sum=df_TS[:17604].sum(axis=1)
col=dt
fig, axs=plt.subplots(figsize=(5,5))
pm1_plot=plt.scatter(x=pm1['PM1'], y=ts_sum, c=date.dt.month, cmap='Greys')
axs.set_xlabel('$PM_1 (\mu g·m^{-3})$')
axs.set_ylabel('Apportioned ' +'$PM_1 (\mu g·m^{-3})$')
axs.set_xlim(-1,70)
axs.set_ylim(-1,70)
a=plt.colorbar()
a.set_label('Month')
# fig.suptitle(run_name + '  '+str(num_f) + 'F')
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
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col',gridspec_kw={'width_ratios': [3, 2]})
        for i in range(0,nf):
            org_prop=str(100.0*((df.iloc[:,i][:73].sum()/df.iloc[:,i].sum()).round(0)))
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,0].text(x=65, y=max(df.iloc[:,i][:73])-0.1*max(df.iloc[:,i][:73]), s='OA(%) = '+org_prop+'%', fontsize=20)
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            axes[i,1].grid(axis='x')
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)

            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)
#%%
def Profiles_std(df,df_std, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col',gridspec_kw={'width_ratios': [3, 2]})
        for i in range(0,nf):
            org_prop=str(100.0*((df.iloc[:,i][:73].sum()/df.iloc[:,i].sum()).round(0)))
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            axes[i,0].errorbar(mz['lab'][:73], df.iloc[:,i][:73], yerr=df_std.iloc[:,i][:73],  fmt='.', color='k')#, uplims=True, lolims=True)
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,0].text(x=65, y=max(df.iloc[:,i][:73])-0.1*max(df.iloc[:,i][:73]), s='OA(%) = '+org_prop+'%', fontsize=20)
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].errorbar(mz['lab'][73:], df.iloc[:,i][73:], yerr=df_std.iloc[:,i][73:],  fmt='.', color='k')#, uplims=True, lolims=True)
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            axes[i,1].grid(axis='x')
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)

            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)    
#%% Plotting
Profiles_std(centroids, df_std, rel_f3, str(num_f)+' factors',num_f)
#%%
print((100.0*df.iloc[:73, i].sum()/df.iloc[:,i].sum()).round(0))
print()
#%%
def Profiles2(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=3, figsize=(40,20), sharex='col')
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
            axes[i,2].bar()
            axes[i,1].grid(axis='x')
            axes[i,1].tick_params(labelrotation=90)
            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)
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
Profiles(centroids, rel_f3, str(num_f)+' factors',num_f)
#%%
def Profiles(df, relf, name, nf):
    fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(40,20), sharex='col')
    for i in range(0,nf):
        OA_perc=(100.0*df.iloc[:73, i].sum()/df.iloc[:,i].sum()).round(0)
        axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
        ax2=axes[i,0].twinx()
        axes[i,0].tick_params(labelrotation=90)
        ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
        axes[i,0].grid(axis='x')
        axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
        axes[i,1].set_yscale('log')
        ax4=axes[i,1].twinx()
        ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
        axes[i,0].text(x=-3,y=max(df.iloc[:73,i])*0.8,s='OA = '+str(OA_perc)+'%', fontsize=16 )
        # ax4.set_yscale('log')
        # axes[i,0].set_ylabel(labels_2[i], fontsize=17)
        axes[i,1].grid(axis='x')
        axes[i,0].tick_params(labelrotation=90, labelsize=16)
        axes[i,1].tick_params(labelrotation=90, labelsize=16)
        axes[i,0].tick_params('x', labelrotation=90, labelsize=16)
        axes[i,0].tick_params('y', labelrotation=0, labelsize=16)
        ax4.tick_params(labelrotation=0, labelsize=16)
        ax2.tick_params(labelrotation=0, labelsize=16)
        axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
        axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)

            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)   
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

#%% Input pie plot
conc=[3.94,1.52,1.38,1.02,0.06,0.76,0.08,0.21]
colors=['chartreuse', 'red', 'blue', 'orange', 'fuchsia', 'black', 'brown', 'grey']
labels=['OA', 'SO4', 'NO3', 'NH4', 'Cl', 'BCff', 'BCwb', 'F']
fig, axs=plt.subplots(figsize=(5,5))
plt.pie(conc, labels=labels, colors=colors,autopct='%1.1f%%', pctdistance=pd)
# labels_pie=['P'+str(i) for i in range(1,len(pie)+1)]

# pie_relf.plot.pie(autopct='%1.1f%%', fontsize=10, colors=['gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey'])#'white',#,'slategrey'])
# plt.ylabel("")



