# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:51:33 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import scipy.integrate as sp
path_py="C:/Users/maria/Documents/Marta Via/1. PhD/F. Scripts/Python Scripts"
os.chdir(path_py)
from Histograms import *
#%%
zero=0
hist=Histogram_treatment(zero)
hist.Hello()
#%% 
'''This cell imports the scaled residuals of all runs, concatenates them and separates them into HR, Lr'''
li_R,li_hr,li_lr,n=[],[],[],[]
dfi_names=['30m', '1h', '2h', '3h', '6h', '12h', '24h']
names=pd.Series(data=[17603,7071,3553,2377,1196,604,245], index=['30m', '1h', '2h', '3h', '6h', '12h', '24h'] )
for r1 in dfi_names:
    run_name=r1+'_Sweep'
    print(run_name)
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/' 
    os.chdir(path)
    all_files = glob.glob(path+'Res_run*') 
    all_files.sort()
    df, hr, lr =pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for filename in all_files:
        dfi=pd.read_csv(filename, sep='\t')
        if dfi.empty or len(dfi.columns)==2:
            df=pd.concat([df, pd.Series(np.nan)],axis=1)
            hr=pd.concat([hr, pd.Series(np.nan)],axis=1)
            lr=pd.concat([lr,pd.Series(np.nan)],axis=1)
            continue
        else:
            dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
            dfi=dfi.astype(float)
            df=pd.concat([df, dfi.iloc[:,2]],axis=1)
            hr=pd.concat([hr, dfi.iloc[:names[r1],2]],axis=1)
            lr=pd.concat([lr, dfi.iloc[names[r1]:,2]],axis=1)
    df_orig=df.copy(deep=True)
    num_ts=names[r1]
    print(num_ts)
    li_R.append(df) #list of dataframes of different r1 containing the 420 runs concatenated
    li_hr.append(hr)
    li_lr.append(lr)
    n.append(len(df))

#%%
'''Just puts name to the columns '''
dfs,dfs_hr,dfs_lr=[],[],[]
for i in range(0,len(li_R)):
    dfi=li_R[i]#
    dfi.columns=range(0,len(dfi.columns))
    dfs.append(dfi)
    dfi_hr=li_hr[i]
    dfi_hr.columns=range(0,len(dfi_hr.columns))
    dfs_hr.append(dfi_hr)
    dfi_lr=li_lr[i]
    dfi_lr.columns=range(0,len(dfi_lr.columns))
    dfs_lr.append(dfi_lr)
R_sum=pd.concat(dfs)
df_hr=pd.concat(dfs_hr)
df_lr=pd.concat(dfs_lr)

#%%
'''For anything but narrow: 12 C values combinations --> 120 runs per nb of factors'''

method='all'
a=df_hr.iloc[:,:140]#120 for all, 140 for 24h
b=df_hr.iloc[:,140:280]
c=df_hr.iloc[:,280:]
a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_hr=pd.concat([a,b,c],axis=0)
a=df_lr.iloc[:,:140]
b=df_lr.iloc[:,140:280]
c=df_lr.iloc[:,280:]
a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_lr=pd.concat([a,b,c],axis=0)

df_hr2, df_hr2_std =pd.DataFrame(), pd.DataFrame()
for i in range(0,140,10):
    if method=='all':
        df_hr2[str(i)]=pd.concat([df_conc_hr.iloc[:, i],df_conc_hr.iloc[:, i+1],df_conc_hr.iloc[:, i+2],df_conc_hr.iloc[:, i+3],df_conc_hr.iloc[:, i+4],df_conc_hr.iloc[:, i+5],
                                  df_conc_hr.iloc[:, i+6],df_conc_hr.iloc[:, i+7],df_conc_hr.iloc[:, i+8],df_conc_hr.iloc[:, i+9]])
    # if method='mean':
    #     df_hr2[str(i)]
df_lr2, df_hr2_std=pd.DataFrame(), pd.DataFrame()
for i in range(0,140,10):
    df_lr2[str(i)]=pd.concat([df_conc_lr.iloc[:, i],df_conc_lr.iloc[:, i+1],df_conc_lr.iloc[:, i+2],df_conc_lr.iloc[:, i+3],df_conc_lr.iloc[:, i+4],df_conc_lr.iloc[:, i+5],
                              df_conc_lr.iloc[:, i+6],df_conc_lr.iloc[:, i+7],df_conc_lr.iloc[:, i+8],df_conc_lr.iloc[:, i+9]])

# '''For narrow: 7 C values combinations --> 70 runs per nb of factors''' 
# #
# a=hr.iloc[:,:90]
# b=hr.iloc[:,90:180]
# c=hr.iloc[:,180:]
# a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
# df_conc_hr=pd.concat([a,b,c],axis=0)
# a=lr.iloc[:,:90]
# b=lr.iloc[:,90:180]
# c=lr.iloc[:,180:]
# a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
# df_conc_lr=pd.concat([a,b,c],axis=0)
# #
# df_hr=pd.DataFrame()
# for i in range(0,90,10):
#     df_hr[str(i)]=pd.concat([df_conc_hr.iloc[:, i],df_conc_hr.iloc[:, i+1],df_conc_hr.iloc[:, i+2],df_conc_hr.iloc[:, i+3],df_conc_hr.iloc[:, i+4],df_conc_hr.iloc[:, i+5],
#                               df_conc_hr.iloc[:, i+6],df_conc_hr.iloc[:, i+7],df_conc_hr.iloc[:, i+8],df_conc_hr.iloc[:, i+9]])
# df_lr=pd.DataFrame()
# for i in range(0,90,10):
#     df_lr[str(i)]=pd.concat([df_conc_lr.iloc[:, i],df_conc_lr.iloc[:, i+1],df_conc_lr.iloc[:, i+2],df_conc_lr.iloc[:, i+3],df_conc_lr.iloc[:, i+4],df_conc_lr.iloc[:, i+5],
#                               df_conc_lr.iloc[:, i+6],df_conc_lr.iloc[:, i+7],df_conc_lr.iloc[:, i+8],df_conc_lr.iloc[:, i+9]])

#%% We take a look at the residuals per each combination
plt.rcParams['ytick.labelsize'] = 13 
li_comb=['(1000,1)', '(100,1)','(10,1)','(1,1)','(0.1,1)','(0.01,1)','(0.001,1)',
          '(1,1000)','(1,100)','(1,10)','(1,1)', '(1,0.1)','(1,0.01)','(1,0.001)']
# li_comb=['(1,0.1)','(1,0.2)','(1,0.5)','(1,1)','(1,2)','(1,5)','(1,10)','(1,15)','(1,20)']
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
df_hr3, df_lr3=df_hr, df_lr
hr2=pd.DataFrame(df_hr2)
hr2.columns=li_comb
lr2=pd.DataFrame(df_lr2)
lr2.columns=li_comb

hr2 = hr2.reindex(li_comb, axis=1)
lr2 = lr2.reindex(li_comb, axis=1)

fig,axs=plt.subplots(2,1, figsize=(5,10),sharex=True)
hr2.boxplot(showfliers=False,showmeans=True, ax=axs[0],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
lr2.boxplot(showfliers=False,showmeans=True,ax=axs[1],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
axs[0].set_title('HR', fontsize=14)
axs[0].set_ylabel('Scaled Residuals (adim.)\n', fontsize=15)
axs[1].set_title('LR', fontsize=14)
axs[1].set_ylabel('Scaled Residuals (adim.)', fontsize=15)
axs[1].set_ylim(-100,100)
# axs[1].set_xticklabels( li_comb,rotation=90)

#%%
hist_intersection_r1=[]
hist_intersection_r1=[]

for i in range(0,len(df_lr2.columns)):
    res_HR=df_hr2.iloc[:,i]
    res_LR=df_lr2.iloc[:,i]
    num_r=50
    # quant=pd.Series([res_HR.quantile(0.25),res_HR.quantile(0.75), res_LR.quantile(0.25), res_LR.quantile(0.75)])
    quant=pd.Series([num_r*(-1), num_r, num_r*(-1), num_r])
    fig2, axs2=plt.subplots(figsize=(5,5), dpi=100)
    HR_f = res_HR[(res_HR >=quant.iloc[0]) & (res_HR <=quant.iloc[1])]
    LR_f = res_LR[(res_LR >=quant.iloc[2]) & (res_LR <=quant.iloc[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=1700)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=8)
    '''Normalise? NO'''
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    # values_a2, values_b2 = values_a, values_b
    values_a3=averages(list(bins_a[:-1]), list(bins_b), values_a2)
    values_a3=pd.Series(values_a3).replace(np.nan,0)
    hist_int=hist.histogram_intersection_trapezoids(values_a3.values, values_b2, bins_b).round(1)
    hist_intersection_r1.append(hist_int)
    
    fig,axis=plt.subplots(figsize=(4,4), dpi=100)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    axis.plot(hr_vals_a, color='grey')
    axis2=axis.twinx()
    axis.plot(hr_vals_b, color='lightgrey')
    axis.set_title( ' C = '+str(li_comb[i]))
    # axs.legend(['HR', 'LR'])
    # axis.text(x=-10,y=0, s='F$_{overlap}$= '+str(hist_int.round(1))+'%', fontsize=10)
    path_result ="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Treatment/hi/tests"
    os.chdir(path_result)
    # plt.savefig(' C = '+str(df_lr.columns[i])+'_'+str(num_r)+'.png')    
#%%

#%%
hi=pd.Series(hist_intersection_r1)
# li_comb=['(1,0.001)','(1,0.01)','(1,0.1)','(1,1)','(1,10)','(1,100)','(1,1000)','(0.001,1)','(0.01,1)','(0.1,1)', '(1,1)','(10,1)','(100,1)', '(1000,1)']
hi.index=li_comb
u_u=pd.Series((hi[3]+hi[8])/2.0, index=['(1,1)'])
hi.drop(labels=['(1,1)'],inplace=True)
hi=hi.append(u_u)
hi = hi.reindex(index = li_comb)#['(1,0.001)','(1,0.01)','(1,0.1)','(1,1)','(1,10)','(1,100)','(1,1000)','(0.001,1)','(0.01,1)','(0.1,1)', '(10,1)','(100,1)', '(1000,1)'])
# hi = hi.reindex(index = ['(0.5,1)','(1,1)','(2,1)','(5,1)','(10,1)', '(15,1)', '(20,1)'])
fid,axis=plt.subplots()
# hi.drop(['(1,100)','(1,1000)','(100,1)', '(1000,1)'], axis=0, inplace=True)
hi.plot.bar(color='grey', ax=axis)
axis.set_xlabel('Combinations', fontsize=13)
axis.grid()
axis.set_ylabel('Histogram intersection (%)',fontsize=13)
# axis.text(x=0, y=3, s='Range:'+ ' [-'+str(num_r)+','+str(num_r)+']') # [-'+str(num_r)+','+str(num_r)+']',fontsize=10)
fid.suptitle('H.I. = H.I. (C$_1$, C$_2$)',fontsize=15)

#%%


#%% HI ERROR!
'''ERROR of HI  || STABILITY CHECKS    '''

a=df_hr.iloc[:,:140]#120 for all, 140 for 24h
b=df_hr.iloc[:,140:280]
c=df_hr.iloc[:,280:]
a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_hr=pd.concat([a,b,c],axis=0)
a=df_lr.iloc[:,:140]
b=df_lr.iloc[:,140:280]
c=df_lr.iloc[:,280:]
a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_lr=pd.concat([a,b,c],axis=0)
a=R_sum.iloc[:,:140]
b=R_sum.iloc[:,140:280]
c=R_sum.iloc[:,280:]
a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
dfs_mean=pd.concat([a,b,c],axis=0)
li_comb=['(1000,1)', '(100,1)','(10,1)','(1,1)','(0.1,1)','(0.01,1)','(0.001,1)',
         '(1,1000)','(1,100)','(1,10)', '(1,1)','(1,0.1)','(1,0.01)','(1,0.001)']
#
df_mean, df_hr_mean, df_hr_std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i in range(0,140,10):
    df_hr_mean[str(i)]=df_conc_hr.iloc[:, i:i+10].mean(axis=1)
    df_hr_std[str(i)]=df_conc_hr.iloc[:, i:i+10].std(axis=1)
df_hr_mean.columns=li_comb
df_hr_std.columns = li_comb 

df_lr_mean, df_lr_std = pd.DataFrame(), pd.DataFrame()
for i in range(0,140,10):
    df_lr_mean[str(i)]=df_conc_lr.iloc[:, i:i+10].mean(axis=1)
    df_lr_std[str(i)]=df_conc_lr.iloc[:, i:i+10].std(axis=1)
df_lr_mean.columns = li_comb
df_lr_std.columns = li_comb

df_mean, df_std = pd.DataFrame(), pd.DataFrame()
for i in range(0,140,10):
    df_mean[str(i)]=df_conc_lr.iloc[:, i:i+10].mean(axis=1)
    df_std[str(i)]=df_conc_lr.iloc[:, i:i+10].std(axis=1)
df_mean.columns = li_comb
df_std.columns = li_comb
# df_lr_std.boxplot(showfliers=False, rot=90)
df_std_mean=df_std.mean(axis=0)
#%% We eliminate the duplicity of (1,1)
u_u_hr_m, u_u_lr_m = pd.Series((df_hr_mean.iloc[:,3]+df_hr_mean.iloc[:,10])/2.0), pd.Series((df_lr_mean.iloc[:,3]+df_lr_mean.iloc[:,10])/2.0)
u_u_hr_s, u_u_lr_s = pd.Series((df_hr_std.iloc[:,3]+df_hr_std.iloc[:,10])/2.0), pd.Series((df_lr_std.iloc[:,3]+df_lr_std.iloc[:,10])/2.0)
u_u_m, u_u_s = pd.Series((df_mean.iloc[:,3]+df_mean.iloc[:,10])/2.0), pd.Series((df_std.iloc[:,3]+df_std.iloc[:,10])/2.0)

df_hr_mean.drop(labels=['(1,1)'],inplace=True,axis=1)
df_lr_mean.drop(labels=['(1,1)'],inplace=True,axis=1)
df_hr_std.drop(labels=['(1,1)'],inplace=True,axis=1)
df_lr_std.drop(labels=['(1,1)'],inplace=True,axis=1)
df_mean.drop(labels=['(1,1)'],inplace=True,axis=1)
df_std.drop(labels=['(1,1)'],inplace=True,axis=1)


df_hr_mean['(1,1)']=u_u_hr_m
df_hr_std['(1,1)']=u_u_hr_s
df_lr_mean['(1,1)']=u_u_lr_m
df_lr_std['(1,1)']=u_u_lr_s
df_mean['(1,1)']=u_u_m
df_std['(1,1)']=u_u_m

li_comb1=['(1000,1)', '(100,1)','(10,1)','(1,1)','(0.1,1)','(0.01,1)','(0.001,1)',
         '(1,1000)','(1,100)','(1,10)', '(1,0.1)','(1,0.01)','(1,0.001)']
df_hr_mean = df_hr_mean.reindex(li_comb1,axis=1)
df_lr_mean = df_lr_mean.reindex(li_comb1,axis=1)
df_hr_std = df_hr_std.reindex(li_comb1,axis=1)
df_lr_std = df_lr_std.reindex(li_comb1,axis=1)
df_mean = df_mean.reindex(li_comb1,axis=1)
df_std = df_std.reindex(li_comb1,axis=1)

#%%
fig,axs=plt.subplots(2,2, figsize=(12,10),sharex=True)
fig.tight_layout(pad=7)
df_hr_mean.boxplot(showfliers=False,showmeans=True, ax=axs[0,0],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
df_hr_std.boxplot(showfliers=False,showmeans=True, ax=axs[0,1],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
df_lr_mean.boxplot(showfliers=False,showmeans=True, ax=axs[1,0],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
df_lr_std.boxplot(showfliers=False,showmeans=True,ax=axs[1,1],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
axs[0,0].set_title('HR', fontsize=14)
axs[0,0].set_ylabel('Scaled Residuals (adim.)\n MEAN', fontsize=15)
axs[0,1].set_ylabel('Scaled Residuals (adim.)\n ST. DEV.', fontsize=15)
axs[1,0].set_title('LR', fontsize=14)
axs[1,0].set_ylabel('Scaled Residuals (adim.)\n MEAN ', fontsize=15)
axs[1,1].set_ylabel('Scaled Residuals (adim.)\n ST. DEV. ', fontsize=15)
axs[1,0].set_ylim(-100,100)
axs[1,1].set_ylim(-10,100)

# axs[1].set_xticklabels( li_comb,rotation=90)
#%%
path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Treatment/hi/mean/"  
hi_hm=[]
hist_intersection_r1=[]
for i in range(0,len(df_hr_mean.columns)):
    res_HR=df_hr_mean.iloc[:, i]
    res_LR=df_lr_mean.iloc[:,i]
    limit=50
    quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
    fig, axs=plt.subplots()
    HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
    LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=4400)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=21)
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    values_a2, values_b2=values_a, values_b
    values_a3=averages(list(bins_a[:-1]), list(bins_b), values_a2)
    values_a3=pd.Series(values_a3).replace(np.nan,0)
    hist_int=(histogram_intersection_new(values_a3, values_b2,bins_b))
    hist_intersection_r1.append(hist_int)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    fig2,axs2=plt.subplots()
    axs2.plot(hr_vals_a, color='grey')
    axs2.plot(hr_vals_b, color='lightgrey')
    axs.legend(['HR', 'LR'])
    axs2.text(0,0,str(hist_int.round(0))+' %'+'\n'+'C='+df_hr_mean.columns[i])
    # if i==0:    
    fig2.savefig(path_result+'fig'+str(i)+'.png')
#%%
hi_hm_mean=pd.Series(hist_intersection_r1)
hi_hm_mean.index=li_comb1
fig,axs=plt.subplots()
hi_hm_mean.plot.bar(ax=axs, color='grey', label='_nolegend_')
axs2=axs.twinx()
axs.grid()

df_hr_std_mean, df_lr_std_mean = df_hr_std.mean(), df_lr_std.mean()
df_hr_std_mean.plot(ls='', marker='<', ax=axs2, color='k')
df_lr_std_mean.plot(ls='', marker='>', ax=axs2, color='k')
# axs2.set_ylim(0,100)
axs2.legend(['HR', 'LR'], loc='upper right')
axs.set_ylabel('$F_{overlap}$ (%)', fontsize=12)
axs2.set_ylabel('Scaled residuals \n standard deviation (adim.)', fontsize=11)
#%%

'''SWEEP HEATMAP'''

#%%
# li_hr, li_lr
#%%
'''For anything but narrow: 14 C values combinations --> 140 runs per nb of factors'''
method='mean'
df_lr_r1, df_hr_r1= [],[]
for i in range(0,len(li_lr)):
    print(i)
    df_hr,df_lr=li_hr[i],li_lr[i]
    a=df_hr.iloc[:,:140]#120 for all, 140 for 24h
    b=df_hr.iloc[:,140:280]
    c=df_hr.iloc[:,280:]
    a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
    df_conc_hr=pd.concat([a,b,c],axis=0)
    a=df_lr.iloc[:,:140]
    b=df_lr.iloc[:,140:280]
    c=df_lr.iloc[:,280:]
    a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
    df_conc_lr=pd.concat([a,b,c],axis=0)
    
    df_hr2, df_hr2_std=pd.DataFrame(), pd.DataFrame()
    for i in range(0,140,10):
        if method=='all':
            df_hr2[str(i)]=pd.concat([df_conc_hr.iloc[:, i],df_conc_hr.iloc[:, i+1],df_conc_hr.iloc[:, i+2],df_conc_hr.iloc[:, i+3],df_conc_hr.iloc[:, i+4],df_conc_hr.iloc[:, i+5],
                                      df_conc_hr.iloc[:, i+6],df_conc_hr.iloc[:, i+7],df_conc_hr.iloc[:, i+8],df_conc_hr.iloc[:, i+9]])
        if method=='mean':
            df_hr2[str(i)]=df_conc_hr.iloc[:, i:i+10].mean(axis=1)
            df_hr2_std[str(i)]=df_conc_hr.iloc[:, i:i+10].std(axis=1)
    df_lr2, df_lr2_std=pd.DataFrame(),pd.DataFrame()
    for i in range(0,140,10):
        if method=='all':
            df_lr2[str(i)]=pd.concat([df_conc_lr.iloc[:, i],df_conc_lr.iloc[:, i+1],df_conc_lr.iloc[:, i+2],df_conc_lr.iloc[:, i+3],df_conc_lr.iloc[:, i+4],df_conc_lr.iloc[:, i+5],
                                      df_conc_lr.iloc[:, i+6],df_conc_lr.iloc[:, i+7],df_conc_lr.iloc[:, i+8],df_conc_lr.iloc[:, i+9]])     
        if method=='mean':
            df_lr2[str(i)]=df_conc_lr.iloc[:, i:i+10].mean(axis=1)
            df_lr2_std[str(i)]=df_conc_lr.iloc[:, i:i+10].std(axis=1)               
    df_lr_r1.append(df_lr2)
    df_hr_r1.append(df_hr2)
#%%
path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Treatment/hi/mean/"  
hi_hm=[]
for k in range(0,len(df_lr_r1)):
    hist_intersection_r1=[]
    for i in range(0,len(df_hr_r1[k].columns)):
        res_HR=df_hr_r1[k].iloc[:, i]
        res_LR=df_lr_r1[k].iloc[:,i]
        limit=50
        quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
        fig, axs=plt.subplots()
        HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
        LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
        values_a, bins_a, patches_a = plt.hist(HR_f, bins=int(0.25*n[k]))
        values_b, bins_b, patches_b = plt.hist(LR_f, bins=21)
        # values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
        values_a2, values_b2=values_a, values_b
        values_a3=averages(list(bins_a[:-1]), list(bins_b), values_a2)
        values_a3=pd.Series(values_a3).replace(np.nan,0)
        hist_int=(histogram_intersection_new(values_a3, values_b2,bins_b))
        hist_intersection_r1.append(hist_int)
        hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
        hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
        fig2,axs2=plt.subplots()
        axs2.plot(hr_vals_a, color='grey')
        axs2.plot(hr_vals_b, color='lightgrey')
        axs2.legend(['HR', 'LR'])
        axs2.text(0,0,str(hist_int.round(0))+' %'+'\n'+'C='+li_comb[i])
        # if k==0:    
        fig2.savefig(path_result+str(dfi_names[k])+'_'+li_comb[i]+'.png')
    hi_hm.append(hist_intersection_r1)
#%%
hi=pd.DataFrame(hi_hm)
hi.index=['30m','1h', '2h','3h','6h', '12h', '24h' ]
hi.columns=li_comb
u_u=pd.Series((hi.iloc[:,3]+hi.iloc[:,8])/2.0)
hi.drop(labels=['(1,1)'],inplace=True,axis=1)
hi['(1,1)']=u_u
hi = hi.reindex(li_comb1,axis=1)
# hi=hi.T
#%%
fig, axs=plt.subplots(figsize=(10,5))

hm=axs.pcolor(hi, cmap='viridis',edgecolors='w', linewidth=2)
cb=plt.colorbar(hm)
cb.set_label('$F_{overlap}$ (%)', fontsize=15)
axs.set_xticks(np.arange(0,13)+0.5)
axs.set_xticklabels(hi.columns,rotation=90, fontsize=12)
axs.set_yticks(np.arange(0,7)+0.5)
axs.set_yticklabels(hi.index, fontsize=12)
axs.set_xlabel('($C_{HR}, C_{LR}$)',fontsize=15)
axs.set_ylabel('$R_1$',fontsize=15)
fig.text(x=0.75,y=0.005,s='Range: [-'+str(limit)+','+str(limit)+']')
for i in range(len(hi)):
    for j in range(0,len(hi.columns)):
        text = axs.text(j+0.5, i+0.5, hi.iloc[i, j].round(1),
                       ha="center", va="center", color="w",fontsize=12)
#%%

df_30m=hi.loc['30m']
fig,axis=plt.subplots()
df_30m.plot.bar(color='grey', ax=axis, grid=True)
axis.set_xlabel('($C_{HR}, C_{LR}$)', fontsize=12)
axs2=axis.twinx()
df_hr_std_mean.plot(ls='', marker='<', ax=axs2, color='k')
df_lr_std_mean.plot(ls='', marker='>', ax=axs2, color='k')
axs2.legend(['HR', 'LR'], loc='upper right')
axis.set_ylabel('$F_{overlap}$ (%)', fontsize=12)
axs2.set_ylabel('Standard deviation \nof averaged runs (adim.)', fontsize=12)
# fig.text(x=0.75,y=-0.1,s='Range: [-'+str(limit)+','+str(limit)+']')
fig.suptitle('H.I.|$_{30m}$ = H.I.|$_{30m}$ (C$_1$,C$_2$)',fontsize=15)
#%%

#%%
'''For only one resolution'''
li_R,li_hr,li_lr=[],[],[]
r1='30m'   
n=17603
run_name=r1+'_Sweep'
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/30m_Sweep/"
os.chdir(path)
all_files = glob.glob(path+'Res_run*') 
all_files.sort()
df=pd.DataFrame()
hr=pd.DataFrame()
lr=pd.DataFrame()
for filename in all_files:
    dfi=pd.read_csv(filename, sep='\t')
    print(filename[-7:])
    if dfi.empty or len(dfi.columns)==2:
        # print(filename[-7:])
        df=pd.concat([df, pd.Series(np.nan)],axis=1)
        hr=pd.concat([hr, pd.Series(np.nan)],axis=1)
        lr=pd.concat([lr,pd.Series(np.nan)],axis=1)
        continue
    else:
        dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
        dfi=dfi.astype(float)
        # print(names[r1])
        df=pd.concat([df, dfi.iloc[:,2]],axis=1)
        hr=pd.concat([hr, dfi.iloc[:n,2]],axis=1)
        lr=pd.concat([lr, dfi.iloc[n:,2]],axis=1)
df_orig=df.copy(deep=True)
num_ts=names[r1]   #30m: 17603, 1h: 7071, 2h: 3553, 3h: 2377,6h: 1196, 12h:604, 24h:245
print(num_ts)
#%%
df_hr,df_lr=hr,lr
a=df_hr.iloc[:,:140]#120 for all, 140 for 24h
b=df_hr.iloc[:,140:280]
c=df_hr.iloc[:,280:]
a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_hr=pd.concat([a,b,c],axis=0)
a=df_lr.iloc[:,:140]
b=df_lr.iloc[:,140:280]
c=df_lr.iloc[:,280:]
a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_lr=pd.concat([a,b,c],axis=0)
df_hr_mean, df_hr_std, df_lr_mean, df_lr_std = pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
df_hr2=pd.DataFrame()
for i in range(0,140,10):
    df_hr2[str(i)]=pd.concat([df_conc_hr.iloc[:, i],df_conc_hr.iloc[:, i+1],df_conc_hr.iloc[:, i+2],df_conc_hr.iloc[:, i+3],df_conc_hr.iloc[:, i+4],df_conc_hr.iloc[:, i+5],
                              df_conc_hr.iloc[:, i+6],df_conc_hr.iloc[:, i+7],df_conc_hr.iloc[:, i+8],df_conc_hr.iloc[:, i+9]])
    df_hr_mean[str(i)]=df_conc_hr.iloc[:, i:i+10].mean(axis=1)
    df_hr_std[str(i)]=df_conc_hr.iloc[:, i:i+10].std(axis=1)
df_lr2=pd.DataFrame()
for i in range(0,140,10):
    df_lr2[str(i)]=pd.concat([df_conc_lr.iloc[:, i],df_conc_lr.iloc[:, i+1],df_conc_lr.iloc[:, i+2],df_conc_lr.iloc[:, i+3],df_conc_lr.iloc[:, i+4],df_conc_lr.iloc[:, i+5],
                              df_conc_lr.iloc[:, i+6],df_conc_lr.iloc[:, i+7],df_conc_lr.iloc[:, i+8],df_conc_lr.iloc[:, i+9]])
    df_lr_mean[str(i)]=df_conc_lr.iloc[:, i:i+10].mean(axis=1)
    df_lr_std[str(i)]=df_conc_lr.iloc[:, i:i+10].std(axis=1)
    #%%

hist_intersection_r1=[]
for i in range(0,len(df_hr2.columns)):
    res_HR=df_hr2.iloc[:,i]
    res_LR=df_lr2.iloc[:,i]
    limit=50
    print(i)
    quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
    fig, axs=plt.subplots()
    HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
    LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=100)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=100)
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    hist_int=(histogram_intersection(values_a2, values_b2, 100, 50))
    hist_intersection_r1.append(hist_int)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    fig,axs=plt.subplots()
    axs.plot(hr_vals_a, color='grey')
    axs.plot(hr_vals_b, color='lightgrey')
    axs.legend(['HR', 'LR'])
    # path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Plots/" +r1  
    # os.chdir(path_result)
    # plt.savefig(namerun + ' C = '+df_lr.columns[i]+'r50.png')
hi=pd.Series(hist_intersection_r1)
hi.index =li_comb
#%%
num_r=limit
fig, ax=plt.subplots(figsize=(6,4))
hi.plot.bar(color='gray',ax=ax)
ax.set_xlabel('(C$_{1}$, C$_{2}$) values', fontsize=14)
ax.set_ylabel('Histogram intersection (%)', fontsize=13)
axs2=ax.twinx()
axs2.plot(df_hr_std.mean(), marker='>', color='k', ls='')
axs2.plot(df_lr_std.mean(), marker='<', color='k', ls='')
ax.set_xticks(list(range(0,14)))
ax.set_xticklabels(li_comb)
axs2.set_ylabel('Standard deviation of averaged runs (adim.)', fontsize=13)
axs2.legend([ 'HR', 'LR'],loc='upper left')
ax.set_xlabel('Combinations', fontsize=13)
ax.grid()
ax.set_ylim(0,70)
ax.set_ylabel('Histogram intersection (%)',fontsize=13)
ax.text(x=10,y=65, s='Range:'+ ' [-'+str(num_r)+','+str(num_r)+']') # [-'+str(num_r)+','+str(num_r)+']',fontsize=10)
fig.suptitle('H.I.|$_{30m}$ = H.I.|$_{30m}$ (C$_1$,C$_2$)',fontsize=15)
#%% Now same for the mean!

hist_intersection_r1=[]

for i in range(0,len(df_hr2.columns)):
    res_HR=df_hr_mean.iloc[:,i]
    res_LR=df_lr_mean.iloc[:,i]
    limit=50
    print(i)
    quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
    fig, axs=plt.subplots()
    HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
    LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=100)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=100)
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    hist_int=(histogram_intersection(values_a2, values_b2, 100, 50))
    hist_intersection_r1.append(hist_int)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    fig,axs=plt.subplots()
    axs.plot(hr_vals_a, color='grey')
    axs.plot(hr_vals_b, color='lightgrey')
    axs.legend(['HR', 'LR'])

    # path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Plots/" +r1  
    # os.chdir(path_result)
    # plt.savefig(namerun + ' C = '+df_lr.columns[i]+'r50.png')
hi=pd.Series(hist_intersection_r1)
hi.index =li_comb
#%%
fig, ax=plt.subplots(figsize=(6,4))
hi.plot.bar(color='gray',ax=ax, label='_nolegend_')
ax.set_xlabel('(C$_{1}$, C$_{2}$) values', fontsize=14)
ax.set_ylabel('Average hHistogram intersection (%)', fontsize=13)
ax.grid()
fig.text(x=0.7,y=-0.1,s='Range: [-'+str(limit)+','+str(limit)+']')
fig.suptitle('H.I.|$_{30m}$ = H.I.|$_{30m}$ (C$_1$,C$_2$)', fontsize=16)
axs2=ax.twinx()
axs2.plot(df_hr_std.mean(), marker='>', color='k', ls='')
axs2.plot(df_lr_std.mean(), marker='<', color='k', ls='')
ax.set_xticks(list(range(0,14)))
ax.set_xticklabels(li_comb)
axs2.set_ylabel('Standard deviation of averaged runs (adim.)', fontsize=13)
axs2.legend([ 'HR', 'LR'])
#%%
# Plotting the selected histogram results

res_HR=df_hr2.iloc[:,0]
res_LR=df_lr2.iloc[:,0]
limit=50
quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
fig, axs=plt.subplots()
HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
values_a, bins_a, patches_a = plt.hist(HR_f, bins=100)
values_b, bins_b, patches_b = plt.hist(LR_f, bins=100)
values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
hist_int=(histogram_intersection(values_a2, values_b2, 100))
hist_intersection_r1.append(hist_int)
hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
fig,axs=plt.subplots(figsize=(5,5))
axs.plot(hr_vals_a, color='grey')
axs.plot(hr_vals_b, color='lightgrey')
axs.legend(['HR', 'LR'])
axs.text(x=-50, y=0.05, s='Histogram intersection: 80%')
axs.text(x=-50, y=0.047, s='Range: [-'+str(limit)+','+str(limit)+']')
axs.set_ylabel('Frequency (adim.)')
axs.set_xlabel('Scaled Residuals (adim.)')
fig.suptitle('R$_1$ = 30 min ; C$_1$=0.5, C$_2$=1')

#%%
''' INDIVIDUAL RUN CHECK  '''
#%%
run_name='6h_C_1_10'
namerun='$R_1 = 30m, C=(1,1)$'
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Results/"+run_name+'/'
os.chdir(path)
all_files = glob.glob(path+'Res_run*') 
df=pd.DataFrame()
for filename in all_files:
    dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
    dfi=dfi.astype(float)
    df=pd.concat([df, dfi.iloc[:,2]],axis=1)
df_orig=df.copy(deep=True)
#%%%
num_ts=17603#245 for 24h, 604 for 12h, 1196 for 6h, 2377 for 3h, 3553 for 2h, 7071 for 1h, 17603 for 30m
residuals_HR, residuals_LR=pd.Series(), pd.Series()
residuals_HR=df.iloc[:num_ts]
residuals_LR=df.iloc[num_ts:]
res_HR, res_LR = pd.Series(), pd.Series()
for i in range(0,len(residuals_HR.columns)):
    res_HR=pd.concat([res_HR, residuals_HR.iloc[:,i]])
    res_LR=pd.concat([res_LR, residuals_LR.iloc[:,i]])
#%%
quant=pd.Series([res_HR.quantile(0.20),res_HR.quantile(0.8), 
                 res_LR.quantile(0.2), res_LR.quantile(0.8)],
                index=['P25_HR', 'P75_HR','P25_LR', 'P75_LR'])
# quant=pd.Series([-10,10,-10,10])
# HR_f=res_HR.loc[(res_HR> quant[0]) & (res_HR <= quant[1])]
HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
values_a, bins_a, patches_a = plt.hist(HR_f, bins=100, density=True)
values_a=values_a/values_a.sum()
values_b, bins_b, patches_b = plt.hist(LR_f, bins=100, density=True)
values_b=values_b/values_b.sum()
hist_int=(histogram_intersection(values_a, values_b,100))
#%%
fig,ax=plt.subplots()
ax.bar(x=bins_a[:-1],height=values_a,  color='grey')
ax.bar(x=bins_b[:-1],height=values_b,  color='lightgray')
#%%
fig, axs=plt.subplots(figsize=(3,3))
axs.boxplot([res_HR, res_LR], labels=('HR', 'LR'),showfliers=False)
axs.grid()
axs.set_xlabel('Subdataset')
axs.set_ylabel('Scaled residuals ($μg·m^{-3}$)')
fig.suptitle(namerun)
axs.text(x=0.6,y=-20, s='Histogram intersection: '+str(hist_int.round(1))+'%')
# residuals.boxplot(showmeans=True, showfliers=False)
#%%
res_HR.plot.kde()
res_LR.plot.kde()
#%%

'SWEEP RESOLUTIONS'
#%%
dfi_names=['30m', '1h', '2h', '3h', '6h', '12h', '24h']
# hr_bp,lr_bp=pd.DataFrame(), pd.DataFrame()
hr_bp,lr_bp=[],[]
for i in range(0,len(li_hr)):
    print(dfi_names[i])
    hr_i_bp, lr_i_bp=pd.DataFrame(), pd.DataFrame()
    for j in range(0,len(li_hr[i].columns)):
        hr_i_bp=pd.concat([hr_i_bp, li_hr[i].iloc[:,j]], axis=0)
        lr_i_bp=pd.concat([lr_i_bp, li_lr[i].iloc[:,j]], axis=0)
    hr_bp.append(hr_i_bp)
    lr_bp.append(lr_i_bp)
    # lr_bp[dfi_names[i]]=li_lr[i].mean(axis=1)#%%.plot()
#%%
hr_bp_df,lr_bp_df=pd.DataFrame(),pd.DataFrame()
hr_bp_df=pd.concat([hr_bp[0][0],hr_bp[1][0],hr_bp[2][0],hr_bp[3][0],hr_bp[4][0],hr_bp[5][0],hr_bp[6][0]],axis=1)
# lr_bp_df=pd.concat([lr_bp[0],lr_bp[1],lr_bp[2],lr_bp[3],lr_bp[4],lr_bp[5],lr_bp[6]],axis=1)
# hr_bp_df.columns,lr_bp_df.columns=dfi_names,dfi_names
#%%
for i in range(0,len(hr_bp)):
    hr_bp[i].to_csv(dfi_names[i]+'_hr.txt',sep='\t')
    lr_bp[i].to_csv(dfi_names[i]+'_lr.txt',sep='\t')
    
#%%
hr_bp_df=pd.DataFrame()
hr_bp_df['30m']=hr_bp[0][0].values
hr_bp_df.index=range(0,len(hr_bp_df))
hr_bp_df['1h']=pd.concat([hr_bp[1][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[1][0])))],ignore_index=True)
hr_bp_df['2h']=pd.concat([hr_bp[2][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[2][0])))],ignore_index=True)
hr_bp_df['3h']=pd.concat([hr_bp[3][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[3][0])))],ignore_index=True)
hr_bp_df['6h']=pd.concat([hr_bp[4][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[4][0])))],ignore_index=True)
hr_bp_df['12h']=pd.concat([hr_bp[5][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[5][0])))],ignore_index=True)
hr_bp_df['24h']=pd.concat([hr_bp[6][0], pd.Series(np.nan, index=range(len(hr_bp[0][0])-len(hr_bp[6][0])))],ignore_index=True)
#%%
lr_bp_df=pd.DataFrame()
lr_bp_df['30m']=lr_bp[0][0].values
lr_bp_df.index=range(0,len(lr_bp_df))
lr_bp_df['1h']=pd.concat([lr_bp[1][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[1][0])))],ignore_index=True)
lr_bp_df['2h']=pd.concat([lr_bp[2][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[2][0])))],ignore_index=True)
lr_bp_df['3h']=pd.concat([lr_bp[3][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[3][0])))],ignore_index=True)
lr_bp_df['6h']=pd.concat([lr_bp[4][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[4][0])))],ignore_index=True)
lr_bp_df['12h']=pd.concat([lr_bp[5][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[5][0])))],ignore_index=True)
lr_bp_df['24h']=pd.concat([lr_bp[6][0], pd.Series(np.nan, index=range(len(lr_bp[0][0])-len(lr_bp[6][0])))],ignore_index=True)
#%%
#%%

fig,axs=plt.subplots(figsize=(4,9),nrows=2,sharex=True)
# axs[0].boxplot(hr_bp_df, showfliers=False)
# axs[1].boxplot(lr_bp_df, showfliers=False)
hr_bp_df.boxplot(ax=axs[0], showfliers=False, boxprops=boxprops, meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)    
lr_bp_df.boxplot(ax=axs[1], showfliers=False, boxprops=boxprops, meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)    
axs[1].set_ylim(-200,200)
axs[0].set_title('HR', fontsize=12)
axs[1].set_title('LR', fontsize=12)
axs[0].set_ylabel('Scaled Residuals(adim.)', fontsize=12)
axs[1].set_ylabel('Scaled Residuals(adim.)', fontsize=12)
#%%
li_dfs_hr=[]
# aa,bb=pd.DataFrame(), pd.DataFrame()
aa=pd.DataFrame()
for k in range(0,len(li_hr)):
    for i, j in zip(li_hr[k].columns,li_lr[k].columns ):
        aa=pd.concat([aa,li_hr[k].iloc[i]])
    li_dfs_hr.append(aa)
#%%%
li_dfs_lr=[]
# df_lr_s=pd.DataFrame()
bb=pd.DataFrame()
for k in range(0,len(li_lr)):
    for j in range(0,len(li_lr[k].columns)):
        bb=pd.concat([bb,li_lr[k].iloc[:,j]])
    li_dfs_lr.append(bb)
    # df_lr_s=pd.concat([df_lr_s, bb], axis=1)
#%%
for i in range(0,len(li_dfs_hr)):
    li_dfs_hr[i].plot()
#%%Plotting for R1
li_r1=['30m','1h', '2h','3h','6h', '12h', '24h' ]
hist_intersection_r1=[]
for i in range(0,len(li_dfs_hr)):
    res_HR=li_dfs_hr[i]
    res_LR=li_dfs_lr[i]
    limit=1000
    quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
    fig, axs=plt.subplots()
    HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
    LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=100)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=100)
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    hist_int=(histogram_intersection(values_a2, values_b2, 100))
    hist_intersection_r1.append(hist_int)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    fig,axs=plt.subplots(figsize=(4,4), dpi=100)
    axs.plot(hr_vals_a, color='grey')
    axs.plot(hr_vals_b, color='lightgrey')
    axs.set_title( ' R$_1$ = '+str(li_r1[i]))
    axs.text(x=-0.1,y=0.002, s='F$_{overlap}$= '+str(hist_int.round(1))+'%', fontsize=10)
    # axs.legend(['HR', 'LR'])
    path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Plots/ByR/"+str(limit)+'/'  
    os.chdir(path_result)
    plt.savefig(namerun + ' C = '+str(li_r1[i])+'_'+str(limit)+'.png')
#%% HI plotting
hi=pd.Series(hist_intersection_r1)
hi.index=['30m','1h', '2h','3h','6h', '12h', '24h' ]
fid,axis=plt.subplots()
hi.plot.bar(color='grey', ax=axis, grid=True)
axis.set_xlabel('Combinations')
axis.set_ylabel('Histogram intersection (%)')
axis.text(x=4.5, y=40, s='Range: [-'+str(limit)+','+str(limit)+']',fontsize=10)
fid.suptitle('H.I. = H.I. (R$_1$)',fontsize=15)

#%% We take a look at the residuals per each combination
'''Check this, it does not work perfectly'''
plt.rcParams['ytick.labelsize'] = 13 
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
res1=['30m','1h', '2h','3h','6h', '12h', '24h' ]
fig,axs=plt.subplots(1,3, figsize=(15,5), gridspec_kw={'width_ratios': [6, 6,1]})
for i in range(0,7):
    li_dfs_hr[i].boxplot(positions=[i],showfliers=False, widths=(0.5),ax=axs[0], #showmeans=True,
                      boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
for i in range(0,6):
    li_dfs_lr[i].boxplot(positions=[i],ax=axs[1],showfliers=False,widths=(0.5), #showmeans=True,
                    boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
li_dfs_lr[-1].boxplot(ax=axs[2],showfliers=False,widths=(0.5), #showmeans=True,
                     boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
axs[0].set_xticks(range(0,7))
axs[0].set_xticklabels(res1, fontsize=15)
axs[1].set_xticks(range(0,6))
axs[1].set_xticklabels(res1[:-1], fontsize=15)
axs[2].set_xticks([0,1])
axs[2].set_xticklabels(['', '24h'], fontsize=15)
axs[0].set_title('HR', fontsize=16)
axs[0].set_ylabel('Scaled Residuals (adim.)', fontsize=15)
axs[1].set_title('LR', fontsize=16)
axs
# axs[1].set_xticklabels( li_comb,rotation=90)
#%% We only select those whose combination is (1,1)
li_dfs_hr2, li_dfs_hr=[],[]
aa=pd.DataFrame()
for k in range(0,len(li_hr)):
    aa2=pd.DataFrame()
    li_hr[k].columns=range(0,len(li_hr[k].columns))
    aa2=pd.concat([li_hr[k].iloc[:, 40:50],li_hr[k].iloc[:, 170:180],li_hr[k].iloc[:, 310:320]],
                 ignore_index=True, axis=1)
    li_dfs_hr2.append(aa2)
    aa=pd.DataFrame()
    for i in range(0,len(li_dfs_hr2[k].columns)):
        aa=pd.concat([aa, li_dfs_hr2[k].iloc[:,i]])
    li_dfs_hr.append(aa)

li_dfs_lr2, li_dfs_lr=[],[]
aa=pd.DataFrame()
for k in range(0,len(li_lr)):
    aa2=pd.DataFrame()
    li_lr[k].columns=range(0,len(li_lr[k].columns))
    aa2=pd.concat([li_lr[k].iloc[:, 40:50],li_lr[k].iloc[:, 170:180],li_lr[k].iloc[:, 310:320]],
                 ignore_index=True, axis=1)
    li_dfs_lr2.append(aa2)
    aa=pd.DataFrame()
    for i in range(0,len(li_dfs_lr2[k].columns)):
        aa=pd.concat([aa, li_dfs_lr2[k].iloc[:,i]])
    li_dfs_lr.append(aa)
#%%Plotting for R1
li_r1=['30m','1h', '2h','3h','6h', '12h', '24h' ]
hist_intersection_r1=[]
for i in range(0,len(li_dfs_hr)):
    res_HR=li_dfs_hr[i]
    res_LR=li_dfs_lr[i]
    limit=50
    quant=pd.Series([limit*(-1),limit,limit*(-1), limit])#-10,10,-10,10])#])-50,50,-50,50])#
    fig, axs=plt.subplots()
    HR_f = res_HR[(res_HR >=quant[0]) & (res_HR <=quant[1])]
    LR_f = res_LR[(res_LR >=quant[2]) & (res_LR <=quant[3])]
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=100)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=100)
    values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    hist_int=(histogram_intersection(values_a2, values_b2, 100))
    hist_intersection_r1.append(hist_int)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    fig,axs=plt.subplots(figsize=(4,4), dpi=100)
    axs.plot(hr_vals_a, color='grey')
    axs.plot(hr_vals_b, color='lightgrey')
    axs.set_title( ' R$_1$ = '+str(li_r1[i]))
    axs.text(x=-0.1,y=0.002, s='F$_{overlap}$= '+str(hist_int.round(1))+'%', fontsize=10)
    # axs.legend(['HR', 'LR'])
    path_result = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Plots/ByR/"+str(limit)+'/'  
    os.chdir(path_result)
    plt.savefig(str(li_r1[i])+'_'+str(limit)+'.png')
    #%%
hi=pd.Series(hist_intersection_r1)
hi.index=['30m','1h', '2h','3h','6h', '12h', '24h' ]
fid,axis=plt.subplots()
hi.plot.bar(color='grey', ax=axis, grid=True)
axis.set_xlabel('Combinations', fontsize=12)
axis.set_xticklabels(labels=hi.index, fontsize=12)
axis.set_ylabel('F$_{overlap}$ (%)', fontsize=13)
axis.text(x=4.9, y=95, s='Range: [-'+str(limit)+','+str(limit)+']',fontsize=10)
fid.suptitle('H.I.|$_{30m}$ = H.I.|$_{30m}$ (R$_1$)',fontsize=15)    




