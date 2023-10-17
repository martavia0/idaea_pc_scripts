# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:08:35 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt

#%%
'''Function definition'''
def histogram_intersection(h1, h2, bins, rang):
    sm = 0
    sM=0
    dx=rang*2.0/bins
    for i in range(bins):
        sm += min(h1[i], h2[i])*dx
    return sm*100

def histogram_intersection_new(h1, h2, bins):

    min_f = [min(h1[i], h2[i]) for i in range(0,len(h1))]
    max_f = [max(h1[i], h2[i]) for i in range(0,len(h2))]
    hist_int = sp.trapezoid(min_f, x=bins[:-1])*100.0/np.trapz(max_f, x=bins[:-1])
    return hist_int
#%%
def averages(range_in, range_out, vals_in):
    vals_out=[]
    for i in range(1,len(range_out)):
        acum=[]
        for j in range(0,len(range_in)):
            if range_in[j]>range_out[i-1] and range_in[j]<= range_out[i]:
                acum.append(vals_in[j])
        vals_out.append(pd.Series(acum).sum()/float(len(acum)))
    return vals_out

#%% This cell imports the scaled residuals of all runs, concatenates them and separates them into HR, Lr
''' Importing narrow '''
run_name='30m_Narrow'
li_R,li_hr,li_lr=[],[],[]
n=[]
limiter=17603
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/' 
os.chdir(path)
all_files = glob.glob(path+'Res_run*') 
all_files.sort()
df,hr,lr=pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
        # print(names[r1])
        df=pd.concat([df, dfi.iloc[:,2]],axis=1)
        hr=pd.concat([hr, dfi.iloc[:limiter,2]],axis=1)
        lr=pd.concat([lr, dfi.iloc[limiter:,2]],axis=1)
    print(filename[-7:])
#%%
'''No filtering'''
df2,hr2,lr2=df.copy(deep=True),hr.copy(deep=True),lr.copy(deep=True)
print(df.sum().sum())
#%% 
'''Filtering by mean'''
df2, hr2,lr2=df.copy(deep=True), hr.copy(deep=True), lr.copy(deep=True)
df2.columns=[str(j) for j in range(0,len(df2.columns))]
hr2.columns, lr2.columns=df2.columns, df2.columns
li_exclud=[]
for i in range(0,len(df.columns)):
    mean_i=df2.iloc[:,i].mean()
    if mean_i> 500 or mean_i<-500 or mean_i==np.nan:
        print (mean_i)
        li_exclud.append(i)
print(len(li_exclud),', ', 100*len(li_exclud)/len(df.columns))
for j in range(1,1+len(li_exclud)):
    df2[str(li_exclud[-j])]=pd.Series(np.nan)
    hr2[str(li_exclud[-j])]=pd.Series(np.nan)
    lr2[str(li_exclud[-j])]=pd.Series(np.nan)
#%%
df2.mean().plot()
#%%
'''For narrow: 7 C values combinations --> 70 runs per nb of factors''' 
#
method='mean'
a=hr2.iloc[:,:90]
b=hr2.iloc[:,90:180]
c=hr2.iloc[:,180:]
a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_hr=pd.concat([a,b,c],axis=0)
a=lr2.iloc[:,:90]
b=lr2.iloc[:,90:180]
c=lr2.iloc[:,180:]
a.columns,b.columns, c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
df_conc_lr=pd.concat([a,b,c],axis=0)
#
df_hr, df_hr_std=pd.DataFrame(),pd.DataFrame()
for i in range(0,90,10):
    if method=='all':
        df_hr[str(i)]=pd.concat([df_conc_hr.iloc[:, i],df_conc_hr.iloc[:, i+1],df_conc_hr.iloc[:, i+2],df_conc_hr.iloc[:, i+3],df_conc_hr.iloc[:, i+4],df_conc_hr.iloc[:, i+5],
                              df_conc_hr.iloc[:, i+6],df_conc_hr.iloc[:, i+7],df_conc_hr.iloc[:, i+8],df_conc_hr.iloc[:, i+9]])
    if method=='mean':
        df_hr[str(i)]=df_conc_hr.iloc[:, i:i+10].mean(axis=1)
        df_hr_std[str(i)]=df_conc_hr.iloc[:, i:i+10].std(axis=1) 
df_lr, df_lr_std=pd.DataFrame(), pd.DataFrame()
for i in range(0,90,10):
    if method=='all':
        df_lr[str(i)]=pd.concat([df_conc_lr.iloc[:, i],df_conc_lr.iloc[:, i+1],df_conc_lr.iloc[:, i+2],df_conc_lr.iloc[:, i+3],df_conc_lr.iloc[:, i+4],df_conc_lr.iloc[:, i+5],
                                 df_conc_lr.iloc[:, i+6],df_conc_lr.iloc[:, i+7],df_conc_lr.iloc[:, i+8],df_conc_lr.iloc[:, i+9]])
    if method=='mean':
        df_lr[str(i)]=df_conc_lr.iloc[:, i:i+10].mean(axis=1)
        df_lr_std[str(i)]=df_conc_lr.iloc[:, i:i+10].std(axis=1) 

#%% We take a look at the residuals per each combination
# plt.rcParams['ytick.labelsize'] = 13 
li_comb=['(1,0.1)','(1,0.2)','(1,0.5)','(1,1)','(1,2)','(1,5)','(1,10)','(1,15)','(1,20)']
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
df_hr2, df_lr2=df_hr, df_lr
hr3=pd.DataFrame(df_hr2)
hr3.columns=li_comb
lr3=pd.DataFrame(df_lr2)
lr3.columns=li_comb
# u_u_hr=(hr2.iloc[:,3]+hr2.iloc[:,10])/2.0
# hr2.drop('(1,1)',axis=1,inplace=True)
# hr2['(1,1)']=u_u_hr
# u_u_lr=(lr2.iloc[:,3]+lr2.iloc[:,10])/2.0
# lr2.drop('(1,1)',axis=1,inplace=True)
# lr2['(1,1)']=u_u_lr
hr3 = hr3.reindex(li_comb, axis=1)
lr3 = lr3.reindex(li_comb, axis=1)
'''Here I make a modification to get rid of the blank columns'''
# hr2=hr2.drop(columns=['(1,100)', '(1,1000)', '(1000,1)'], axis=1)
# lr2=lr2.drop(columns=['(1,100)', '(1,1000)', '(1000,1)'], axis=1)

'''The modification ends here'''
fig,axs=plt.subplots(2,1, figsize=(5,10),sharex=True)
hr3.boxplot(showfliers=False, showmeans=True,ax=axs[0],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
lr3.boxplot(showfliers=False,showmeans=True, ax=axs[1],rot=90,boxprops=boxprops, medianprops=medianprops,meanprops=meanprops, whiskerprops=whiskerprops)
axs[0].set_title('HR', fontsize=14)
axs[0].set_ylabel('Scaled Residuals (adim.)', fontsize=15)
axs[1].set_title('LR', fontsize=14)
axs[1].set_ylabel('Scaled Residuals (adim.)', fontsize=15)
axs[1].set_ylim(-100,100)
fig.suptitle('Not filtered')
# axs[1].set_xticklabels( li_comb,rotation=90)
#%%
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
    values_a, bins_a, patches_a = plt.hist(HR_f, bins=4401)
    values_b, bins_b, patches_b = plt.hist(LR_f, bins=21)
    # values_a2, values_b2=values_a/values_a.sum(), values_b/values_b.sum()
    values_a2, values_b2 = values_a, values_b
    values_a3=averages(list(bins_a[:-1]), list(bins_b), values_a2)
    values_a3=pd.Series(values_a3).replace(np.nan,0)
    hist_int=histogram_intersection_new(values_a3.values, values_b2, bins_b).round(1)
    hist_intersection_r1.append(hist_int)
    
    fig,axis=plt.subplots(figsize=(4,4), dpi=100)
    hr_vals_a=pd.Series(values_a2, index=bins_a[:-1])
    hr_vals_b=pd.Series(values_b2, index=bins_b[:-1])
    axis.plot(hr_vals_a, color='grey')
    axis.plot(hr_vals_b, color='lightgrey')
    axis.set_title( ' C = '+str(li_comb[i]))
    # axs.legend(['HR', 'LR'])
    axis.text(x=-10,y=0, s='F$_{overlap}$= '+str(hist_int.round(1))+'%', fontsize=10)
    path_result ="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Treatment/hi_narrow/"
    os.chdir(path_result)
    fig.savefig(path_result+' C = '+str(df_lr.columns[i])+'_'+str(num_r)+'.png')    
    #%%
hi=pd.Series(hist_intersection_r1)
hi.index=li_comb
u_u=pd.Series((hi[3]+hi[8])/2.0, index=['(1,1)'])
hi.drop(labels=['(1,1)'],inplace=True)
hi=hi.append(u_u)
hi = hi.reindex(index = li_comb)#['(1,0.001)','(1,0.01)','(1,0.1)','(1,1)','(1,10)','(1,100)','(1,1000)','(0.001,1)','(0.01,1)','(0.1,1)', '(10,1)','(100,1)', '(1000,1)'])

fid,axis=plt.subplots()
hi.plot.bar(color='grey', ax=axis)
axs2=axis.twinx()
axs2.plot(df_hr_std.mean(), marker='>', color='k', ls='')
axs2.plot(df_lr_std.mean(), marker='<', color='k', ls='')
axis.set_xticks(list(range(0,9)))
axis.set_xticklabels(li_comb)
axs2.set_ylabel('Standard deviation \nof averaged runs (adim.)', fontsize=12)
axs2.legend([ 'HR', 'LR'],loc='upper left')
axis.set_xlabel('($C_{HR}, C_{LR}$)', fontsize=12)
axis.grid()
axis.set_ylabel('$F_{overlap}$(%)',fontsize=13)
# axis.text(x=0, y=30, s='Range:'+ ' [-'+str(num_r)+','+str(num_r)+']') # [-'+str(num_r)+','+str(num_r)+']',fontsize=10)
fid.suptitle('H.I.|$_{30m}$ = H.I.|$_{30m}$ (C$_1$,C$_2$)',fontsize=15)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    