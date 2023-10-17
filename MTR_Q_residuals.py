# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:43:27 2022

@author: Marta Via
"""
import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
'''
Sweep
'''
#%%
dfi_names=['30m', '1h', '2h', '3h', '6h', '12h', '24h']
nums=pd.Series(data=[17603,7071,3553,2377,1196,604,245], index=dfi_names)
li_df=[]
for i in range(0,len(dfi_names)): 
    r1=dfi_names[i]
    print(r1)
    run_name=r1+'_Sweep'
    namerun='$R_1 = $'+r1
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/"+run_name+'/'
    os.chdir(path)
    all_files = glob.glob(path+'Res_run*') 
    all_files.sort()
    df=pd.DataFrame()
    nums=[]
    for filename in all_files:
        nums.append(filename[-8:])
        dfi=pd.read_csv(filename, sep='\t')
        if dfi.empty or len(dfi.columns)==2:
            df=pd.concat([df, pd.Series()],axis=1)
            continue
        else:
            dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
            dfi=dfi.astype(float)
            dfi.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df=pd.concat([df, dfi.iloc[4]], ignore_index=True)
    li_df.append(df)    
#%%
li_df2=[]
method='mean'
for k in range(0,len(li_df)):
    df=li_df[k]
    a=df.iloc[:,:140]#120 for all, 140 for 24h
    b=df.iloc[:,140:280]
    c=df.iloc[:,280:]
    a.columns,b.columns,c.columns=range(0,len(a.columns)),range(0,len(b.columns)),range(0,len(c.columns))
    df_conc=pd.concat([a,b,c],axis=0)
    df2=pd.DataFrame()
    for i in range(0,140,10):
        if method=='all':
            df2[str(i)]=pd.concat([df_conc.iloc[:, i],df_conc.iloc[:, i+1],df_conc.iloc[:, i+2],df_conc.iloc[:, i+3],df_conc.iloc[:, i+4],df_conc.iloc[:, i+5],
                                      df_conc.iloc[:, i+6],df_conc.iloc[:, i+7],df_conc.iloc[:, i+8],df_conc.iloc[:, i+9]])
        if method=='mean':
            df2[str(i)]=df_conc.iloc[:, i:i+10].mean()
    li_df2.append(df2)
  #%%
positions=range(0,7)
plt.rcParams['ytick.labelsize'] = 13 
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
fig,ax=plt.subplots(1,2, gridspec_kw={'width_ratios': [6, 1]}, figsize=(12,5))  
for i in range(0,len(li_df2)):
    li_df2[i].boxplot(positions[i],showfliers=False, widths=(0.5)),#, 0.5),
                      # showmeans=True,ax=ax[0],boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
# Qlists[-1].boxplot(ax=ax[1],showfliers=False, showmeans=True,widths=(0.5),
#                    boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
ax[0].set_xticks(range(0,6))
ax[0].set_xticklabels(dfi_names[:-1], fontsize=15) 
ax[1].set_xticklabels(['24h'], fontsize=15) 
ax[0].set_ylabel('Q Residual (adim.)', fontsize=15)
ax[0].set_xlabel('                       '+'Resolution', fontsize=15)

# ax[0].set_yticklabels(fontsize=13)
#%%
'''Sumatory of all Qs!'''

#%%
li_Q=[]
n=[]
dfi_names=['30m', '1h', '2h', '3h', '6h', '12h', '24h']
for i in dfi_names:
    r1=i    
    run_name=r1+'_Sweep'
    namerun='$R_1 = $'+r1
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Solutions/"+run_name+'/'
    os.chdir(path)
    all_files = glob.glob(path+'Res_run*') 
    all_files.sort()
    df=pd.DataFrame()
    nums=[]
    for filename in all_files:
        nums.append(filename[-8:])
        dfi=pd.read_csv(filename, sep='\t')
        if dfi.empty:
            print(filename[-8:])
            df=pd.concat([df, pd.Series()],axis=1)
            continue
        else:
            dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
            dfi=dfi.astype(float)
            df=pd.concat([df, dfi.iloc[:,4]],axis=1)
    df_orig=df.copy(deep=True)
    nums=pd.Series(data=[17603,7071,3553,2377,1196,604,245], index=['30m', '1h', '2h', '3h', '6h', '12h', '24h'] )
    num_ts=nums[r1]   #30m: 17603, 1h: 7071, 2h: 3553, 3h: 2377,6h: 1196, 12h:604, 24h:245
    print(num_ts)
    df_sum=df.sum()
    li_Q.append(df_sum)
    n.append(len(df))
#%%
Q=[]
m=117.0
p=7.0
for i,j in zip(li_Q,n):
    Qexp=m*j-p*(m+j)
    Q.append(pd.DataFrame(i)/Qexp)

  #%%
Qlists=Q
plt.rcParams['ytick.labelsize'] = 13 
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
fig,ax=plt.subplots(1,2, gridspec_kw={'width_ratios': [6, 1]}, figsize=(12,5))  
for i in range(0,6):
    Qlists[i].boxplot(positions=[i],showfliers=False, widths=(0.5),ax=ax[0], #showmeans=True,
                      boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
Qlists[-1].boxplot(ax=ax[1],showfliers=False,widths=(0.5), #showmeans=True,
                    boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)
ax[0].set_xticks(range(0,6))
ax[0].set_xticklabels(dfi_names[:-1], fontsize=15) 
ax[1].set_xticklabels(['24h'], fontsize=15) 
ax[0].set_ylabel('Q/Q$_{exp}$  (adim.)', fontsize=15)
ax[0].set_xlabel(' ')


#%% 
#%%
'''Reproduce the same as the first analysis but for Cs. '''
#%%
li_Q=[]
n=[]
nb=120
dfi_names=['30m', '1h', '2h', '3h', '6h', '12h', '24h']
for i in dfi_names:
    r1=i    
    run_name=r1+'_Sweep'
    namerun='$R_1 = $'+r1
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Solutions/"+run_name+'/'
    os.chdir(path)
    all_files = glob.glob(path+'Res_run*')
    all_files.sort()
    df=pd.DataFrame()
    nums=[]
    for filename in all_files:
        nums.append(filename[-8:])
        dfi=pd.read_csv(filename, sep='\t')
        if dfi.empty:
            print(filename[-8:])
            df=pd.concat([df, pd.Series(np.nan)],axis=1)
            continue
        else:
            dfi=pd.read_csv(filename, skiprows=1,header=None, engine='python',sep='\t',keep_default_na=True,na_values='np.nan')
            dfi=dfi.astype(float)
            df=pd.concat([df, dfi.iloc[:,4]],axis=1)
    df_orig=df.copy(deep=True)
    li_Q.append(df)
    n.append(len(df))
    
#%%
    #%%
li_Q2=[]
m=117.0
p=7.0
for i in range(0,len(li_Q)):
    Qexp=n[i]*m-p*(m+n[i])
    li_Q2.append(li_Q[i]/Qexp)
    ''''AQUI M?HE QEUDAT!!! '''
#%%
li_i=li_Q2[1].values.tolist()
a,b,c=li_i[:120],li_i[120:240],li_i[240:]
abc=[]
for i in range(0,120):
    abc.append(a[i]+b[i]+c[i])
cc=[]    
for j in range(0,120,10):
    bb=[]
    for i in abc[j:j+10]:
        # print(len(i))
        bb=bb+i#[i:i+10]
    cc.append(bb)
sum_1_1=pd.DataFrame([cc[3], cc[8]])
sum_1_1=sum_1_1.T
df_Q=pd.DataFrame(data={'(1,0.001)':cc[0], '(1,0.01)': cc[1],'(1,0.1)':cc[2], '(1,1)': sum_1_1.sum(axis=1)/2.0, 
                          '(1,10)':cc[4], '(0.001,1)': cc[5],'(0.01,1)':cc[6], '(0.1,1)': cc[7],
                          '(10,1)': cc[9],'(100,1)':cc[10], '(1000,1)': cc[11]})#%%   
    #%%
#%%
plt.rcParams['ytick.labelsize'] = 13 
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
fig,ax=plt.subplots(figsize=(12,5))  
df_Q.boxplot(showfliers=False,widths=(0.5),ax=ax,
             boxprops=boxprops,meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops)

# ax.set_ylabel('Q Residuals  (adim.)', fontsize=15)
ax.set_ylabel('Q/Q$_{exp}$  (adim.)', fontsize=15)

ax.set_xlabel(' C values ')


#
