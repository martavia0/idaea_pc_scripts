# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:08:13 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
# import seaborn as sns
import matplotlib.pyplot as plt
import scipy.integrate as sp
from scipy import stats
#%% t-test 
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022_06/Solutions_20220909_w_CrCdCoTi/SA/Final solution/"
os.chdir(path)
input_pm1=pd.read_csv('PM1.txt', sep='\t')
a=input_pm1['PM1']
output_pm1=pd.read_csv('TS.txt', sep='\t')
b=output_pm1.sum(axis=1).iloc[:]
print(sp.stats.ttest_ind(a, b, axis=0, equal_var=False) )
                      # nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
#
#%% F* calculation
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/Solutions/SA/")
res_hr=pd.read_csv('Def_errors.txt', sep='\t')
res_hr.columns=['ind', 'Res_HR']
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/LR/solutions/F_20221111_Mg/SA/")
res_lr=pd.read_csv('Def_Res.txt', sep='\t')
res_lr.columns=['ind', 'Res_LR']
#%%
# mask_hr = (res_hr['Res_HR'] > res_hr['Res_HR'].quantile(0.05)) & (res_hr['Res_HR'] < res_hr['Res_HR'].quantile(0.95))
mask_hr = (res_hr['Res_HR'] > -50) & (res_hr['Res_HR']<50)
res_hr2=res_hr[mask_hr]
# mask_lr = (res_lr['Res_LR'] > res_lr['Res_LR'].quantile(0.05)) & (res_lr['Res_LR'] < res_lr['Res_LR'].quantile(0.95))
mask_hr = (res_lr['Res_LR'] > -50) & (res_lr['Res_LR']<50)
res_lr2=res_lr[mask_lr]

#%%
'''Function definition'''
def histogram_intersection(h1, h2, bins,rang):
    sm = 0
    sM=0
    dx=rang*2.0/bins
    for i in range(bins):
        sm =sm+ dx*min(h1[i], h2[i])
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
#%%
scres=pd.DataFrame()
scres['Res_HR'], scres['Res_LR']=HR_f, LR_f
scres.boxplot(showfliers=False)
#%%
bins=100
hist_intersection_r1=[]
fig, axs=plt.subplots(figsize=(5,5), dpi=100)
HR_f = res_hr['Res_HR']# res_hr[(res_hr['Res_HR']>=-50) & (res_hr['Res_HR'] <=50)]
LR_f = res_lr['Res_LR']#res_lr[(res_lr['Res_LR'] >=-50) & (res_lr['Res_LR'] <=50)]
# del HR_f['ind'], LR_f['ind']
values_a, bins_a, patches_a = plt.hist(HR_f, bins=4401, range=[-50,50])
values_b, bins_b, patches_b = plt.hist(LR_f, bins=21, range=[-50,50])
df_hr=pd.Series(values_b)
# df_hr.set_index(pd.Series(bins_a)[:100], inplace=True)


fig2, axs2=plt.subplots(figsize=(4,4))
values_a3=averages(list(bins_a[:-1]), list(bins_b), values_a)
values_a3=pd.Series(values_a3).replace(np.nan,0)
hist_int=histogram_intersection_new(values_a3.values, values_b, bins_b).round(1)
hist_intersection_r1.append(hist_int)
axs2.plot(bins_b[:-1],values_a3,  color='gray', label='HR')
axs2.plot(bins_b[:-1],values_b, color='silver', label='LR')
axs2.set_ylabel('Frequency (adim.)')
axs2.set_xlabel('Scaled residuals')
axs2.text(x=-50,y=40, s='F$^{*}$= '+str(hist_int.round(1))+'%', fontsize=12)
axs2.legend(['HR', 'LR'], loc='upper left')
#%%

#%%
plt.text(x=-45,y=0.17,s='F*= '+ str(histogram_intersection(values_a, values_b, bins,50).round(1))+'%', fontsize=12)
plt.legend(['HR', 'LR'],fontsize=12)
axs2.set_ylabel('Frequency (adim.)', fontsize=12)
axs2.set_xlabel('Bins', fontsize=12)
axs2.set_title('Scaled Residuals', fontsize=12)
axs2.set_xlim([-50,50])
#%%
print(histogram_intersection(values_a, values_b, bins))
#%%
'''Output pie'''
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/30m_1_2/SA/')
ts=pd.read_csv('TS_Def.txt', sep='\t')
pr=pd.read_csv('Profiles_Def.txt', sep='\t')
#%%
# iterate through rows of X
for i in range(len(X)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]
















