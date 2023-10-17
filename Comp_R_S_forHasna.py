# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:56:12 2021

@author: Marta Via
"""

import pandas as pd
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import stats
import statsmodels.api as sm
import glob
import math
#%% Importing a txt file which should include Rolling and Seasonal OA factors as well as the externals.
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Barcelona")
f1=pd.read_csv("Barcelona.txt", sep="\t",low_memory=False)
#%%Date-time arrangement:
f1['Time']=pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
dr_all=pd.date_range("2017/09/21 00:00",end="2018/10/30") # BCN
#%%  R2 function
def R2(a,b):
    c=pd.DataFrame({"a":a, "b":b})
    cm=c.corr(method='pearson')
    r=cm.iloc[0,1]
    return r**2
def slope(b,a):
    c=pd.DataFrame({"a":a, "b":b})
    mask = ~np.isnan(a) & ~np.isnan(b)
    a1=a[mask]
    b1=b[mask]
    if (a1.empty) or (b1.empty):
        s=np.nan
    else:
        s, intercept, r_value, p_value, std_err = linregress(a1,b1)
    return s
#%%
# ************************* ABSOLUTE AND FRACTION FACTORS PLOT ******************
#
# It will ask whether if you prefer to plot it in Absolute or Fraction concentrations, a well as the city name.
#
#       Red:Rolling     Blue: Seasonal
#
choose = input("Enter -Abs- or -Rel-: ")
if choose == "Abs":
    R=pd.DataFrame({'datetime':f1.Time, 'COA':f1['COA_Rolling'],'HOA': f1['HOA_Rolling'],
                'BBOA':f1['BBOA_Rolling'],'LO-OOA':f1['LO-OOA_Rolling'],'MO-OOA': f1['MO-OOA_Rolling']})
    S=pd.DataFrame({'datetime':f1.Time, 'COA':f1['COA_seas'],'HOA': f1['HOA_seas'],
                'BBOA':f1['BBOA_seas'],'LO-OOA':f1['LO-OOA_seas'],'MO-OOA': f1['MO-OOA_seas']})
if choose == "Rel":
    R=pd.DataFrame({'datetime':f1.Time, 'COA':f1['COA_Rolling']/f1['OA_app_rolling'],'HOA': f1['HOA_Rolling']/f1['OA_app_rolling'],
                    'BBOA':f1['BBOA_Rolling']/f1['OA_app_rolling'],'LO-OOA':f1['LO-OOA_Rolling']/f1['OA_app_rolling'],'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_rolling']})

    S=pd.DataFrame({'datetime':f1.Time, 'COA':f1['COA_seas']/f1['OA_app_s'],'HOA': f1['HOA_seas']/f1['OA_app_s'],
                    'BBOA':f1['BBOA_seas']/f1['OA_app_s'],'LO-OOA':f1['LO-OOA_seas']/f1['OA_app_s'],'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s']})
R=R.set_index('datetime')
S=S.set_index('datetime')

fig,axes=plt.subplots(5,1, figsize=(28,26), constrained_layout=True)
fig.canvas.set_window_title('Comparison')
plt.rcParams.update({'font.size': 22})
city=input("City name?")
fig.suptitle(city+" ("+choose+")", fontsize=28)
count=0
for c in range(5):
    name1=R.columns[c]
    name2=name1
    axes[c].plot(R.index, R[name1], marker='o', color='red')
    ax2=axes[c].twinx()
    ax2.plot(S.index, S[name2],marker='o', color='blue')
    axes[c].grid(axis='x')
    axes[c].grid(axis='y')
    axes[c].set_axisbelow(True)
    axes[c].set_title(name1)
    count=count+1
plotname_PRO = "Factors_"+choose+".png"
#plt.savefig(plotname_PRO)
#%%
# ******* ROLLING R2 PLOT FOR EXTERNAL CORRELATIONS*******
#
# The outcoming graphs are the correlation timelines for R, S and their intercomparison
#
#       Red:Rolling     Blue: Seasonal
#
def External_Corr(wdw_l): #wdw_l is the window length
    lR=[]
    lS=[]
    for i in range(0,len(dr_all)):
        st_d=dr_all[i]
        dr_14=pd.date_range(st_d, periods=wdw_l) #You can change the length of rolling R2 window here (periods=14)
        en_d=dr_14[-1]
        print(st_d,en_d)
        mask_i=(f1['Time']>st_d) & (f1['Time']<=en_d)
        f3=f1.loc[mask_i]
#You can add/substract more correlations in the following lines. 
        rsq_R=[R2(f3['HOA_Rolling'], f3['BCff']),#R2(f3['HOA_Rolling'], f3['NO']),R2(f3['HOA_Rolling'], f3['NO2']),
               R2(f3['HOA_Rolling'], f3['NOx']),R2(f3['OOA_Rolling'], f3['NH4']),R2(f3['MO-OOA_Rolling'], f3['SO4']),
               R2(f3['BBOA_Rolling'], f3['BCwb'])]
        rsq_S=[R2(f3['HOA_seas'], f3['BCff']),#R2(f3['HOA_Rolling'], f3['NO']),R2(f3['HOA_Rolling'], f3['NO2']),
               R2(f3['HOA_seas'], f3['NOx']),R2(f3['OOA_seas'], f3['NH4']),R2(f3['MO-OOA_seas'], f3['SO4']),
               R2(f3['BBOA_seas'], f3['BCwb'])]
        lR.append(rsq_R)
        lS.append(rsq_S)
#Rolling
    R=pd.DataFrame(lR, columns=['HOA vs. BCff','HOA vs. NOx', #'HOAf vs. NO','HOAf vs. NO2',
                            'OOA vs. NH4','MO-OOA vs. SO4','BBOA vs. BCwb'])
    R['datetime']=dr_all
    R=R.set_index('datetime')
    R.to_csv('Rolling_R2_14_R'+city+'.txt')
    fig_R=R.plot(subplots=True,figsize=(25,20), grid=True)[1].get_figure()
    fig_R.savefig('Rolling_R2_'+str(wdw_l)+'_R'+city+'.png')
#Seasonal    
    S=pd.DataFrame(lS, columns=['HOA vs. BCff','HOA vs. NOx', #'HOAf vs. NO','HOAf vs. NO2',
                            'OOA vs. NH4','MO-OOA vs. SO4','BBOA vs. BCwb'])
    S['datetime']=dr_all
    S=S.set_index('datetime')
    S.to_csv('Rolling_R2_14_S'+city+'.txt')
    fig_S=S.plot(subplots=True,figsize=(25,20), grid=True)[1].get_figure()
    fig_S.savefig('Rolling_R2_'+str(wdw_l)+'_S'+city+'.png')
# 
#       ROLLING-SEASONAL ROLLING R2 comparison
#
    fig,axes=plt.subplots(5,1, figsize=(28,26), constrained_layout=True)
    fig.canvas.set_window_title('Comparison')
    plt.rcParams.update({'font.size': 22})
    fig.suptitle("BARCELONA \n "+str(wdw_l)+"days window", fontsize=28)
    for c in range(5):
        name1=R.columns[c]
        name2=name1
        axes[c].plot(R.index, R[name1], marker='o', color='red')
        ax2=axes[c].twinx()
        ax2.plot(S.index, S[name2],marker='o', color='blue')
        axes[c].grid(axis='x')
        axes[c].grid(axis='y')
        axes[c].set_axisbelow(True)
        axes[c].set_title(name1)
        plotname_PRO = "Mobile_R2_Comparison_"+str(wdw_l)+".png"
        plt.savefig(plotname_PRO)
    return R-S
#%%
#Dif_7=External_Corr(7)
Dif_14=External_Corr(14)
#Dif_28=External_Corr(28)
#Dif_56=External_Corr(56)
#Dif_112=External_Corr(112)
#%%
#       ROLLING R2 for the Rolling Seasonal Dataset Comparison
#
l=[]
l2=[]
for i in range(0,len(dr_all)):
    st_d=dr_all[i]
    dr_14=pd.date_range(st_d, periods=14) #You can change the length of rolling R2 window here (periods=14)
    en_d=dr_14[-1]
    mask_i=(f1['Time']>st_d) & (f1['Time']<=en_d)
    f3=f1.loc[mask_i]
    print(st_d,en_d)
    rsq=[R2(f3['COA_Rolling'],f3['COA_seas']),R2(f3['HOA_Rolling'], f3['HOA_seas']),
         R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
         R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']),R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])]
    slp=[slope(f3['COA_Rolling'], f3['COA_seas']),slope(f3['HOA_Rolling'], f3['HOA_seas']),
         slope(f3['BBOA_Rolling'], f3['BBOA_seas']),
         slope(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']),slope(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])]
    l.append(rsq)
    l2.append(slp)
R=pd.DataFrame(l, columns=['COA_R vs. COA_S','HOA_R vs. HOA_S','BBOA_R vs. BBOA_S','LO-OOA_R vs. LO-OOA_S','MO-OOA_R vs. MO-OOA_S'])
S=pd.DataFrame(l2, columns=['COA_R vs. COA_S','HOA_R vs. HOA_S','BBOA_R vs. BBOA_S','LO-OOA_R vs. LO-OOA_S','MO-OOA_R vs. MO-OOA_S'] )
R['datetime']=dr_all
S['datetime']=dr_all
R.to_csv('Rolling_R2_Rolling_vs_Seas_f_14.txt', sep="\t")
S.to_csv('Rolling_slope_Rolling_vs_Seas_f_14.txt')
R=R.set_index('datetime')
S=S.set_index('datetime')
# Plot rolling R2 for the Rolling dataset
fig_R, axes=plt.subplots(nrows=5,ncols=1,sharex=True,figsize=(25,20), constrained_layout=True)
fig_R.canvas.set_window_title('Comparison')
fig_R.suptitle(city+" (Absolute)", fontsize=28)
plt.rcParams.update({'font.size': 22})
count=0
for c in range(5):
    name1=R.columns[c]
    name2=name1
    axes[c].plot(R.index, R[name1], marker='o', color='black')
    ax2=axes[c].twinx()
    axes[c].set_ylabel('R²')
    ax2.set_ylabel('Slope (x=S, y=R)', color='grey')
    ax2.plot(S.index, S[name2],marker='o', color='grey')
    axes[c].grid(axis='x')
    axes[c].grid(axis='y')
    axes[c].set_axisbelow(True)
    axes[c].set_title(name1)
    count=count+1
fig_R.savefig('Rolling_R2slope_Rolling_vs_Seas_14_f.png')