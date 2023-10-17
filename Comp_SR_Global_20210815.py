# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:57:29 2021

@author: Marta Via
"""
from scipy.odr import Model, Data, ODR
import scipy
from scipy.stats.distributions import chi2
from numpy.polynomial.polynomial import polyfit
from scipy.stats import ttest_ind
import scipy as sp
import pandas as pd
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import stats
#import statsmodels.api as sm

import glob
import math
#import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#%%
#%%
cities=['Barcelona', 'Magadino','Lille','Bucharest', 'Dublin','Marseille','Tartu','Cyprus', 'SIRTA']
city_acr=['BCN', 'MAG', 'LIL', 'BUC', 'DUB',  'MAR', 'TAR', 'CYP', 'SIR']
wdw_l=14
#%%
counter=0
R_list=[]
S_list=[]
for city_i in cities:
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i+'/'
    os.chdir(path)
    a=os.listdir()
    R_list.append(pd.read_csv('Rolling_R2_14_R_'+city_acr[counter]+'.txt', sep="\t", low_memory=False))
    S_list.append(pd.read_csv('Rolling_R2_14_S_'+city_acr[counter]+'.txt', sep="\t", low_memory=False))
    counter=counter+1
#%%
R2, R2_2, R2_3, R2_4, R2_5, R=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
hoa_R, hoa_S, hoa_2_R, hoa_2_S, bboa_R, bboa_S, ooa_R, ooa_S, moooa_R, moooa_S = [],[],[],[],[],[],[],[],[],[]
for i in range(0,len(R_list)):
    for j in range(0,len(R_list[i])):
        hoa_R.append(R_list[i]['HOA vs. BCff'].iloc[j])
        hoa_S.append(S_list[i]['HOA vs. BCff'].iloc[j])
        if city_acr[i]!= 'LIL' and city_acr[i]!='SIR':
            hoa_2_R.append(R_list[i]['HOA vs. NOx'].iloc[j])
            hoa_2_S.append(S_list[i]['HOA vs. NOx'].iloc[j])
        if city_acr[i] != 'DUB':
            bboa_R.append(R_list[i]['BBOA vs. BCwb'].iloc[j])
            bboa_S.append(S_list[i]['BBOA vs. BCwb'].iloc[j])
        moooa_R.append(R_list[i]['MO-OOA vs. SO4'].iloc[j])
        moooa_S.append(S_list[i]['MO-OOA vs. SO4'].iloc[j])
        if city_acr[i] != 'CYP':
            ooa_R.append(R_list[i]['OOA vs. NH4'].iloc[j])
            ooa_S.append(S_list[i]['OOA vs. NH4'].iloc[j])
R2['HOA vs. BCff (R)']=hoa_R    
R2['HOA vs. BCff (S)']=hoa_S   
R2_2['HOA vs. NOx (R)']=hoa_2_R    
R2_2['HOA vs. NOx (S)']=hoa_2_S 
R2_3['BBOA vs. BCwb (R)']=bboa_R    
R2_3['BBOA vs. BCwb (S)']=bboa_S 
R2_4['OOA vs. NH4 (R)']=ooa_R    
R2_4['OOA vs. NH4 (S)']=ooa_S 
R2_5['MO-OOA vs. SO4 (R)']=moooa_R    
R2_5['MO-OOA vs. SO4 (S)']=moooa_S 
R=pd.concat([R2, R2_2, R2_3, R2_4, R2_5], axis=1)
#%% Print median values
print('HOA vs BCff:',pd.Series(hoa_R).median(), pd.Series(hoa_S).median())
print('HOA vs. NOx:',pd.Series(hoa_2_R).median(), pd.Series(hoa_2_S).median())
print('BBOA vs. BCwb:',pd.Series(bboa_R).median(), pd.Series(bboa_S).median())
print('MO-OOA SO4:',pd.Series(moooa_R).median(), pd.Series(moooa_S).median())
print('OOA NH4:',pd.Series(ooa_R).median(), pd.Series(ooa_S).median())

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
fig, ax = plt.subplots() 
boxprops = dict(linestyle='-', linewidth=0.6,edgecolor='black')  
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
box=R.boxplot(ax=ax, patch_artist=True, rot=90,boxprops=boxprops, color='black', showmeans=True, figsize=(10,8),meanprops=meanprops)
ax.set_ylabel('Pearson Squared Correlation Coefficient')
redpatch=mpatches.Patch(color='indianred', label='Rolling')
bluepatch=mpatches.Patch(color='cornflowerblue', label='Seasonal')
#plt.legend(handles=[redpatch, bluepatch], loc='upper right')
plt.title('Correlation with externals')
for i in range(0,10):
    box.findobj(mpatches.Patch)[i].set_edgecolor("black")
    box.findobj(mpatches.Patch)[i].set_facecolor('None')
    # if i%2 ==0:
    #     print('red')
    #     box.findobj(mpatches.Patch)[i].set_facecolor("tomato")
    # else:    
    #     box.fin
    # dobj(mpatches.Patch)[i].set_facecolor("cornflowerblue")        

#%%
SCR_list=[]
for city_i in cities:
    print(city_i)
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i+'/'
    os.chdir(path)
    a=os.listdir()
    SCR_list.append(pd.read_csv('Sc_Res_orig_res.txt', sep="\t", low_memory=False))        
#%%
SR, SR_site=pd.DataFrame(), pd.DataFrame()
seas, roll= [],[]
for i in range(0,len(SCR_list)):
    print(i)
    SR_site=pd.concat([SR_site,SCR_list[i].iloc[:,1:]],axis=1)
    for j in range(0,len(SCR_list[i])):
        roll.append(SCR_list[i]['Sc Res_R'].iloc[j])
        seas.append(SCR_list[i]['Sc Res_S'].iloc[j])
SR['Rolling']=roll 
SR['Seasonal']=seas 
#cities=['Barcelona','Cyprus','Dublin','Lille', 'Magadino','Bucharest', 'Marseille', 'Tartu']
  
#%% Scaled Residuals
fig,ax=plt.subplots(figsize=(8,8))
hist1=SR.plot.hist(bins=2000,ax=ax, density=True, grid=True, color=['tomato', 'cornflowerblue'], fontsize=16, linewidth=3, alpha=0.5)

# plt.text(0.50,0.5,'nbins=2000',  fontsize=12)
plt.xlim(-1,1)
plt.xlabel('Scaled Residuals', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.legend(fontsize=16)
# plt.title('SCALED RESIDUALS', fontsize=20)
        

#%%
boxprops = dict(linestyle='-', linewidth=0.6,edgecolor='black')  
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
fig, ax = plt.subplots(figsize=(6,6)) 

# box=R.boxplot(ax=ax, patch_artist=True, rot=90,boxprops=boxprops, color='black', showmeans=True, figsize=(10,8),meanprops=meanprops)
bp=SR.boxplot(ax=ax,patch_artist=True, boxprops=boxprops, color='black', showfliers=False, 
              showmeans=True, fontsize=18, meanprops=meanprops, figsize=(8,5))
redpatch=mpatches.Patch(color='tomato', label='Rolling')
bluepatch=mpatches.Patch(color='cornflowerblue', label='Seasonal')
bp.findobj(mpatches.Patch)[0].set_edgecolor("black")
bp.findobj(mpatches.Patch)[0].set_facecolor("tomato")
bp.findobj(mpatches.Patch)[1].set_edgecolor("black")
bp.findobj(mpatches.Patch)[1].set_facecolor("cornflowerblue")        

plt.xlabel('Scaled Residuals', fontsize=18)
plt.ylabel('Density', fontsize=18)
# plt.title('SCALED RESIDUALS', fontsize=20)
        
#%%S
cols=[]
for i in (city_acr):
    cols.append(i + ' (R)')
    cols.append(i + ' (S)')
SR_site.columns=cols

#%%
SR_site_R=SR_site[['BCN (R)', 'MAG (R)', 'LIL (R)', 'BUC (R)', 'DUB (R)','MAR (R)', 'TAR (R)', 'CYP (R)']]
SR_site_S=SR_site[['BCN (S)', 'MAG (S)', 'LIL (S)', 'BUC (S)', 'DUB (S)','MAR (S)', 'TAR (S)', 'CYP (S)']]
fig, ax =plt.subplots()

ax1=SR_site_R.boxplot(ax=ax, showfliers=False, rot=90, return_type='axes')#
SR_site_S.boxplot( showfliers=False, rot=90, ax=ax1)#
#plt.xlim(-1,1)
#%%
fig, ax =plt.subplots()

ax1=SR_site_R.boxplot(ax=ax, showfliers=False, rot=90, return_type='axes')#
SR_site_S.boxplot( showfliers=False, rot=90, ax=ax1)#

#%%
cities=['Barcelona', 'Cyprus','Dublin','Lille', 'Magadino','Marseille','SIRTA', 'Tartu']
city_acr=['BCN', 'CYP', 'DUB', 'LIL', 'MAG', 'MAR','TAR']
wdw_l=14    
        #%%
Sub_list_4443=[]
Sub_list_6044=[]
for city_i in cities:
    print(city_i)
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i+'/Profiles'
    os.chdir(path)
    a=os.listdir()
    Sub_list_4443.append(pd.read_csv('Substr_4443.txt', sep="\t", low_memory=False))
    #Sub_list_6044.append(pd.read_csv('Substr_6044.txt', sep="\t", low_memory=False))
 #%%
Sub_list=Sub_list_4443
Sub, Sub_site=pd.DataFrame(), pd.DataFrame()
seas, roll, time= [],[],[]
for i in range(0,len(Sub_list)):
    Sub_site=pd.concat([Sub_site,Sub_list[i].iloc[:,1:]],axis=1)
    for j in range(0,len(Sub_list[i])):
        roll.append(Sub_list[i]['Substr_R'].iloc[j])
        seas.append(Sub_list[i]['Substr_S'].iloc[j])
        time.append(pd.to_datetime(Sub_list[i]['Time'].iloc[j]))
Sub['Rolling']=roll 
Sub['Seasonal']=seas 
Sub['Time']=(time)
#%%
Subs=pd.DataFrame()
mask = (Sub['Rolling'] > Sub['Rolling'].quantile(0.05)) & (Sub['Rolling'] < Sub['Rolling'].quantile(0.95)) & (Sub['Seasonal'] > Sub['Seasonal'].quantile(0.05)) & (Sub['Seasonal'] < Sub['Seasonal'].quantile(0.95))
Subs['Rolling']= Sub['Rolling'].loc[mask] 
Subs['Seasonal']= Sub['Seasonal'].loc[mask] 
Subs['Time']=Sub['Time'].loc[mask]
#%%
fig,ax=plt.subplots(figsize=(6,6))
hist1=Subs[['Rolling', 'Seasonal']].plot.kde(ax=ax, grid=True, color=['tomato', 'cornflowerblue'], fontsize=16, linewidth=3)        
plt.legend(fontsize=18)
plt.xlabel('OOA apportioned mz44/mz43 - OOA profiles mz44/mz43', fontsize=14)
plt.ylabel('Density', fontsize=18)
plt.title('Substraction mz44/mz43', fontsize=20)
                
#%%
Sub.index=Sub['Time']
#Sub.drop(Sub['Time'],inplace=True)
fig2,ax2=plt.subplots()
Sub[['Rolling', 'Seasonal']].plot(grid=True)  
#%%
plotall=Subs[['Rolling', 'Seasonal']].boxplot(showfliers=False, showmeans=True,fontsize=16,return_type='both',
               patch_artist = True,meanprops=dict(marker='o',markerfacecolor='black',markeredgecolor='black'),
               medianprops=dict(color='black'),boxprops=dict(color='black'),whiskerprops=dict(color='black'))  
colors = ['indianred', 'cornflowerblue' ]
for i,box in enumerate(row['boxes']):
    box.set_facecolor(colors[i])
#%%

Subs['Month']=Subs['Time'].dt.month
season=[]
for i in Subs['Month']:
    if i==1 or i==2 or i==12:
        season.append('DJF')
    if i==3 or i==4 or i==5:
        season.append('MAM')
    if i==6 or i==7 or i==8:
        season.append('JJA')    
    if i==9 or i==10 or i==11:
        season.append('SON')
Subs['Season']=season

a=Subs[['Rolling', 'Seasonal']].groupby(by=Subs['Season'])

#%%
import matplotlib.patches as mpatches
fig,ax=plt.subplots(figsize=(10,10),sharey='col',sharex=True)
plot=a.boxplot(showfliers=False, showmeans=True,ax=ax,fontsize=16,layout=(2,2),return_type='both',
               patch_artist = True,meanprops=dict(marker='o',markerfacecolor='black',markeredgecolor='black'),
               medianprops=dict(color='black'),whiskerprops=dict(color='black'))
colors = ['indianred', 'cornflowerblue' ]
for row_key, (ax,row) in plot.iteritems():
    ax.set_xlabel('','')
    ax.title.set_size(20)
    for i,box in enumerate(row['boxes']):
        box.set_facecolor(colors[i])

plt.show()
#%%
#Let's import all the sites
cities=['Barcelona', 'Bucharest', 'Cyprus','Dublin','Lille', 'Magadino','Marseille','SIRTA','Tartu']
city_acr=['BCN','BUC',  'CYP', 'DUB', 'LIL', 'MAG', 'MAR','SIR','TAR']
wdw_l=14    
data=[]
data_list=[]
for city_i in cities:
    print(city_i)
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i
    os.chdir(path)
    a=os.listdir()
    data.append(pd.read_csv(city_i+'.txt', sep="\t", low_memory=False)) #List of the datasets
    #Sub_list_6044.append(pd.read_csv('Substr_6044.txt', sep="\t", low_memory=False))
#%% We create a whole dataset with the info we might need
df_list=[]
for i in range(0,len(data)):
    df=pd.DataFrame()
    df['datetime']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    city=city_acr[i]
    print(city)
    df['HOA_Rolling']=data[i]['HOA_Rolling']
    df['HOA_seas']=data[i]['HOA_seas']
    if city=='BCN' or city=='MAR':
        df['COA_Rolling']=data[i]['COA_Rolling']
        df['COA_seas']=data[i]['COA_seas']
    if city=='MAG':
        df['LOA_Rolling']=data[i]['LOA_Rolling']
        df['LOA_seas']=data[i]['LOA_seas']
    if city=='MAR':
        df['SHINDOA_Rolling']=data[i]['SHINDOA_Rolling']
        df['SHINDOA_seas']=data[i]['SHINDOA_seas']
    if city!='DUB':
        df['BBOA_Rolling']=data[i]['BBOA_Rolling']
        df['BBOA_seas']=data[i]['BBOA_seas']
    if city=='DUB':
        df['Wood_Rolling']=data[i]['Wood_Rolling']
        df['Wood_seas']=data[i]['Wood_seas']
        df['Peat_Rolling']=data[i]['Peat_Rolling']
        df['Peat_seas']=data[i]['Peat_seas']
        df['Coal_Rolling']=data[i]['Coal_Rolling']
        df['Coal_seas']=data[i]['Coal_seas']
    df['LO-OOA_Rolling']=data[i]['LO-OOA_Rolling']
    df['LO-OOA_seas']=data[i]['LO-OOA_seas']
    df['MO-OOA_Rolling']=data[i]['MO-OOA_Rolling']
    df['MO-OOA_seas']=data[i]['MO-OOA_seas']
    # df['OOA_Rolling']=data[i]['OOA_Rolling']
    # df['OOA_seas']=data[i]['OOA_seas'] 
    df['Org']=data[i]['Org']
    # df['BCff']=data[i]['BCff']
    # df['BCwb']=data[i]['BCwb']
    df_list.append(df)    #list of dataframes with the convenient information.
df_to_csv=pd.concat([df_list[0], df_list[1], df_list[2], df_list[3], df_list[4], df_list[5],df_list[6],df_list[7]]) #concatenated df
path_whole="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i
os.chdir(path_whole)
df_to_csv.to_csv('Whole_dataset.txt',sep='\t')
#%%
df_to_csv['wb_ff']=df_to_csv['BCwb']/df_to_csv['BCff']
print(df_to_csv['BCwb_ff'])
#%%
df_to_csv['Wood_seas'].plot()
#%% We will create diurnals and monthly plots for the whole dataset.
# *******  MONTHLY  ********
df_plot=df_to_csv #We create a copy to leave the other one as originally was
df_plot['Month']=df_plot['datetime'].dt.month
df_plot.index = df_plot['Month'] 
df_month=df_plot.groupby(by=df_plot['Month']).mean()

#plot for main factors

fig, axs= plt.subplots(5, figsize=(10,20),sharex='all')
axs[0].plot(df_month['HOA_Rolling'], marker='o', color='grey')
axs[0].plot(df_month['HOA_seas'],marker='o', color='grey', linestyle='-.')
axs[0].set_title('Monthly differences all-sites'+'\n'+ '-- Rolling    -· Seasonal'+'\n'+ 'HOA', fontsize=18)
axs[0].grid()
axs[1].plot(df_month['BBOA_Rolling'], marker='o', color='sienna')
axs[1].plot(df_month['BBOA_seas'],marker='o', color='sienna', linestyle='-.')
axs[1].set_title('BBOA', fontsize=18)
axs[1].grid()
axs[2].plot(df_month['LO-OOA_Rolling'], marker='o', color='lightgreen')
axs[2].plot(df_month['LO-OOA_seas'],marker='o', color='lightgreen', linestyle='-.')
axs[2].set_title('LO-OOA', fontsize=18)
axs[2].grid()
axs[3].plot(df_month['MO-OOA_Rolling'], marker='o', color='darkgreen')
axs[3].plot(df_month['MO-OOA_seas'],marker='o', color='darkgreen', linestyle='-.')
axs[3].set_title('MO-OOA', fontsize=18)
axs[3].grid()
axs[4].plot(df_month['MO-OOA_Rolling']+df_month['LO-OOA_Rolling'], marker='o', color='green')
axs[4].plot(df_month['MO-OOA_seas']+df_month['LO-OOA_seas'],marker='o', color='green', linestyle='-.')
axs[4].set_title('OOA', fontsize=18)
axs[4].grid()
axs[2].set_ylabel('Concentration ($μg·m^{-3}$)', fontsize=20)
axs[4].set_xlabel('Month', fontsize=18)
axs[4].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
axs[4].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=16)

# #plot for extra factors

# fig, axs= plt.subplots(6, figsize=(10,20),sharex='all')
# axs[0].plot(df_month['COA_Rolling'], marker='o', color='mediumpurple')
# axs[0].plot(df_month['COA_seas'],marker='o', color='mediumpurple', linestyle='-.')
# axs[0].set_title('Monthly differences all-sites'+'\n'+ '-- Rolling    -· Seasonal'+'\n'+ 'COA', fontsize=18)
# axs[0].grid()
# axs[1].plot(df_month['LOA_Rolling'], marker='o', color='gold')
# axs[1].plot(df_month['LOA_seas'],marker='o', color='gold', linestyle='-.')
# axs[1].set_title('LOA', fontsize=18)
# axs[1].grid()
# axs[2].plot(df_month['SHINDOA_Rolling'], marker='o', color='indianred')
# axs[2].plot(df_month['SHINDOA_seas'],marker='o', color='indianred', linestyle='-.')
# axs[2].set_title('SHINDOA', fontsize=18)
# axs[2].grid()
# axs[3].plot(df_month['Wood_Rolling'], marker='o', color='burlywood')
# axs[3].plot(df_month['Wood_seas'],marker='o', color='burlywood', linestyle='-.')
# axs[3].set_title('WCOA', fontsize=18)
# axs[3].grid()
# axs[4].plot(df_month['Coal_Rolling'], marker='o', color='peru')
# axs[4].plot(df_month['Coal_seas'],marker='o', color='peru', linestyle='-.')
# axs[4].set_title('CCOA', fontsize=18)
# axs[4].grid()
# axs[5].plot(df_month['Peat_Rolling'], marker='o', color='saddlebrown')
# axs[5].plot(df_month['Peat_seas'],marker='o', color='saddlebrown', linestyle='-.')
# axs[5].set_title('PCOA', fontsize=18)
# axs[5].grid()
# axs[3].set_ylabel('\t'+'               Concentration ($μg·m^{-3}$)', fontsize=20)
# axs[5].set_xlabel('Month', fontsize=18)
# axs[5].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# axs[5].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=16)
# # #%%

#%% We will create diurnals plots for the whole dataset.
# *******  DIEL  ********
df_plot=df_to_csv #We create a copy to leave the other one as originally was
df_plot['Hour']=df_plot['datetime'].dt.hour
df_plot.index = df_plot['Hour'] 
df_hour=df_plot.groupby(by=df_plot['Hour']).mean()

#plot for main factors

fig, axs= plt.subplots(5, figsize=(10,20),sharex='all')
axs[0].plot(df_hour['HOA_Rolling'], marker='o', color='grey')
axs[0].plot(df_hour['HOA_seas'],marker='o', color='grey', linestyle='-.')
axs[0].set_title('Hourly differences all-sites'+'\n'+ '-- Rolling    -· Seasonal'+'\n'+ 'HOA', fontsize=18)
axs[0].grid()
axs[1].plot(df_hour['BBOA_Rolling'], marker='o', color='sienna')
axs[1].plot(df_hour['BBOA_seas'],marker='o', color='sienna', linestyle='-.')
axs[1].set_title('BBOA', fontsize=18)
axs[1].grid()
axs[2].plot(df_hour['LO-OOA_Rolling'], marker='o', color='lightgreen')
axs[2].plot(df_hour['LO-OOA_seas'],marker='o', color='lightgreen', linestyle='-.')
axs[2].set_title('LO-OOA', fontsize=18)
axs[2].grid()
axs[3].plot(df_hour['MO-OOA_Rolling'], marker='o', color='darkgreen')
axs[3].plot(df_hour['MO-OOA_seas'],marker='o', color='darkgreen', linestyle='-.')
axs[3].set_title('MO-OOA', fontsize=18)
axs[3].grid()
axs[4].plot(df_hour['MO-OOA_Rolling']+df_hour['LO-OOA_Rolling'], marker='o', color='green')
axs[4].plot(df_hour['MO-OOA_seas']+df_hour['LO-OOA_seas'],marker='o', color='green', linestyle='-.')
axs[4].set_title('OOA', fontsize=14)
axs[4].grid()
axs[2].set_ylabel('Concentration ($μg·m^{-3}$)', fontsize=20)
axs[4].set_xlabel('Hour', fontsize=18)
axs[4].set_xticks([0,4,8,12,16,20,24])
axs[4].set_xticklabels(['0', '4', '8','12','18', '20', '24'], fontsize=16)

#plot for extra factors

# fig, axs= plt.subplots(6, figsize=(10,20),sharex='all')
# axs[0].plot(df_hour['COA_Rolling'], marker='o', color='mediumpurple')
# axs[0].plot(df_hour['COA_seas'],marker='o', color='mediumpurple', linestyle='-.')
# axs[0].set_title('Hourly differences all-sites'+'\n'+ '-- Rolling    -· Seasonal'+'\n'+ 'COA', fontsize=18)
# axs[0].grid()
# axs[1].plot(df_hour['LOA_Rolling'], marker='o', color='gold')
# axs[1].plot(df_hour['LOA_seas'],marker='o', color='gold', linestyle='-.')
# axs[1].set_title('LOA', fontsize=18)
# axs[1].grid()
# axs[2].plot(df_hour['SHINDOA_Rolling'], marker='o', color='indianred')
# axs[2].plot(df_hour['SHINDOA_seas'],marker='o', color='indianred', linestyle='-.')
# axs[2].set_title('SHINDOA', fontsize=18)
# axs[2].grid()
# axs[3].plot(df_hour['Wood_Rolling'], marker='o', color='burlywood')
# axs[3].plot(df_hour['Wood_seas'],marker='o', color='burlywood', linestyle='-.')
# axs[3].set_title('WCOA', fontsize=18)
# axs[3].grid()
# axs[4].plot(df_hour['Coal_Rolling'], marker='o', color='peru')
# axs[4].plot(df_hour['Coal_seas'],marker='o', color='peru', linestyle='-.')
# axs[4].set_title('CCOA', fontsize=18)
# axs[4].grid()
# axs[5].plot(df_hour['Peat_Rolling'], marker='o', color='saddlebrown')
# axs[5].plot(df_hour['Peat_seas'],marker='o', color='saddlebrown', linestyle='-.')
# axs[5].set_title('PCOA', fontsize=18)
# axs[5].grid()
# axs[3].set_ylabel('\t'+'               Concentration ($μg·m^{-3}$)', fontsize=20)
# axs[5].set_xlabel('Hour', fontsize=18)
# axs[5].set_xticks([0,4,8,12,16,20,24])
# axs[5].set_xticklabels(['0', '4', '8','12','18', '20', '24'], fontsize=16)
#%%
path_whole="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path_whole)
df_hour.to_csv('Factors_diel.txt', sep='\t')
df_month.to_csv('Factors_monthly.txt', sep='\t')
#%%
city='BCN'
f1=df_to_csv
f1['Time']=f1['datetime']
dr_all=pd.date_range("2013/08/01 00:00", end="2018/11/01")
df_to_csv['OOA_Rolling']=df_to_csv['MO-OOA_Rolling']+df_to_csv['LO-OOA_Rolling']
df_to_csv['OOA_seas']=df_to_csv['MO-OOA_seas']+df_to_csv['LO-OOA_seas']
df_season=Factors_averaging('Abs',90)
df_biweek=Factors_averaging('Abs',14)
df_day=Factors_averaging('Abs',1)

# df_season=(data_averager(df_to_csv,df_to_csv['datetime'], dr_all,90,1).outdata)
# df_biweek=(data_averager(df_to_csv,df_to_csv['datetime'], dr_all,14,1).outdata)
# df_day=(data_averager(df_to_csv,df_to_csv['datetime'], dr_all,1,1).outdata)
#%%T-Test
from scipy.stats import ttest_ind
per='Period'
if per=='Period':
    # print('COA:',ttest_ind(df_to_csv['COA_Rolling'], df_to_csv['COA_Season'], nan_policy='omit', equal_var=False))
    print('HOA:',ttest_ind(df_to_csv['HOA_Rolling'], df_to_csv['HOA_seas'], nan_policy='omit', equal_var=False))
    print('BBOA:',ttest_ind(df_to_csv['BBOA_Rolling'], df_to_csv['BBOA_seas'], nan_policy='omit', equal_var=False))
    # print('LOA:',ttest_ind(df_to_csv['LOA_Rolling'], df_to_csv['LOA_seas'], nan_policy='omit', equal_var=False))
    # print('SHINDOA:',ttest_ind(df_to_csv['SHINDOA_Rolling'], df_to_csv['SHINDOA_seas'], nan_policy='omit', equal_var=False))
    # print('Wood:',ttest_ind(df_to_csv['Wood_Rolling'], df_to_csv['Wood_seas'], nan_policy='omit', equal_var=False))
    # print('Coad:',ttest_ind(df_to_csv['Coal_Rolling'], df_to_csv['Coal_seas'], nan_policy='omit', equal_var=False))
    # print('Peat:',ttest_ind(df_to_csv['Peat_Rolling'], df_to_csv['Peat_seas'], nan_policy='omit', equal_var=False))
    print('LO-OOA:',ttest_ind(df_to_csv['LO-OOA_Rolling'], df_to_csv['LO-OOA_seas'], nan_policy='omit', equal_var=False))
    print('MO-OOA:',ttest_ind(df_to_csv['MO-OOA_Rolling'], df_to_csv['MO-OOA_seas'], nan_policy='omit', equal_var=False))
    print('OOA:',ttest_ind(df_to_csv['OOA_Rolling'], df_to_csv['OOA_seas'], nan_policy='omit', equal_var=False))
if per!='Period':
    if per=='Season':
        gen_df=df_season
    if per=='Biweek':
        gen_df=df_biweek
    if per=='Day':
        gen_df=df_day
    print('COA:',ttest_ind(gen_df['COA_Rolling'], gen_df['COA_Seasonal'], nan_policy='omit', equal_var=False))
    print('HOA:',ttest_ind(gen_df['HOA_Rolling'], gen_df['HOA_Seasonal'], nan_policy='omit', equal_var=False))
    print('BBOA:',ttest_ind(gen_df['BBOA_Rolling'], gen_df['BBOA_Seasonal'], nan_policy='omit', equal_var=False))
    # print('LOA:',ttest_ind(gen_df['LOA_Rolling'], gen_df['LOA_Seasonal'], nan_policy='omit', equal_var=False))
    # print('SHINDOA:',ttest_ind(gen_df['SHINDOA_Rolling'], gen_df['SHINDOA_Seasonal'], nan_policy='omit', equal_var=False))
    # print('Wood:',ttest_ind(gen_df['Wood_Rolling'], gen_df['Wood_Seasonal'], nan_policy='omit', equal_var=False))
    # print('Coad:',ttest_ind(gen_df['Coal_Rolling'], gen_df['Coal_Seasonal'], nan_policy='omit', equal_var=False))
    # print('Peat:',ttest_ind(gen_df['Peat_Rolling'], gen_df['Peat_Seasonal'], nan_policy='omit', equal_var=False))
    print('LO-OOA:',ttest_ind(gen_df['LO-OOA_Rolling'], gen_df['LO-OOA_Seasonal'], nan_policy='omit', equal_var=False))
    print('MO-OOA:',ttest_ind(gen_df['MO-OOA_Rolling'], gen_df['MO-OOA_Seasonal'], nan_policy='omit', equal_var=False))
    print('OOA:',ttest_ind(gen_df['OOA_Rolling'], gen_df['OOA_Seasonal'], nan_policy='omit', equal_var=False))

#%% Rolling - Seasonal monthly BP plots
df_list=[]
for i in range(0,len(data)):
    Dif=pd.DataFrame()
    Dif['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    city=city_acr[i]
    print(city)
    Dif['HOA']=data[i]['HOA_Rolling'] - data[i]['HOA_seas']
    if city=='BCN' or city=='MAR':
        Dif['COA']=data[i]['COA_Rolling'] - data[i]['COA_seas']
    if city=='MAG':
        Dif['LOA']=data[i]['LOA_Rolling'] - data[i]['LOA_seas']
    if city=='MAR':
        Dif['SHINDOA']=data[i]['SHINDOA_Rolling'] - data[i]['SHINDOA_seas']
    if city!='DUB':
        Dif['BBOA']=data[i]['BBOA_Rolling'] - data[i]['BBOA_seas']
    if city=='DUB':
        Dif['Wood']=data[i]['Wood_Rolling'] - data[i]['Wood_seas']
        Dif['Peat']=data[i]['Peat_Rolling'] - data[i]['Peat_seas']
        Dif['Coal']=data[i]['Coal_Rolling'] - data[i]['Coal_seas']
    Dif['LO-OOA']=data[i]['LO-OOA_Rolling'] - data[i]['LO-OOA_seas']
    Dif['MO-OOA']=data[i]['MO-OOA_Rolling'] - data[i]['MO-OOA_seas']
    Dif['OOA']= data[i]['MO-OOA_Rolling'] +data[i]['LO-OOA_Rolling'] - data[i]['MO-OOA_seas'] - data[i]['LO-OOA_seas']
    # Dif['OOA']=data[i]['OOA_Rolling'] - data[i]['OOA_seas']   
    Dif['Month']=Dif.Time.dt.month
    Dif['Time'].drop(index=1, inplace=True)
    Dif.boxplot(by='Month',showfliers=False, showmeans=True,figsize=(15,15),fontsize=20)
    plt.suptitle(cities[i], fontsize=25) # that's what you're after
    plt.show()
    df_list.append(Dif)   
#%%
a=pd.concat([df_list[0], df_list[1], df_list[2], df_list[3], df_list[4], df_list[5],df_list[6],df_list[7]])
cols=['HOA', 'BBOA', 'MO-OOA', 'LO-OOA', 'OOA', 'Month']
# cols=['COA', 'LOA', 'SHINDOA', 'Wood','Coal','Peat','Month']
a=a[cols]
plt.title('Rolling - Seasonal')
plt.ylabel('Rolling - Seasonal ($\mu g·m^{-3}$')
#%%
import matplotlib.patches as mpatches
boxprops = dict(linestyle='-', linewidth=0.6,edgecolor='black',facecolor='white')
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
medianprops=dict( linewidth=0.6, color='black')
whiskerprops=dict( linewidth=0.6, color='black')
fig,axes=plt.subplots(len(columns),1, figsize=(10,20),sharex='all')
columns = ['HOA', 'BBOA', 'LO-OOA', 'MO-OOA', 'OOA']
# columns=['COA', 'LOA', 'SHINDOA', 'WCOA','CCOA','PCOA','Month']
for i in range(len(columns)):
    b=pd.DataFrame()
    b[cols[i]]=a[cols[i]]
    b['Month']=a['Month']
    box=b.boxplot(by='Month', showfliers=False, showmeans=True, figsize=(10,20),ax=axes[i],
              patch_artist=True,boxprops=boxprops, meanprops=meanprops, 
              medianprops=medianprops,whiskerprops=whiskerprops, fontsize=18)
    axes[i].set_xlabel('')
    axes[i].set_title(columns[i], fontsize=20)
    axes[i].set_ylabel('Rolling minus Seasonal '+'\n'+'($\mu g·m^{-3}$)', fontsize=14)
fig.suptitle('')
zxcvbnm,
# ax[3].set_xlabel('Month', fontsize=18)
axes[4].set_xticks(range(0,13))
labels=['','J','F','M','A','M','J','J','A','S','O','N','D',]
# axes[3].set_xticklabels(labels)

#%%
absR_list=[]
absS_list=[]

for i in range(0,len(data)):
    Abs_R=pd.DataFrame()
    Abs_S=pd.DataFrame()
    Abs_R['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    Abs_S['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    city=city_acr[i]
    Abs_R['HOA']=data[i]['HOA_Rolling']
    Abs_S['HOA']=data[i]['HOA_seas']
    if city=='BCN' or city=='MAR':
        Abs_R['COA']=data[i]['COA_Rolling']
        Abs_S['COA']=data[i]['COA_seas']
    if city=='MAG':
        Abs_R['LOA']=data[i]['LOA_Rolling']
        Abs_S['LOA']=data[i]['LOA_seas']
    if city=='MAR':
        Abs_R['SHINDOA']=data[i]['SHINDOA_Rolling']
        Abs_S['SHINDOA']=data[i]['SHINDOA_seas']   
    if city!='DUB':
        Abs_R['BBOA']=data[i]['BBOA_Rolling']
        Abs_S['BBOA']=data[i]['BBOA_seas']        
    if city=='DUB':
        Abs_R['Wood']=data[i]['Wood_Rolling']
        Abs_R['Peat']=data[i]['Peat_Rolling']
        Abs_R['Coal']=data[i]['Coal_Rolling']
        Abs_S['Wood']=data[i]['Wood_seas']
        Abs_S['Peat']=data[i]['Peat_seas']
        Abs_S['Coal']=data[i]['Coal_seas']
    Abs_R['LO-OOA']=data[i]['LO-OOA_Rolling']
    Abs_R['MO-OOA']=data[i]['MO-OOA_Rolling']
    Abs_R['OOA']=data[i]['MO-OOA_Rolling']+data[i]['LO-OOA_Rolling']
    Abs_S['LO-OOA']=data[i]['LO-OOA_seas']
    Abs_S['MO-OOA']=data[i]['MO-OOA_seas']
    Abs_S['OOA']=data[i]['MO-OOA_seas']+data[i]['LO-OOA_seas']
    # Abs['OOA']=data[i]['OOA_Rolling'] - data[i]['OOA_seas']   
    Abs_R['Time'].drop(index=1, inplace=True)
    Abs_S['Time'].drop(index=1, inplace=True)
    absR_list.append(Abs_R)  
    absS_list.append(Abs_S)   

#%%Same for rel

relR_list=[]
relS_list=[]

for i in range(0,len(data)):
    rel_R=pd.DataFrame()
    rel_S=pd.DataFrame()
    rel_R['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    rel_S['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    city=city_acr[i]
    prova1="OA_app_S" in data[i]
    prova2='OA_app_s' in data[i] 
    prova3='OA_app_R' in data[i]
    if prova3==False:
        data[i]['OA_app_R']=data[i]['OA_app_Rolling']
    if prova1==False :
        if prova2==False:
            data[i]['OA_app_S']=data[i]['OA_app_seas']
        if prova2==True:
            data[i]['OA_app_S']=data[i]['OA_app_s']

    rel_R['HOA']=data[i]['HOA_Rolling']/data[i]['OA_app_R']
    rel_S['HOA']=data[i]['HOA_seas']/data[i]['OA_app_S']
    if city=='BCN' or city=='MAR':
        rel_R['COA']=data[i]['COA_Rolling']/data[i]['OA_app_R']
        rel_S['COA']=data[i]['COA_seas']/data[i]['OA_app_S']
    if city=='MAG':
        rel_R['LOA']=data[i]['LOA_Rolling']/data[i]['OA_app_R']
        rel_S['LOA']=data[i]['LOA_seas']/data[i]['OA_app_S']
    if city=='MAR':
        rel_R['SHINDOA']=data[i]['SHINDOA_Rolling']/data[i]['OA_app_R']
        rel_S['SHINDOA']=data[i]['SHINDOA_seas']   /data[i]['OA_app_S']
    if city!='DUB':
        rel_R['BBOA']=data[i]['BBOA_Rolling']/data[i]['OA_app_R']
        rel_S['BBOA']=data[i]['BBOA_seas']       /data[i]['OA_app_S'] 
    if city=='DUB':
        rel_R['Wood']=data[i]['Wood_Rolling']/data[i]['OA_app_R']
        rel_R['Peat']=data[i]['Peat_Rolling']/data[i]['OA_app_R']
        rel_R['Coal']=data[i]['Coal_Rolling']/data[i]['OA_app_R']
        rel_S['Wood']=data[i]['Wood_seas']/data[i]['OA_app_S']
        rel_S['Peat']=data[i]['Peat_seas']/data[i]['OA_app_S']
        rel_S['Coal']=data[i]['Coal_seas']/data[i]['OA_app_S']
    rel_R['LO-OOA']=data[i]['LO-OOA_Rolling']/data[i]['OA_app_R']
    rel_R['MO-OOA']=data[i]['MO-OOA_Rolling']/data[i]['OA_app_R']
    rel_R['OOA']=(data[i]['MO-OOA_Rolling']+data[i]['LO-OOA_Rolling'])/data[i]['OA_app_R']
    rel_S['LO-OOA']=data[i]['LO-OOA_seas']/data[i]['OA_app_S']
    rel_S['MO-OOA']=data[i]['MO-OOA_seas']/data[i]['OA_app_S']
    rel_S['OOA']=data[i]['MO-OOA_seas']+data[i]['LO-OOA_seas']
    # rel['OOA']=data[i]['OOA_Rolling'] - data[i]['OOA_seas']   
    rel_R['Time'].drop(index=1, inplace=True)
    rel_S['Time'].drop(index=1, inplace=True)
    relR_list.append(rel_R)  
    relS_list.append(rel_S)   

#%%
abs_R=pd.concat([absR_list[0], absR_list[1], absR_list[2], absR_list[3], absR_list[4], absR_list[5],absR_list[6],absR_list[7]])
abs_S=pd.concat([absS_list[0], absS_list[1], absS_list[2], absS_list[3], absS_list[4], absS_list[5],absS_list[6],absS_list[7]])
# abs_R.to_csv('Abs_R.txt', sep='\t')
# abs_S.to_csv('Abs_S.txt', sep='\t')
# cols=['COA','LOA', 'SHINDOA','Wood','Coal','Peat']
cols=['HOA','BBOA', 'LO-OOA','MO-OOA','OOA']
abs_R=abs_R[cols]
abs_S=abs_S[cols]
# abs_R.plot.pie()
#%%
rel_R=pd.concat([relR_list[0], relR_list[1], relR_list[2], relR_list[3], relR_list[4], relR_list[5],relR_list[6],relR_list[7]])
rel_S=pd.concat([relS_list[0], relS_list[1], relS_list[2], relS_list[3], relS_list[4], relS_list[5],relS_list[6],relS_list[7]])
cols=['HOA','BBOA', 'LO-OOA','MO-OOA','OOA']
rel_R=rel_R[cols]
rel_S=rel_S[cols]
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path)
rel_R.to_csv('Rel_extra_R.txt', sep='\t')
rel_S.to_csv('Rel_extra_S.txt', sep='\t')
#%%

#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path)
abs_R.to_csv('Abs_extra_R.txt', sep='\t')
abs_S.to_csv('Abs_extra_S.txt', sep='\t')
# a=pd.concat([dif_list[0], dif_list[1], dif_list[2], dif_list[3], dif_list[4], dif_list[5],dif_list[6],dif_list[7]])
# cols=['COA', 'LOA','SHINDOA', 'Wood','Coal', 'Peat']
#%%
plt.title('Rolling - Seasonal')
plt.ylabel('Rolling - Seasonal ($\mu g·m^{-3}$')
import matplotlib.patches as mpatches
boxprops = dict(linestyle='-', linewidth=0.6,edgecolor='black',facecolor='white')
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
medianprops=dict( linewidth=0.6, color='black')
whiskerprops=dict( linewidth=0.6, color='black')
fig,axes=plt.subplots(5,1, figsize=(10,20),sharex='all')
for i in range(5):
    b=pd.DataFrame()
    b[cols[i]]=a[cols[i]]
    b['Month']=a['Month']
    box=b.boxplot(by='Month', showfliers=False, showmeans=True, figsize=(10,20),ax=axes[i],
              patch_artist=True,boxprops=boxprops, meanprops=meanprops, 
              medianprops=medianprops,whiskerprops=whiskerprops, fontsize=15)
    axes[i].set_xlabel('')
    axes[i].set_title(cols[i], fontsize=20)
    axes[i].set_ylabel('Rolling minus Seasonal '+'\n'+'($\mu g·m^{-3}$)', fontsize=18)
fig.suptitle('')
ax[3].set_xlabel('Month', fontsize=18)
axes[3].set_xticks(range(0,13))
labels=['','J','F','M','A','M','J','J','A','S','O','N','D',]
axes[3].set_xticklabels(labels)
#%%
abs_R=pd.concat([absR_list[0], absR_list[1], absR_list[2], absR_list[3], absR_list[4], absR_list[5],absR_list[6],absR_list[7]])
abs_S=pd.concat([absS_list[0], absS_list[1], absS_list[2], absS_list[3], absS_list[4], absS_list[5],absS_list[6],absS_list[7]])
abs_R.to_csv('Abs_R.txt', sep='\t')
abs_S.to_csv('Abs_S.txt', sep='\t')
cols=['HOA', 'BBOA', 'MO-OOA', 'LO-OOA', 'OOA']
abs_R=abs_R[cols]
abs_S=abs_S[cols]
# abs_R.plot.pie()
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path)
abs_R.to_csv('Abs_some_R.txt', sep='\t')
abs_S.to_csv('Abs_some_S.txt', sep='\t')
#%%
abs_R['HOA'].plot.bar()
#%%
rel_R=pd.read_csv('Rel_R.txt', sep='\t')
rel_S=pd.read_csv('Rel_S.txt', sep='\t')
print('Rolling: ', rel_R.mean(axis=0, skipna=True))
print('Seasonal: ', rel_R.mean(axis=0, skipna=True))
#%%
my_labels='HOA','BBOA', 'LO-OOA','MO-OOA' #,' Wood', 'Coal','Peat'
rel_R_m=pd.Series([0.1173,	0.1717,	0.3013,	0.4803])
rel_S_m=pd.Series([0.1270,	0.1423,	0.2642,	0.5003])
#%%
#fig.suptitle(cityname,fontsize=30, y=0.75)
fig, axs = plt.subplots(1,2, figsize=(15,15))

axs[0].pie(rel_R_m,labels=my_labels,autopct='%1.0f%%', 
           colors=['grey','brown','limegreen','darkgreen'], textprops={'fontsize': 25})#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[0].set_title('Rolling',fontsize=28)
axs[1].pie(rel_S_m, labels=my_labels,  autopct='%1.0f%%',
           colors=['grey', 'brown','limegreen','darkgreen' ],textprops={'fontsize': 25})#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1].set_title('Seasonal', fontsize=28)
        
    
    
    
    
    #%%
#%%
#Let's import all the sites
cities=['Barcelona', 'Bucharest', 'Cyprus','Dublin','Lille', 'Magadino','Marseille','Tartu']
city_acr=['BCN','BUC','CYP', 'DUB', 'LIL', 'MAG', 'MAR','TAR', 'BCN']
wdw_l=14    
data=[]
data_list=[]
for city_i in cities:
    print(city_i)
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i
    os.chdir(path)
    a=os.listdir()
    data.append(pd.read_csv(city_i+'.txt', sep="\t", low_memory=False)) #List of the datasets
    #Sub_list_6044.append(pd.read_csv('Substr_6044.txt', sep="\t", low_memory=False))
#%% We create a whole dataset with the info we might need
print(type(data[0]['mz57_R'][564]))
#%%
df_list=[]
for i in range(0,len(data)):
    df=pd.DataFrame()
    df['datetime']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
    city=city_acr[i]
    print(city)
    df['mz44']=data[i]['mz44']
    df['mz43']=data[i]['mz43']
    df['mz55']=data[i]['mz55']
    df['mz57']=data[i]['mz57']
    df['mz60']=data[i]['mz60']
    df['mz73']=data[i]['mz73']
    df['mz43_R']=data[i]['mz43_R']
    df['mz43_S']=data[i]['mz43_S']
    df['mz44_R']=data[i]['mz44_R']
    df['mz44_S']=data[i]['mz44_S']
    df['mz55_R']=data[i]['mz55_R']
    df['mz55_S']=data[i]['mz55_S'] 
    df['mz57_R']=data[i]['mz57_R']
    df['mz57_S']=data[i]['mz57_S']      
    df['mz60_R']=data[i]['mz60_R']
    df['mz60_S']=data[i]['mz60_S']    
    df['mz73_R']=data[i]['mz73_R']
    df['mz73_S']=data[i]['mz73_S']
    df_list.append(df)    #list of dataframes with the convenient information.
df_to_csv=pd.concat([df_list[0], df_list[1], df_list[2], df_list[3], df_list[4], df_list[5],df_list[6],df_list[7]]) #concatenated df
path_whole="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path_whole)
df_to_csv.to_csv('Whole_dataset_ions.txt',sep='\t')    
    #%%
path_whole="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole"
os.chdir(path_whole)
ions=pd.read_csv('Whole_ions.txt', sep='\t', low_memory=False)
ion='mz60'
dif['Rolling']=ions[ion]-ions[ion+'_R']
dif['Seasonal']=ions[ion]-ions[ion+'_S']
dif.boxplot(showfliers=False,showmeans=True)
#%%
boxprops = dict(linestyle='-', linewidth=0.8,color='black')  
whiskerprops=dict(linestyle='-', linewidth=0.8,color='black')  
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
fig, ax = plt.subplots(figsize=(6,6)) 
#%%

fig, ax = plt.subplots(1,3,figsize=(20,10)) 
plt.tick_params(labelsize=16)
ion1='mz43'
ion2='mz44'
dif_ratio=pd.DataFrame()
dif_ratio['Raw']=ions[ion1]/ions[ion2]
dif_ratio['Rolling']=ions[ion1+'_R']/ions[ion2+'_R']
dif_ratio['Seasonal']=ions[ion1+'_S']/ions[ion2+'_S']
dif_ratio.boxplot(ax=ax[0], showfliers=False, showmeans=True, fontsize=18,boxprops=boxprops,meanprops=meanprops, color='black')
ax[0].set_title(ion1+' / '+ion2, fontsize=18)
ax[0].set_ylabel('(adim.)',fontsize=18)
ion1='mz60'
ion2='mz44'
dif_ratio=pd.DataFrame()
dif_ratio['Raw']=ions[ion1]/ions[ion2]
dif_ratio['Rolling']=ions[ion1+'_R']/ions[ion2+'_R']
dif_ratio['Seasonal']=ions[ion1+'_S']/ions[ion2+'_S']
# ax[2].boxplot(dif_ratio, showfliers=False)
dif_ratio.boxplot(ax=ax[1],showfliers=False, showmeans=True, fontsize=18,boxprops=boxprops,meanprops=meanprops, color='black')
ax[1].set_title(ion1+' / '+ion2, fontsize=18)
ion1='mz57'
ion2='mz44'
# ax[0].set_xticklabels(fontsize=14)
dif_ratio=pd.DataFrame()
dif_ratio['Raw']=ions[ion1]/ions[ion2]
dif_ratio['Rolling']=ions[ion1+'_R']/ions[ion2+'_R']
dif_ratio['Seasonal']=ions[ion1+'_S']/ions[ion2+'_S']
dif_ratio.boxplot(ax=ax[2],showfliers=False, showmeans=True, fontsize=18,boxprops=boxprops,meanprops=meanprops, color='black')
ax[2].set_title(ion1+' / '+ion2, fontsize=18)
#
#
#%%Transition period import and concatenation
transition=pd.DataFrame()
cities=['Barcelona', 'Cyprus', 'Dublin', 'Lille', 'Magadino', 'Bucharest','Marseille', 'SIRTA','Tartu']
city_acr=['BCN','CAO','DUB','ATOLL','MAG','INO','MRS','SIR','TAR']
city_names=['Barcelona - Palau Reial', 'Cyprus Atm. Obs. - Agia Xyliatou', 'Dublin', 'Lille', 'Magadino', 'Magurele-INOE','Marseille - Longchamp', 'SIRTA', 'Tartu']
for city_i in cities:
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/'+city_i+'/')
    file_i=pd.read_csv('f1_transition.txt', sep="\t", low_memory=False)
    transition=pd.concat([transition,file_i])
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole')
transition.to_csv('Transition.txt', sep='\t')
#%%Transition period: Sc Residuals
fig,ax=plt.subplots(figsize=(5,5))
ax.set_title('Whole dataset'+'\n'+'TRANSITION PERIOD', fontsize=15)
boxprops = dict(linestyle='-', linewidth=0.6)
whiskerprops = dict(linestyle='-', linewidth=0.6)
medianprops = dict(linestyle='-', linewidth=0.6,color='black')
meanprops = dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
transition[['Sc Res_R','Sc Res_S']].boxplot(showfliers=False,medianprops=medianprops,meanprops=meanprops, showmeans=True,ax=ax, boxprops=boxprops,whiskerprops=whiskerprops)
ax.set_xticklabels(['Rolling', 'Seasonal'], fontsize=14)
ax.set_ylabel('Scaled Residuals ($μg·m^{-3}$)', fontsize=15)
#%%
fig,ax=plt.subplots(figsize=(5,5))
transition[['Sc Res_R', 'Sc Res_S']].plot.kde(alpha=0.5,color=['red', 'blue'],grid=True,subplots=False,ax=ax,legend=False,fontsize=11)
ax.legend(['Rolling', 'Seasonal'],fontsize=13)
ax.set_title('Scaled Residuals'+'\n'+'TRANSITION PERIOD',fontsize=14)
# ax.set_xlim(-3,3)
# ax.text(-2.5,2,'(b)', fontsize=15)
ax.text(-22,2,'(a)', fontsize=15)
ax.set_ylabel('Density', fontsize=13)
#%% Transition period
print(R2(transition['HOA_Rolling'], transition['BCff']).round(2), R2(transition['HOA_seas'], transition['BCff']).round(2))
print(R2(transition['HOA_Rolling'], transition['Nox']).round(2), R2(transition['HOA_seas'], transition['Nox']).round(2))
print(R2(transition['BBOA_Rolling'], transition['BCwb']).round(2),R2(transition['BBOA_seas'], transition['BCwb']).round(2))
print(R2(transition['LO-OOA_Rolling'], transition['NO3']).round(2),R2(transition['LO-OOA_seas'], transition['NO3']).round(2))
print(R2(transition['MO-OOA_Rolling'], transition['SO4']).round(2),R2(transition['MO-OOA_seas'], transition['SO4']).round(2))
print(R2(transition['OOA_Rolling'], transition['NH4']).round(2),R2(transition['OOA_seas'], transition['NH4']).round(2))

#%% PREPARATION OF ALL FACTORS IN A DF
Dif2=pd.DataFrame()
df_to_csv=df_to_csv.reset_index(drop=True)
Dif2['COA']=df_to_csv['COA_Rolling']-df_to_csv['COA_seas']
Dif2['SHINDOA']=df_to_csv['SHINDOA_Rolling']-df_to_csv['SHINDOA_seas']
Dif2['LOA']=df_to_csv['LOA_Rolling']-df_to_csv['LOA_seas']
Dif2['WCOA']=df_to_csv['Wood_Rolling']-df_to_csv['Wood_seas']
Dif2['CCOA']=df_to_csv['Coal_Rolling']-df_to_csv['Coal_seas']
Dif2['PCOA']=df_to_csv['Peat_Rolling']-df_to_csv['Peat_seas']
Dif2['Time']=pd.to_datetime(df_to_csv['datetime'])
Dif2['Hour']=Dif2.Time.dt.hour
Dif2['Month']=Dif2.Time.dt.month
# Dif2['Time'].drop(index=1, inplace=True)
# Dif2.reset_index(range(0,len(Dif2)), inplace=True)
#%%
Dif2=pd.DataFrame()
# Dif2=Dif2.reset_index()
Dif2['Time']=pd.to_datetime(data[i]['datetime'], dayfirst=True, errors='coerce')
Dif2['Hour']=Dif2.Time.dt.hour
Dif2['Month']=Dif2.Time.dt.month
Dif2['Time'].drop(index=1, inplace=True)
Dif2[Dif2.index.duplicated()]
#%%
Dif2=pd.DataFrame()
Dif2['HOA']=df_to_csv['HOA_Rolling']-df_to_csv['HOA_seas']
Dif2['BBOA']=df_to_csv['BBOA_Rolling']-df_to_csv['BBOA_seas']
Dif2['LO-OOA']=df_to_csv['LO-OOA_Rolling']-df_to_csv['LO-OOA_seas']
Dif2['MO-OOA']=df_to_csv['MO-OOA_Rolling']-df_to_csv['MO-OOA_seas']
Dif2['OOA']=df_to_csv['MO-OOA_Rolling']+df_to_csv['LO-OOA_Rolling']-df_to_csv['MO-OOA_seas']-df_to_csv['LO-OOA_seas']
Dif2=Dif2.reset_index()
Dif2['Time']=df_to_csv['Time']
Dif2['Month']=Dif2.Time.dt.month
Dif2['Hour']=Dif2.Time.dt.hour
Dif2['Time'].drop(index=1, inplace=True)

#%% Rolling - Seasonal monthly + diel BP plots
import matplotlib.patches as mpatches
boxprops = dict(linestyle='-', linewidth=0.6, edgecolor='black', facecolor='white')
meanprops=dict(marker='o', linewidth=0.6, markeredgecolor='black', markerfacecolor='black')
medianprops=dict( linewidth=0.6, color='black')
whiskerprops=dict( linewidth=0.6, color='black')
fig, axs=plt.subplots(len(columns),2, figsize=(18,20), sharex='col', sharey='row')
# columns=['COA', 'SHINDOA', 'LOA','WCOA','CCOA','PCOA']#,'Month']
columns=['HOA', 'BBOA', 'LO-OOA', 'MO-OOA', 'OOA']
for i in range(0,len(columns)):
    Dif2.boxplot(column=columns[i], by='Month', ax=axs[i,0], showfliers=False, showmeans=True ,patch_artist=True,boxprops=boxprops, meanprops=meanprops, medianprops=medianprops,whiskerprops=whiskerprops, fontsize=20)
    Dif2.boxplot(column=columns[i], by='Hour', ax=axs[i,1], showfliers=False, showmeans=True, patch_artist=True, boxprops=boxprops, meanprops=meanprops,  medianprops=medianprops,whiskerprops=whiskerprops, fontsize=20)
    axs[i,0].set_ylabel(columns[i]+'\n', fontsize=20, loc='center')
    axs[i,0].set_title('')
    axs[i,0].set_xlabel('')
    axs[i,1].set_xlabel('')
    axs[i,1].set_title('')

fig.suptitle('')
axs[4,0].set_xlabel('\n'+'Month', fontsize=20)
axs[4,0].set_xticks(range(0,13))
axs[2,0].set_ylabel('Rolling minus seasonal concentrations($μg·m^{-3}$)' + '\n'+'\n'+'LO-OOA'+'\n')
axs[4,0].set_xticklabels(labels=['', 'J','F','M','A','M','J','J','A','S','O','N','D'])
axs[4,1].set_xticks(range(0,25))
axs[4,1].set_xlabel('\n'+'Hour', fontsize=20)
axs[4,1].set_xticklabels(labels=['','0','','2','','4','', '6','','8','','10','','12','', '14','','16','','18','', '20', '', '22','', ])#, ])

#%%
Dif2=pd.DataFrame()
Dif2['HOA']=df_to_csv['HOA_Rolling']-df_to_csv['HOA_seas']/(df_to_csv['HOA_Rolling']+df_to_csv['HOA_seas'])
Dif2['BBOA']=df_to_csv['BBOA_Rolling']-df_to_csv['BBOA_seas']/(df_to_csv['BBOA_Rolling']+df_to_csv['BBOA_seas'])
Dif2['LO-OOA']=df_to_csv['LO-OOA_Rolling']-df_to_csv['LO-OOA_seas']/(df_to_csv['LO-OOA_Rolling']+df_to_csv['LO-OOA_seas'])
Dif2['MO-OOA']=df_to_csv['MO-OOA_Rolling']-df_to_csv['MO-OOA_seas']/(df_to_csv['MO-OOA_Rolling']+df_to_csv['MO-OOA_seas'])
Dif2['OOA']=(df_to_csv['MO-OOA_Rolling']+df_to_csv['LO-OOA_Rolling']-df_to_csv['MO-OOA_seas']-df_to_csv['LO-OOA_seas'])/(df_to_csv['MO-OOA_Rolling']+df_to_csv['LO-OOA_Rolling']+df_to_csv['MO-OOA_seas']+df_to_csv['LO-OOA_seas'])
Dif2['COA']=df_to_csv['COA_Rolling']-df_to_csv['COA_seas']/(df_to_csv['COA_Rolling']+df_to_csv['COA_seas'])
Dif2['LOA']=df_to_csv['LOA_Rolling']-df_to_csv['LOA_seas']/(df_to_csv['LOA_Rolling']+df_to_csv['LOA_seas'])
Dif2['SHINDOA']=df_to_csv['SHINDOA_Rolling']-df_to_csv['SHINDOA_seas']/(df_to_csv['SHINDOA_Rolling']+df_to_csv['SHINDOA_seas'])
Dif2['WCOA']=df_to_csv['Wood_Rolling']-df_to_csv['Wood_seas']/(df_to_csv['Wood_Rolling']+df_to_csv['Wood_seas'])
Dif2['CCOA']=df_to_csv['Coal_Rolling']-df_to_csv['Coal_seas']/(df_to_csv['Coal_Rolling']+df_to_csv['Coal_seas'])
Dif2['PCOA']=df_to_csv['Peat_Rolling']-df_to_csv['Peat_seas']/(df_to_csv['Peat_Rolling']+df_to_csv['Peat_seas'])

# Dif2 = Dif2[['HOA', 'BBOA', 'LO-OOA', 'MO-OOA','COA', 'SHINDOA', 'LOA','WCOA','CCOA','PCOA']]
#%%
fig, ax=plt.subplots(figsize=(15,5))
Dif2.boxplot(showfliers=False, showmeans=True,patch_artist=True,boxprops=boxprops, meanprops=meanprops, 
             medianprops=medianprops,whiskerprops=whiskerprops,ax=ax, fontsize=14)
ax.set_ylabel('Relative error (adim.)', fontsize=15)