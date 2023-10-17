# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:04:04 2022

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
typedata='MT'
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/')
# os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Traditional/')
# os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/')
# os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/LR/')
# os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/BC/')

amus_txt=pd.read_csv("amus_txt.txt", sep='\t', header=0)
# amus_txt=amus_txt.iloc[:99]
mz=amus_txt
# time=pd.read_csv("date.txt", sep='\t', header=0,dayfirst=True )
# time=pd.read_csv("date_BC.txt", sep='\t', header=0,dayfirst=True )
time=pd.read_csv("date_30m.txt", sep='\t', header=0,dayfirst=True )
date=pd.to_datetime(time['date_out'],dayfirst=True)
pm1=pd.read_csv('pm1.txt', sep='\t')
if typedata=='BC':
    date=date.iloc[17604:]
#%%
if typedata=='HR':
    labels=['0','1','2', '3','4', '5','6']
    num_f=6
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/HR/Solutions/SA/')
    pr=pd.read_csv('Profiles_6.txt', sep='\t')
    pr.columns=labels
    del pr['0']
    rf=pd.read_csv('RelProf_6.txt', sep='\t')
    rf.columns=labels
    del rf['0']
    ts=pd.read_csv('TS_6.txt', sep='\t')
    del ts['d']
if typedata=='LR':
    labels=['0','1','2', '3','4']#], '5','6']
    num_f=4    
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/LR/Solutions/F_20221111_Mg/SA/')
    pr=pd.read_csv('Profiles_Def.txt', sep='\t')
    pr.columns=labels
    del pr['0']
    rf=pd.read_csv('RelProf_Def.txt', sep='\t')
    rf.columns=labels
    del rf['0']
    ts=pd.read_csv('TS_Def.txt', sep='\t')
    ts.columns=['d','1', '2','3','4']
    del ts['d']
if typedata=='Traditional':
    labels=['1','2', '3','4']#], '5','6']
    num_f=4    
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/Traditional/Runs_Mg_1_2/SA/')
    pr=pd.read_csv('Profiles_Def.txt', sep='\t')
    pr.columns=labels
    # del pr['0']
    rf=pd.read_csv('RelProf_Def.txt', sep='\t')
    rf.columns=labels
    # del rf['0']
    ts=pd.read_csv('TS_Def.txt', sep='\t')
    ts.columns=['d', '1', '2','3','4']
    del ts['d']
if typedata=='Base Case':
    labels=['1','2', '3','4', '5','6']
    num_f=6    
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/BC/SA/')
    pr=pd.read_csv('Profiles_Def.txt', sep='\t')
    pr.columns=labels
    # del pr['0']
    rf=pd.read_csv('RelProf_Def.txt', sep='\t')
    del rf['0']    
    rf.columns=labels
    ts=pd.read_csv('TS_Def.txt', sep='\t')
    ts.columns=['d', '1', '2','3','4', '5', '6']
    del ts['d']
if typedata=='MT':
    labels=['0','1','2', '3','4', '5','6', '7', '8']
    num_f=8 
    os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/30m_1_5/SA/')
    pr=pd.read_csv('Profiles_Def.txt', sep='\t')
    pr.columns=labels
    pr_std=pd.read_csv('Profile_std_Def.txt', sep='\t')
    pr_std.columns=labels
    # del pr['0']
    rf=pd.read_csv('RelProf_Def.txt', sep='\t')
    rf.columns=labels
    # del rf['0']
    ts=pd.read_csv('TS_Def.txt', sep='\t')
    ts.columns=['d', '1', '2','3','4', '5', '6','7', '8']
    del ts['d']
    res=pd.read_csv('Res_Def.txt', sep='\t')
    res.columns=['1', '2']
#%%Changing column order
if typedata=='HR':
    pr=pr[['4', '6', '3','5', '2', '1']]
    rf=rf[['4', '6', '3','5', '2', '1']]
    ts=ts[['3', '5', '2','4', '1', '0']]
    labels_2=['Aged SOA + AS','Fresh SOA +\n AN + ACl','Traffic', 'Fresh SOA + \nIndustry', 'Biomass Burning', 'Cooking-like OA']
    pr.columns=labels_2
    rf.columns=labels_2
    ts.columns=labels_2
if typedata=='LR':
    pr=pr[['4', '3', '1',  '2']]
    rf=rf[['4', '3', '1',  '2']]
    ts=ts[['4', '3', '1',  '2']]
    labels_2=['Biomass Burning', 'Heavy-oil Combustion',  'Industry','Road Traffic']
    pr.columns=labels_2
    rf.columns=labels_2
    ts.columns=labels_2
if typedata=='Traditional':
    pr=pr[['4', '3', '2',  '1']]
    rf=rf[['4', '3', '2',  '1']]
    ts=ts[['4', '3', '2',  '1']]
    labels_2=['AN + \nBiomass Burning','AS + Heavy-oil\n Combustion', 'Industry', 'Traffic']
    pr.columns=labels_2
    rf.columns=labels_2
    ts.columns=labels_2
if typedata=='Base Case':
    pr=pr[['1','2','3','4','5','6']]
    rf=rf[['1','2','3','4','5','6']]
    ts=ts[['1','2','3','4','5','6']]
    labels_2=['Aged SOA +\nAS', 'Fresh SOA +\nAN','Aged SOA + \nHeavy oil combustion',  
               'Traffic','Biomass Burning \n + Mineral', 'Industry + \nRoad Dust']
    pr.columns=labels_2
    rf.columns=labels_2
    ts.columns=labels_2
if typedata=='MT':
    pr=pr[['4', '5','6', '1','2','3','8', '7']]
    rf=rf[['4', '5','6', '1','2','3','8', '7']]
    ts=ts[['4', '5','6', '1','2','3','8', '7']]
    pr_std=pr_std[['4', '5','6', '1','2','3','8', '7']]
    labels_2=['AS + Heavy oil\n combustion', 'AN + ACl', 'Aged SOA', 'Traffic', 
              'Biomass \nBurning', 'Fresh SOA +\n Road dust',  'COA', 'Industry']
    pr.columns=labels_2
    rf.columns=labels_2
    ts.columns=labels_2
    pr_std.columns=labels_2
pr.to_csv('Def_Profiles.txt', sep='\t')
pr_std.to_csv('Def_pr_std.txt', sep='\t')
ts.to_csv('Def_TS.txt', sep='\t')
rf.to_csv('Def_RelfProfiles.txt', sep='\t')
#%% Profiles
# mz=pd.read_csv('amus_txt.txt', sep='\t')
if typedata=='HR':
    Profiles_OA(pr, rf, str(num_f)+' factors',num_f)
if typedata=='LR' or typedata=='Traditional':
    Profiles_F(pr, rf, str(num_f)+' factors',num_f)
if typedata=='MT' or typedata=='Base Case':    
    Profiles(pr, rf, str(num_f)+' factors',num_f)
    Profiles_std(pr,pr_std, rf, str(num_f)+' factors',num_f)
#%%Pie
pie=ts.mean(axis=0)
pie_relf = pd.Series([(i/pie.sum()) for i in pie])
plt.pie(pie_relf, startangle=90, counterclock=False,  autopct='%1.0f%%',labels=labels_2,#labels=labels_2,
#        colors=[ 'white', 'gainsboro','silver',  'darkgrey', 'gray',  'dimgrey', '#424242', '#282828']) #'lightgrey',
        colors = ['maroon','navy', 'darkgreen', 'grey', 'saddlebrown', 'limegreen', 'slateblue', 'violet'])
# pie_relf.plot.pie(autopct='%1.1f%%', fontsize=10, startangle=90,
                  # labels=labels,colors=[ 'white','gainsboro', 'lightgrey', 'silver', 'darkgray', 'gray' , 'dimgrey', '#424242'])#'white',#,'slategrey'])
plt.ylabel("")
#%%Time series
# date=date.iloc[:17604]
ts=ts.set_index(date)

fig, axes=plt.subplots(figsize=(7,11), nrows=num_f,ncols=1, sharex=True)
for i in range(0,num_f):
    axes[i].plot(ts.iloc[:,i], color='grey')
    axes[i].grid('x')
    axes[i].set_ylabel(labels_2[i]+'\n(μg·m$^{-3}$)', fontsize=12)
# ts.plot(subplots=True,figsize=(15,20),  grid=True, legend=False, title=labels, color='grey',lw=3, fontsize=15)#marker='o'
#%%Mass Closure
pm1.set_index(pm1['date_in'], inplace=True)
# del pm1['date_out']
# pm1=pd.read_csv('MC_Def.txt', sep='\t')
date=pd.to_datetime(pm1['date_in'])

ts_sum=ts.sum(axis=1)
col=dt
fig, axs=plt.subplots(figsize=(5,5))
pm1_plot=plt.scatter(x=pm1['pm1'][:17605], y=ts_sum[:17605],c=date.dt.month[:17605], cmap='hsv')#
axs.set_xlabel('$PM_1 (\mu g·m^{-3})$', fontsize=12)
axs.set_ylabel('Apportioned ' +'$PM_1 (\mu g·m^{-3})$', fontsize=12)
axs.set_xlim(-1,50)
axs.set_ylim(-1,50)
axs.grid()
a=plt.colorbar()
a.set_label('Month', fontsize=12)
# fig.suptitle(run_name + '  '+str(num_f) + 'F')
# pm1['pm1_in'].reset_index(drop=True, inplace=True)
# ts_sum.reset_index(drop=True, inplace=True)
fig.text(x=0.15,y=0.75,s='Linear regression \n'+ 
         '$R^2 = $' + str(0.89) + '\n'+'y='+
         str(0.88)+'x+'+str(0.16))
         # '$R^2 = $ '+str(R2(pm1['pm1'], ts_sum))+'\n'+'y='+
         # str(slope(ts_sum,pm1['pm1'])[0])+'x+'+str(slope(ts_sum,pm1['pm1'])[1]))
#%% Monthly
ts['d']=ts.index
ts['Month']=ts['d'].dt.month
Month=ts.groupby(ts['Month']).mean()
Month.columns=labels_2#['Traffic', 'Cooking-like', 'Biomass \nBurning', 'Fresh SOA + \nIndustry', 'Fresh SOA +\n AN + ACl', 'Aged SOA +\nAS']
fig,axes=plt.subplots(nrows=num_f, figsize=(5,8), sharex=True)
for i in range(0,num_f):
    axes[i].plot(Month[Month.columns[i]], marker='o', color='grey')
    axes[i].set_xticklabels('')
    axes[i].grid('x')
    axes[i].set_ylabel(Month.columns[i]+'\n(μg·m$^{-3}$)')
# Month.plot(subplots=True,  marker='o', color='grey', legend=False, grid=True, ax=axes)
axes[3].set_xticks(Month.index)
axes[3].set_xticklabels([ 'J','F','M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axes[num_f-1].set_xlabel('Month', fontsize=12)
del ts['Month']
#%% Diel

ts['d']=ts.index
ts['Hour']=ts['d'].dt.hour
Hour=ts.groupby(ts['Hour']).mean()
Hour.columns=labels_2
fig,axes=plt.subplots(nrows=len(Hour.columns), figsize=(5,8), sharex=True)
for i in range(0,len(Hour.columns)):
    axes[i].plot(Hour[Hour.columns[i]], marker='o', color='grey')
    axes[i].set_xticklabels('')
    axes[i].set_ylim(0,0.5+max(Hour[Hour.columns[i]]))
    axes[i].grid('x')
    axes[i].set_ylabel(Hour.columns[i][:])
axes[i].set_xticks(Hour.index)
labels_h=['0', '','2','', '4', '', '6', '', '8', '', '10', '', '12', '','14' , '','16', '','18',  '','20', '','22', '']
axes[i].set_xticklabels(labels_h)
axes[i].set_xlabel('Hour', fontsize=12)
del ts['Hour']
#%%
mask = (ts['Month'] == 6) |  (ts['Month'] == 7) | (ts['Month'] == 8)
ts_jja  =  ts[mask]
del ts_jja['Month']
mask = (ts['Month'] == 12) |  (ts['Month'] == 1) | (ts['Month'] == 2)
ts_djf  =  ts[mask]
del ts_djf['Month']
#%%
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
ts_jja['d']=ts_jja.index
ts_djf['d']=ts_djf.index
ts_jja['Hour']=ts_jja['d'].dt.hour
ts_djf['Hour']=ts_djf['d'].dt.hour
Hour_s=ts_jja.groupby(ts_jja['Hour']).mean()
Hour_w=ts_djf.groupby(ts_djf['Hour']).mean()
Hour_s.columns, Hour_w.columns = labels_2,labels_2
lim_Y=[5,4,4,3,3.5,2,2,2]
lim_y=[0,0,0,0,0,0,0,0]
fig,axes=plt.subplots(nrows=8,ncols=2, figsize=(5,8), sharex=True)
for i in range(0,8):
    axes[i,0].plot(Hour_s[Hour_s.columns[i]], marker='o', color='grey')
    axes[i,0].set_xticklabels('')
    # axes[i,0].set_ylim(0,1+max(Hour_s[Hour_s.columns[i]]))
    # axes[i,0].grid('y')
    # axes[i,0].grid('x', visible=False)
    axes[i,0].set_ylabel(Hour_s.columns[i][:]+'\n')
    axes[i,1].plot(Hour_w[Hour_w.columns[i]], marker='o', color='grey')
    axes[i,1].set_xticklabels('')
    axes[i,0].set_ylim( lim_y[i], lim_Y[i])
    axes[i,1].set_ylim(lim_y[i], lim_Y[i])
    axes[i,1].set_yticklabels('')
    # axes[i,1].grid('x', 'major')
axes[7,0].set_xticks(Hour_s.index)
labels_h=['0', '','','', '4', '', '', '', '8', '', '', '', '12', '','' , '','16', '','',  '','20', '','', '']
axes[7,0].set_xticklabels(labels_h)
axes[7,0].set_xlabel('Hour', fontsize=12)
axes[7,1].set_xticks(Hour_s.index)
axes[7,1].set_xticklabels(labels_h)
axes[7,1].set_xlabel('Hour', fontsize=12)
axes[0,0].set_title ('JJA')
axes[0,1].set_title ('DJF')
del ts_jja['Hour'], ts_djf['Hour']

#%% Polar plots
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/30m_1_2/SA')
# meteo=pd.read_csv('Meteo_6F_FinalArranged.txt', sep='\t')
meteo=pd.read_csv('MT_meteo.txt', sep='\t')
# labels_meteo=['HOA', 'COA+INDOA', 'BBOA', 'Fresh SOA + AN + ACl','Aged SOA + AS + Ship', 'INDOA', 'RD','ws','wd']
# meteo.columns=labels_meteo
# meteo2=ts
# meteo2['wd']=meteo['wd']
# meteo2['ws']=meteo['ws']
# meteo=meteo[['HOA', 'COA+INDOA', 'BBOA', 'INDOA', 'RD', 'Fresh SOA + AN + ACl', 'Aged SOA + AS + Ship', 'ws', 'wd']]

#%% POLLUTION ROSES
meteo2=meteo
factor=meteo2.columns[1]
theta = np.linspace(0,360,17)
r=np.linspace(0,10,11)
li_theta=[]
df=meteo2
for i in range(0,len(theta)-1):
    mask=(df['wd']>= theta[i]) &  (df['wd']<theta[i+1])
    df1=df[mask]
    li_r=[]
    for j in range(0,len(r)-1):
        mask2=(df1['ws']>r[j])&(df1['ws']<r[j+1])
        df2=df1[mask2]
        li_r.append(df2[factor].mean())
    li_theta.append(li_r)
theta_rad=(theta*np.pi/180.0).round(5)
poll=pd.DataFrame(li_theta)#, columns=r[:-1],index=theta_rad[:-1])   

fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(4,4))
R, Theta=np.meshgrid(r,theta*np.pi/180.0)
plot=ax.pcolormesh(Theta, R, poll, cmap='Greys', vmin=-0.19,vmax=(meteo2[factor].quantile(0.80)))
# ax.contourf(Theta, R,poll)
# ax.set_xticks(np.pi/180. * np.linspace(0,360, 16 , endpoint=False))
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")  # theta=0 at the top
ax.set_rticks(r)  # Less radial ticks
ax.grid()
ax.set_title('AS + Heavy oil combustion' +' \n$(\mu g·m^{-3})$', fontsize=15)
cb=fig.colorbar(plot, ax=ax, orientation='horizontal')
# cb.set_clim(0,2.5)    
#%%
'''Residuals plot'''
res=pd.read_csv('Res_Def.txt', sep='\t')
res_hr=res['0'].iloc[:17603]
res_lr=res['0'].iloc[17603:]
bins=100
hist_intersection_r1=[]
fig, axs=plt.subplots(figsize=(5,5), dpi=100)
HR_f = res_hr# res_hr[(res_hr['Res_HR']>=-50) & (res_hr['Res_HR'] <=50)]
LR_f = res_lr#res_lr[(res_lr['Res_LR'] >=-50) & (res_lr['Res_LR'] <=50)]
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
axs2.set_ylabel('Frequency (adim.)', fontsize=13)
axs2.set_xlabel('Scaled residuals', fontsize=13)
axs2.text(x=-50,y=14, s='F$^{*}$= '+str(hist_int.round(1))+'%', fontsize=12)
axs2.legend(['HR', 'LR'], loc='upper left')
#%%
pr2=pr/pr.sum(axis=0)
Profiles_OA(pr2, rf, str(num_f)+' factors',num_f)
#%%
def Profiles_OA(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(38,20), sharex='col',gridspec_kw=dict(width_ratios=[3, 1]))
        for i in range(0,nf):
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=0)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            axes[i,1].grid(axis='x')
            axes[i,0].tick_params('x', labelrotation=90, labelsize=16)
            axes[i,0].tick_params('y', labelrotation=0, labelsize=16)
            axes[i,1].tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_ylabel(df.columns[i], fontsize=18)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            ax2.set_ylim(0,max(relf.iloc[:,i][:73]))
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)

            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)    
#%%
Profiles(pr, rf, str(num_f)+' factors',num_f)

 #%%
def Profiles(df, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(22,15), sharex='col',gridspec_kw={'width_ratios': [1.5, 1]})
        for i in range(0,nf):
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            OA_perc=(100.0*df.iloc[:73, i].sum()/df.iloc[:,i].sum()).round()
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            # axes[i,0].set_ylabel(labels_2[i], fontsize=17)
            axes[i,1].grid(axis='x')
            axes[i,0].set_ylabel(labels_2[i], fontsize=18)
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            axes[i,0].tick_params('x', labelrotation=90, labelsize=16)
            axes[i,0].tick_params('y', labelrotation=0, labelsize=16)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            axes[i,0].text(x=-3,y=max(df.iloc[:73,i])*0.8,s='OA = '+str(OA_perc)+'%', fontsize=16 )
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout(pad=2.0)
            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)  

            #%%
def Profiles_std(df,df_std, relf, name, nf):
        fig,axes=plt.subplots(nrows=nf,ncols=2, figsize=(22,15), sharex='col',gridspec_kw={'width_ratios': [1.5, 1]})
        for i in range(0,nf):
            axes[i,0].bar(mz['lab'][:73], df.iloc[:,i][:73], color='grey')
            axes[i,0].errorbar(mz['lab'][:73], df.iloc[:,i][:73], yerr=df_std.iloc[:,i][:73],  fmt='.', color='k')#, uplims=True, lolims=True)
            OA_perc=(100.0*df.iloc[:73, i].sum()/df.iloc[:,i].sum()).round(0)
            ax2=axes[i,0].twinx()
            axes[i,0].tick_params(labelrotation=90)
            ax2.plot(mz.lab[:73], relf.iloc[:,i][:73], marker='o', linewidth=False,color='black')
            axes[i,0].grid(axis='x')
            axes[i,1].bar(mz['lab'][73:], df.iloc[:,i][73:], color='grey')
            axes[i,1].errorbar(mz['lab'][73:], df.iloc[:,i][73:], yerr=df_std.iloc[:,i][73:],  fmt='.', color='k')#, uplims=True, lolims=True)            
            axes[i,1].set_yscale('log')
            ax4=axes[i,1].twinx()
            ax4.plot(mz.lab[73:], relf.iloc[:,i][73:], marker='o', linewidth=False,color='black')
            # ax4.set_yscale('log')
            # axes[i,0].set_ylabel(labels_2[i], fontsize=17)
            axes[i,1].grid(axis='x')
            axes[i,0].set_ylabel(labels_2[i], fontsize=18)
            axes[i,0].tick_params(labelrotation=90, labelsize=16)
            axes[i,1].tick_params(labelrotation=90, labelsize=16)
            axes[i,0].tick_params('x', labelrotation=90, labelsize=16)
            axes[i,0].tick_params('y', labelrotation=0, labelsize=16)
            ax4.tick_params(labelrotation=0, labelsize=16)
            ax2.tick_params(labelrotation=0, labelsize=16)
            axes[i,0].set_xticklabels(mz.lab[:73], fontsize=17)
            axes[i,1].set_xticklabels(mz.lab[73:], fontsize=17)
            axes[i,0].text(x=-3,y=max(df.iloc[:73,i])*0.9,s='OA = '+str(OA_perc)+'%', fontsize=16 )
            # axes[i,0].tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout(pad=2.0)
            # axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)   

#%%Profiles F
def Profiles_F(df, relf, name, nf):
    fig,axes=plt.subplots(nrows=nf, figsize=(8,8), sharex='col')
    for i in range(0,nf):
        axes[i].bar(mz['lab'], df.iloc[:,i], color='gray')
        ax2=axes[i].twinx()
        axes[i].tick_params(labelrotation=90)
        ax2.plot(mz['lab'], relf.iloc[:,i], marker='o', linewidth=False,color='black')
        axes[i].grid()
        axes[i].set_ylabel(df.columns[i]+'\n(μg·m$^{-3}$)')
        axes[i].set_yscale('log')
    # fig.suptitle(name, fontsize=16)
    # plt.text(x=0, y=10, s='Concentration (μg·m$^{-3}$)', fontsize=14)# \t\t\t\t
    axes[num_f-1].set_xlabel('Species', fontsize=14) 
#%%
Profiles_F(pr,rf, str(num_f), num_f)
#%%
ts_lr=pd.read_csv('TS_HR_LR.txt', sep='\t')    
corrM = (ts_lr.corr())**2.0
corrM_cut=corrM#J.iloc[:6,6:]
#%%
fig,axs=plt.subplots(figsize=(10,6))
plot=axs.matshow(corrM_cut, cmap='Greys', vmin=0.0,vmax=1)
axs.set_yticks(range(0,6))
axs.set_yticklabels(corrM_cut.index)
axs.set_xticks(range(0,18))
axs.set_xticklabels(corrM_cut.columns)
cb=fig.colorbar(plot, ax=axs, orientation='horizontal', shrink=0.5)

#%%
num_run=30
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/12h_Sweep/")
prof=pd.read_csv('Profile_run_'+str(num_run)+'.txt', sep='\t',skiprows=1, header=None)
relp=pd.read_csv('Rel_Profile_run_'+str(num_run)+'.txt', sep='\t', skiprows=1, header=None)
ts=pd.read_csv('TS_run_'+str(num_run)+'.txt', sep='\t',skiprows=1, header=None)
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR-PMF/MT/")
amus_txt=pd.read_csv('amus_txt.txt', sep='\t')
mz=amus_txt
date=pd.read_csv('date_12h.txt', sep='\t', dayfirst=True)
dt=pd.to_datetime(date['date_out'], dayfirst=True)
#%%
Profiles(prof,relp,'run30', 6)
#%%
ts['dt']=dt
ts.index=dt
del ts['dt']
ts.plot(subplots=True, color='grey', figsize=(10,6))
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
        intercept = np.nan
    else:
        s, intercept, r_value, p_value, std_err = linregress(a1, b1)
    return round(s,2), round(intercept,2)

#%%
            #%%
factor=7
co='violet'#0:maroon, 1:navy, 2:darkgreen, 3:grey, 4:saddlebrown, 5:limegreen, 6:slateblue, 7:violet
df=pr      
relf=rf      
fig,axes=plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1.8, 1]}, figsize=(22,4))
axes[0].bar(mz['lab'][:73], df.iloc[:,factor][:73], color=co)
OA_perc=(100.0*df.iloc[:73, factor].sum()/df.iloc[:,factor].sum()).round(0)
ax2=axes[0].twinx()
axes[0].tick_params(labelrotation=90)
ax2.plot(mz.lab[:73], relf.iloc[:,factor][:73], marker='o', linewidth=False,color='black')
axes[0].grid(axis='x')
axes[1].bar(mz['lab'][73:], df.iloc[:,factor][73:], color=co)
axes[1].set_yscale('log')
ax4=axes[1].twinx()
ax4.plot(mz.lab[73:], relf.iloc[:,factor][73:], marker='o', linewidth=False,color='black')
axes[1].grid(axis='x')
axes[0].set_ylabel(labels_2[factor], fontsize=18)
axes[0].tick_params(labelrotation=90, labelsize=16)
axes[1].tick_params(labelrotation=90, labelsize=16)
axes[0].tick_params('x', labelrotation=90, labelsize=16)
axes[0].tick_params('y', labelrotation=0, labelsize=16)
ax4.tick_params(labelrotation=0, labelsize=16)
ax2.tick_params(labelrotation=0, labelsize=16)
axes[0].set_xticklabels(mz.lab[:73], fontsize=17)
axes[1].set_xticklabels(mz.lab[73:], fontsize=17)
axes[0].text(x=-3,y=max(df.iloc[:73,factor])*0.9,s='OA = '+str(OA_perc)+'%', fontsize=16 )
# axes[factor,0].tick_params(axis='both', which='major', labelsize=10)
fig.tight_layout(pad=2.0)
# axes[i,0].set_ylabel(labels_pie[i] + '\n')
            # fig.suptitle(name)   
#%%
factor=7
cols=['maroon', 'navy', 'darkgreen', 'grey', 'saddlebrown', 'limegreen', 'slateblue', 'violet']
col=cols[factor]#0:maroon, 1:navy, 2:darkgreen, 3:grey, 4:saddlebrown, 5:limegreen, 6:slateblue, 7:violet
ts=ts.set_index(date)
fig, axes=plt.subplots(figsize=(11,3), nrows=1,ncols=1, sharex=True)
axes.plot(ts.iloc[:17603,factor], color=col)
axes.grid('x')
axes.set_ylabel(labels_2[factor]+'\n(μg·m$^{-3}$)', fontsize=12)

#%% Monthly
ts['d']=ts.index
ts['Month']=ts['d'].dt.month
Month=ts.groupby(ts['Month']).mean()
Month.columns=labels_2#['Traffic', 'Cooking-like', 'Biomass \nBurning', 'Fresh SOA + \nIndustry', 'Fresh SOA +\n AN + ACl', 'Aged SOA +\nAS']
ts['d']=ts.index
ts['Hour']=ts['d'].dt.hour
Hour=ts.groupby(ts['Hour']).mean()
Hour.columns=labels_2
#%%
factor=6
col=cols[factor]
fig,axes=plt.subplots(nrows=1, figsize=(4,2), sharex=True)
axes.plot(Month[Month.columns[factor]], marker='o', color=col)
axes.set_xticklabels('')
axes.grid('x')
axes.set_ylabel(Month.columns[factor]+'\n(μg·m$^{-3}$)')
# Month.plot(subplots=True,  marker='o', color='grey', legend=False, grid=True, ax=axes)
axes.set_xticks(Month.index)
axes.set_xticklabels([ 'J','F','M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axes.set_xlabel('Month', fontsize=12)
del ts['Month']
#%% Diel
factor=7
col=cols[factor]
fig,axes=plt.subplots(1, figsize=(3,2), sharex=True)
axes.plot(Hour[Hour.columns[factor]], marker='o', color=col)
axes.set_xticklabels('')
# axes.set_ylim(0,0.5+max(Hour[Hour.columns[factor]]))
axes.grid('x')
axes.set_ylabel(Hour.columns[factor][:])
axes.set_xticks(Hour.index)
labels_h=['0', '','2','', '4', '', '6', '', '8', '', '10', '', '12', '','14' , '','16', '','18',  '','20', '','22', '']
axes.set_xticklabels(labels_h)
axes.set_xlabel('Hour', fontsize=12)

