# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:12:52 2023

@author: Marta Via
"""


import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import scipy.integrate as sp
from scipy import stats
#%% t-test 
path="C:/Users/maria/Documents/Marta Via/1. PhD/H. Thesis/5. Discussion/Phenomenology/"
os.chdir(path)
ep=pd.read_csv('Episodes.txt', sep='\t')
#%%
nb_ep_a, nb_ep_b=[], []
episodes_a=['AN', 'AW', 'EU', 'NAF',  'SREG', 'WA']
episodes=['AN', 'AW', 'EU', 'NAF', 'MED', 'SREG', 'WA']
for i in episodes:
    nb_ep_a.append(round(100*len(ep[ep['EP_A'] == i])/17120))
for i in episodes:
    nb_ep_b.append(round(100*len(ep[ep['EP_B'] == i])/17605))
#%%
fig, ax1=plt.subplots(2,2, figsize=(10,10), sharey='row', sharex='col')
boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1.5, color='darkgrey')
meanprops = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
whiskerprops = dict(linestyle='-', linewidth=1, color='k')
ep.boxplot(column=['f43/f44_A'], by=['EP_A'], showfliers=False, rot=90, ax=ax1[0,0],fontsize=12, 
           showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ep.boxplot(column=['f43/f44_B'], by=['EP_B'], showfliers=False, rot=90, ax=ax1[0,1], fontsize=12, 
           showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ep.boxplot(column=['LO/MO_A'], by=['EP_A'], showfliers=False, rot=90, ax=ax1[1,0],fontsize=12,
            showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ep.boxplot(column=['LO/MO_B'], by=['EP_B'], showfliers=False, rot=90, ax=ax1[1,1], fontsize=12, 
            showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ax1[1,1].set_ylim([0,7])
for i in range(len(episodes)):
    ax1[0,0].text(x=i+0.75, y=1.1, s=str(nb_ep_a[i])+'%', fontsize=12)
    ax1[0,1].text(x=i+0.75, y=1.1, s=str(nb_ep_b[i])+'%', fontsize=12)
#ylabel
ax1[0,0].set_ylabel('m/z 43 / m/z 44 (adim.)', fontsize=15)
ax1[1,0].set_ylabel('LO-OOA / MO-OOA (adim.)', fontsize=15)
#xlabel
ax1[1,0].set_xlabel('Episodes', fontsize=15)
ax1[1,1].set_xlabel('Episodes', fontsize=15)
ax1[0,0].set_xlabel('', fontsize=15)
ax1[0,1].set_xlabel('', fontsize=15)
#titles
ax1[0,0].set_title('Period A', fontsize=16)
ax1[0,1].set_title('Period B', fontsize=16)
ax1[1,0].set_title('', fontsize=16)
ax1[1,1].set_title('', fontsize=16)
ax1[0,0].set_ylim([0,1.2])

fig.suptitle('')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
#%%
#%%
fig, ax1=plt.subplots(1,2, figsize=(10,5), sharey=True)
ep.boxplot(column=['LO/MO_A'], by=['EP_A'], showfliers=False, rot=90, ax=ax1[0],fontsize=12, 
           showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ep.boxplot(column=['LO/MO_B'], by=['EP_B'], showfliers=False, rot=90, ax=ax1[1], fontsize=12, 
           showmeans=True, meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)
ax1[0].set_ylim([0,1.2])

ax1[0].set_ylabel('m/z 43 / m/z 44 (adim.)', fontsize=15)
ax1[0].set_xlabel('Episodes', fontsize=15)
ax1[1].set_xlabel('Episodes', fontsize=15)
ax1[0].set_title('Period A', fontsize=16)
ax1[1].set_title('Period B', fontsize=16)
fig.suptitle('')
plt.subplots_adjust(wspace=0.05)
plt.show()