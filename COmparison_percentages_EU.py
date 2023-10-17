# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:18:12 2023

@author: Marta Via
"""


import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%%
path='C:/Users/maria/Documents/Marta Via/1. PhD/H. Thesis/Presentació/Data processing/'
os.chdir(path)
factors= pd.read_csv('European_sites.txt', sep='\t')
bcn=pd.read_csv('Barcelona.txt', sep='\t')
means_eu=factors.mean(axis=0)
std_eu=factors.std(axis=0)
#%% FOR ABS CONCNETRATIONS
fact='OA(ug_m3)'
fact_name='OA'
x=[0,1]
x1=[0,1]
#
eu_bcn=[bcn[fact_name].iloc[0],  means_eu[fact]]
eu_bcn_err=[bcn[fact_name].iloc[1],  std_eu[fact]]
#
plt.rcParams.update({'font.size':16})
fig, ax= plt.subplots(figsize=(6,5))
ax.yaxis.grid(True, zorder=0)
ax.bar(x, height=eu_bcn, color=['grey', 'dimgrey'], zorder=2)
plt.errorbar(x1, eu_bcn, np.array([eu_bcn_err[0], eu_bcn_err[1]]), fmt='r^', color='black', lw=12, zorder=3)
ax.set_xticks([0,1])
ax.set_xticklabels(['Barcelona - \nPalau Reial', 'Urban \nEuropean sites'], fontsize=18)
ax.set_title(fact_name, fontsize=26)
ax.set_title('POA / SOA (adim.)', fontsize=26)
ax.set_ylim(0,1.4)
ax.text(x=0-0.15, y=0.1*eu_bcn[0], s=eu_bcn[0].round(1), fontsize=26, c='white',  weight="bold")
ax.text(x=1-0.15, y=0.1*eu_bcn[0], s=eu_bcn[1].round(1), fontsize=26, c='white',  weight="bold")
ax.set_ylabel('Concentration (µg·m$^{-3}$)', fontsize=18)
# ax.set_ylabel('Ratio (adim.)', fontsize=18)
plt.tight_layout()
plt.savefig(fact_name+'.png')
#%% FOR PERCENTAGES
fact='MO-OOA'
x=[0,1]
x1=[np.nan,1]

plt.rcParams.update({'font.size':16})
fig, ax= plt.subplots(figsize=(6,5))
ax.yaxis.grid(True, zorder=0)
ax.bar(x, height=factors[fact].iloc[0:2], color=['grey', 'dimgrey'], zorder=2)
ax.set_xticks([0,1])
ax.set_xticklabels(['Barcelona - \nPalau Reial', 'Urban \nEuropean sites'], fontsize=18)
ax.set_title(fact)
ax.set_ylabel('Contribution to OA (%)', fontsize=18)
ax.set_title(fact, fontsize=26)
# ax.set_ylim(0,30)
plt.errorbar(x1, factors[fact].iloc[:2], np.array(yerr[fact].iloc[0],yerr[fact].iloc[1]), fmt='r^', color='black', lw=12, zorder=3)
ax.text(x=0-0.1, y=factors[fact].iloc[0]-25, s=factors[fact].iloc[0], fontsize=26)
ax.text(x=1-0.1, y=factors[fact].iloc[1]-20, s=factors[fact].iloc[1], fontsize=26)

plt.tight_layout()
plt.savefig(fact+'.png')