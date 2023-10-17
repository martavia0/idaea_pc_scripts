# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:10:25 2023

@author: Marta Via
"""

from scipy.odr import Model, Data, ODR
import scipy
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
# import statsmodels.api as sm
import glob
import math
# import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#%%
site='Barcelona'
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. Data/Papers/Comp_Rolling_Seasonal/'+site)
mgd=pd.read_csv(site+'.txt', sep='\t',dayfirst=True )

#%%
fig, axs=plt.subplots()
mgd.plot.kde(y=['mz43_R', 'mz43_S', 'mz43'])
#%%
fig, axs=plt.subplots()
mgd.plot.scatter(x='mz44',  y=['mz44_R'], color='red', ax=axs)
mgd.plot.scatter(x='mz44',  y=['mz44_S'], color='blue', ax=axs)
#%%
dif=pd.DataFrame()
dif['57_R']=(mgd['mz57']-HOA_R['57'])/mgd['mz57']
dif['57_S']=(mgd['mz57']-HOA_S['57'])/mgd['mz57']
print(dif.mean())
dif.boxplot(showfliers=False, showmeans=True)
#%%
fig, axs=plt.subplots()
mgd.boxplot(column=['mz43', 'mz44_R', 'mz43_S'],  ax=axs, showfliers=False, showmeans=True)
#%%
import matplotlib
site='SIRTA'
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. Data/Papers/Comp_Rolling_Seasonal/'+site)
mgd=pd.read_csv(site+'.txt', sep='\t',dayfirst=True )
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. Data/Papers/Comp_Rolling_Seasonal/'+site+'/profiles')
LO_R=pd.read_csv('TDP_OOA_Rolling.txt', sep='\t', dayfirst=True)
LO_S=pd.read_csv('TDP_OOA_Seasonal.txt', sep='\t',dayfirst=True )

fig, axs=plt.subplots(figsize=(3,3))
ratio=pd.DataFrame()
ratio['Input']=mgd['mz43']/mgd['mz44']
ratio['Rolling']=LO_R['43']/LO_R['44']
ratio['Seasonal']=LO_S['43']/LO_S['44']

boxprops = dict(linestyle='-', linewidth=1, color='black')
meanprops = dict(marker='o', markerfacecolor='white', mec='k', markersize=3)
medianprops = dict(linestyle='-', linewidth=1, color='black')
whiskerprops = dict(linestyle='-', linewidth=1, color='black')
#notch=True, bootstrap=10000, 
axes=ratio.boxplot(showfliers=False, showmeans=True, patch_artist=True, 
                   boxprops=boxprops, meanprops=meanprops,
                   medianprops=medianprops, whiskerprops=whiskerprops)
axes.findobj(matplotlib.patches.Patch)[0].set_facecolor("grey")
axes.findobj(matplotlib.patches.Patch)[1].set_facecolor("red")
axes.findobj(matplotlib.patches.Patch)[2].set_facecolor("blue")
axs.set_title(site)

#%%
ratio.plot.kde()

# mgd['mz43'].plot(color='k', ax=axs)
# LO_R['43'].plot(color='red', ax=axs)
# LO_S['43'].plot(color='blue', ax=axs)

#%%

# HOA_R['57'].plot.kde(color='red', ax=axs)
# HOA_S['57'].plot.kde(color='blue', ax=axs)

# HOA_S.plot.kde(y.loc[43])
LO=pd.read_csv('A2_4443.txt', sep='\t',dayfirst=True)
LO.boxplot(showfliers=False, showmeans=True)