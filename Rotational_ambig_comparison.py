# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:51:09 2021

@author: Marta Via
"""
import os as os
import pandas as pd
import scipy.stats
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Barcelona")
s = pd.read_csv("spread.txt", sep="\t", low_memory=False)
s['Time']=pd.to_datetime(s['dt'])
#%%
factor='HOA'
ratios=pd.DataFrame()
ratios[factor+'_R']=s[factor+'_R_s']/s[factor+'_R_m']
ratios[factor+'_S']=s[factor+'_S_s']/s[factor+'_S_m']
#%%
bins_factor=np.linspace(-1,50,510)
ratios[factor+'_R_inter']=pd.cut(ratios[factor+'_R'], bins_factor)
ratios[factor+'_S_inter']=pd.cut(ratios[factor+'_S'], bins_factor)
hist_factor=pd.DataFrame()
hist_factor['mc']=bins_factor
hist_factor['Rolling']=ratios[factor+'_R_inter'].value_counts().sort_index()
hist_factor['Seasonal']=ratios[factor+'_S_inter'].value_counts().sort_index()
print(hist_factor.sum())
#%%
#Rolling
hist_factor_2= hist_factor[hist_factor['Rolling'] != 0]
hist_factor_2.plot.bar()
y = hist_factor_2['Rolling']#np.exp(hist_factor['Rolling']) # these values have lognormal distribution
shape, loc, scale=stats.lognorm.fit(y, floc=0)
estimated_mu = np.log(scale)
estimated_sigma = shape
print(estimated_mu, estimated_sigma)
# print(y[y == y.max()].index)
#Seasonal
hist_factor_2= hist_factor[hist_factor['Seasonal'] != 0]
y = hist_factor_2['Seasonal']#np.exp(hist_factor['Rolling']) # these values have lognormal distribution
shape, loc, scale=stats.lognorm.fit(y, floc=0)
estimated_mu = np.log(scale)
estimated_sigma = shape
print(estimated_mu, estimated_sigma)
# print(y[y == y.max()].index)
#%%
hist_factor.to_csv('HOA_hist_log.txt', sep='\t')
