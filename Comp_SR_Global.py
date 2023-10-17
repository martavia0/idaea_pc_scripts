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
R=pd.DataFrame()
S=pd.DataFrame()
#%%
cities=['Barcelona', 'Magadino','Lille', 'Dublin','Bucharest','Marseille','Tartu','Cyprus']
city_acr=['BCN', 'MAG', 'LIL', 'DUB', 'BUC', 'MAR', 'TAR', 'CYP']
wdw_l=14
#%%
cities=['Barcelona','Cyprus']
city_acr=['BCN', 'CYP']
wdw_l=14
#%%
counter=0
R_list=[]
S_list=[]
for city_i in cities:
    path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/"+city_i+'/'
    os.chdir(path)
    a=os.listdir()
    R_list.append(pd.read_csv('Rolling_R2_14_R_'+city_acr[counter]+'.txt', sep="t", low_memory=False))
    S_list.append(pd.read_csv('Rolling_R2_14_S_'+city_acr[counter]+'.txt', sep="\t", low_memory=False))
    counter=counter+1
#%%
hoa_R, hoa_S = [],[]
for i in range(0,len(R_list)):
    hoa_R.append(R_list[i])
    hoa_S.append(S_list[i]['HOA vs. BCff'])
R['HOA vs. BCff']=hoa_R    

    #%5