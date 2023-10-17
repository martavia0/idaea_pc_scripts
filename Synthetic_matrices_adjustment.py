# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:55:18 2021

@author: Marta Via
"""

import pandas as pd
import numpy as np
import os
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic/old matrices/"
os.chdir(path)
mx_data=pd.read_csv('data_mx.txt', sep='\t', index_col=0)
mx_error=pd.read_csv('error_mx.txt', sep='\t', index_col=0)
mz=mx_data.columns
#%%
mu = 0 # mean
sigma = mx_error.mean(axis=0)
err_list=[]
gauss=pd.Series(np.random.normal(0,sigma, len(mx_error.columns)), index=mz)
mx_data_new= mx_data + errgauss
mx_data.to_csv('mx_data_new.txt', sep='\t')
#%%
import matplotlib.pyplot as plt
mx_data_new.mean(axis=0).plot(marker='o', lw=0)
mx_data.mean(axis=0).plot(marker='o', lw=0)

plt.figure()
plt.scatter(mx_data.iloc[99], mx_data_new.iloc[99])
plt.plot(range(0,1), range(0,1))