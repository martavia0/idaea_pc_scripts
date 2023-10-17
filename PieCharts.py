# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:22:13 2019

@author: Marta Via 

Title: CORRELATION PLOTS EASY


Created on Wed Nov  6 09:48:31 2019

@author: Marta Via
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as d
import datetime as dt
import numpy as np
#%% Importing file:
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2014_05_2015_05/Correlation_Factors")
f= pd.read_csv("LOOA.txt", sep="\t", low_memory=False, skip_blank_lines=False)
#%% Presenting corr. matrix table
m=f.corr()
print(m)
#%% Scatter matrix
pd.plotting.scatter_matrix(f, figsize=(6, 6))
plt.show()
#%% Corr matrix plot
plt.matshow(f.corr())
plt.xticks(range(len(f.columns)), f.columns)
plt.yticks(range(len(f.columns)), f.columns)
plt.colorbar()
plt.show()
#%%