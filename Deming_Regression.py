# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 07:54:40 2020

@author: Marta Via
"""
"""
Deming regression is equivalent to the maximum likelihood estimation of 
an errors-in-variables model in which the errors for the two variables 
are assumed to be independent and normally distributed, 
and the ratio of their variances, denoted δ, is known.
In practice, this ratio might be estimated from related data-sources; 
however the regression procedure takes no account for possible errors 
in estimating this ratio.
"""

def deming_regresion(df, X, y, delta = 2.698851141):
    '''Takes a pandas DataFrame, name of the 
    columns as strings and the value of delta, 
    and returns the slope and intercept following deming regression formula'''

    cov = df.cov()
    mean_x = df[X].mean()
    mean_y = df[y].mean()
    s_xx = cov[X][X]
    s_yy = cov[y][y]
    s_xy = cov[X][y]

    slope = (s_yy  - delta * s_xx + np.sqrt((s_yy - delta * s_xx) ** 2 + 4 * delta * s_xy ** 2)) / (2 * s_xy)

    intercept = mean_y - slope  * mean_x

    return slope, intercept
#%%
import os as os
import pandas as pd
import numpy as np
import scipy

os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/Aeth_MP_JY")
f1=pd.read_csv("Reg_MAAP_AE33.txt", sep=";")
#%%
print(deming_regresion(f1, 'BC_MAAP_PM10', 'BC_AE33_PM25'))
#%%

from scipy import stats
print(scipy.stats.pearsonr(f2['BC_MAAP_PM10'], f2['BC_AE33_PM25']))
#%%
f2= f1.dropna(axis=0, how='any')
#Y= f1['BC_AE33_PM25'].dropna(axis=0, how='any')
#%%










