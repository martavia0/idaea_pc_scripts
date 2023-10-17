# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:32:10 2021

@author: Marta Via
"""

from scipy.odr import Model, Data, ODR
import scipy.stats 
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
# import statsmodels.api as sm
import glob
import math
# import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#%% Synthetic
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic")
f1 = pd.read_csv("Synthetic.txt", sep="\t", low_memory=False, index_col=False)
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
dr_all = pd.date_range("2011/02/01 00:00", end="2011/12/31")  # dates
city='SYN'
cityname='Synthetic dataset'
# f1['Sc Res_R'] = f1['Sc Res_R']/85.0
# f1['Sc Res_S'] = f1['Sc Res_S']/85.0
# %% SIRTA
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/SIRTA")
f1 = pd.read_csv("SIRTA.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2015/03/01 00:00", end="2017/01/16")  # TAR
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['OA_app_R']=f1['OA_app_Rolling']
f1['OA_app_seas']=f1['OA_app_S']
city = 'SIR'

# %% CYPRUS
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Cyprus")
f1 = pd.read_csv("Cyprus.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2015/03/01 00:00", end="2017/01/16")  # TAR
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['BBOA_seas'] = f1['BBOA_seas'].replace(0, np.nan)
f1['NOx']=f1['NO']+f1['No2']
f1['OOA_Rolling']=f1['LO-OOA_Rolling']+f1['MO-OOA_Rolling']
f1['OOA_seas']=f1['LO-OOA_seas']+f1['MO-OOA_seas']
f1['OA_app_s'] = f1['OA_app_S']
f1['OA_app_seas'] = f1['OA_app_S']
f1['OA_app_Rolling'] = f1['OA_app_R']
f1['OOA_Rolling'] = f1['LO-OOA_Rolling']+f1['MO-OOA_Rolling']
# f1['Sc Res_R'] = f1['Sc Res_R']/73.0
# f1['Sc Res_S'] = f1['Sc Res_S']/73.0
city = 'CYP'
# %% TARTU
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Tartu")
f1 = pd.read_csv("TARTU.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2016/09/01 00:00", end="2017/07/24")  # TAR
f1['OA_app_R'] = f1['HOA_Rolling']+f1['BBOA_Rolling'] + \
    f1['LO-OOA_Rolling']+f1['MO-OOA_Rolling']
f1['OA_app_Rolling'] = f1['OA_app_R']
f1['Sc Res_R']=f1['Sc Res_R']*73.0
f1['Sc Res_S']=f1['Sc Res_S']*73.0
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
# %% MARSEILLE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Marseille")
f1 = pd.read_csv("Marseille.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2017/01/31 00:00", end="2018/04/13")  # MAR
# f1['OA_app_R']=f1['COA_Rolling']+f1['HOA_Rolling']+f1['BBOA_Rolling'] + f1['SHINDOA_Rolling']+f1['LO-OOA_Rolling']+f1['MO-OOA_Rolling']
f1['OA_app_Rolling'] = f1['OA_app_R']
f1['OA_app_S'] = f1['OA_app_s']
f1['OA_app_seas'] = f1['OA_app_s']
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
mask = ~np.isnan(f1['Org']) & ~ np.isnan(f1['OA_app_R']) & ~np.isnan(
    f1['OA_app_S']) & ~(f1['OA_app_R'] == 0) & ~(f1['OA_app_S'] == 0)
f1 = f1[mask]
# %% BUCHAREST
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Bucharest/")
f1 = pd.read_csv("Bucharest.txt", sep="\t", low_memory=False, index_col=False)
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
dr_all = pd.date_range("2016/09/01 00:00", end="2017/08/31")  # BUC
f1['OA_app_S'] = f1['OA_app_s']
f1['OA_app_seas'] = f1['OA_app_s']
f1['OA_app_Rolling'] = f1['OA_app_R']
f1['OA_app_seas'] = f1['OA_app_s']
f1['NOx'] = f1['Nox']

# %% KOSETICE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Kosetice")
f1 = pd.read_csv("Kosetice.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2019/01/08 00:00", end="2019/10/13")  # KOS
f1['HOA_seas'] = pd.to_numeric(f1['HOA_seas'], errors='coerce')
f1['BBOA_seas'] = pd.to_numeric(f1['BBOA_seas'], errors='coerce')
f1['LO-OOA_seas'] = pd.to_numeric(f1['LO-OOA_seas'], errors='coerce')
f1['MO-OOA_seas'] = pd.to_numeric(f1['MO-OOA_seas'], errors='coerce')
f1['OA_app_R'] = pd.to_numeric(f1['OA_app_R'], errors='coerce')
f1['OA_app_s'] = pd.to_numeric(f1['OA_app_s'], errors='coerce')
f1['OOA_Rolling'] = f1['MO-OOA_Rolling']+f1['LO-OOA_Rolling']  # KOS
f1['OOA_seas'] = f1['MO-OOA_seas']+f1['LO-OOA_seas']  # KOS
# %% Importing DUBLIN
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Dublin")
f1 = pd.read_csv("Dublin.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2016/09/01 00:00", end="2017/08/31")  # DUB#%%%
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
# f1['Sc Res_R']=f1['Sc Res_R']/70.0
f1['Sc Res_S'] = f1['Sc Res_S']/70.0
# Scaled residuals do have to be normalized for the seasonal but not for the rolling.
# %% Importing LILLE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Lille")
f1 = pd.read_csv("Lille.txt", sep="\t", low_memory=False)
dr_all = pd.date_range("2016/10/11 00:00", end="2017/08/18")  # LIL
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['Sc Res_R'] = f1['Sc Res_R']  /73.0
f1['Sc Res_S'] = f1['Sc Res_S']  /73.0
f1['OA_app_S'] = f1['OA_app_s']
f1['OA_app_seas'] = f1['OA_app_s']
f1['OA_app_Rolling'] = f1['OA_app_R']
# %% Importing MAGADINO
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Magadino")
f1 = pd.read_csv("Magadino.txt", sep="\t")
dr_all = pd.date_range("2013/08/28 00:00", end="2014/10/30")  # MAG
f1['Sc Res_R'] = f1['Sc Res_R']/70
f1['Sc Res_S'] = f1['Sc Res_S']/70
f1['OA_app_Rolling'] = f1['HOA_Rolling']+f1['BBOA_Rolling'] + \
    f1['LO-OOA_Rolling']+f1['MO-OOA_Rolling']+f1['LOA_Rolling']  # MAG
f1['OA_app_R'] = f1['OA_app_Rolling']
f7 = pd.DataFrame(
    f1, columns=['HOA_seas', 'BBOA_seas', 'LO-OOA_seas', 'MO-OOA_seas', 'LOA_seas'])
f1['OA_app_seas'] = f7.sum(axis=1, skipna=True)  # MAG
f1['OA_app_s'] = f1['OA_app_seas']
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['OA_app_S'] = f1['OA_app_s']
# f1['OOA_Rolling']=f1['MO-OOA_Rolling']+f1['LO-OOA_Rolling']
# %% Importing BARCELONA
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Barcelona")
f1 = pd.read_csv("Barcelona.txt", sep="\t", low_memory=False)
dr_all = pd.date_range("2017/09/21 00:00", end="2018/10/30")  # BCN
f1['Nox'] = f1['NOx']
f1['Sc Res_R'] = f1['Sc Res_R']/92.0
f1['Sc Res_S'] = f1['Sc Res_S']/92.0
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['OA_app_Rolling'] = f1['OA_app_R']
f1['OA_app_seas'] = f1['OA_app_s']
f1['OA_app_S'] = f1['OA_app_s']
# %% Importing WHOLE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Whole")
f1 = pd.read_csv("Whole_dataset.txt", sep="\t", low_memory=False)
f1['Time'] = pd.to_datetime(f1['datetime'], dayfirst=True, errors='coerce')
f1['OOA_Rolling'] = f1['MO-OOA_Rolling']+f1['LO-OOA_Rolling']
f1['OOA_seas'] = f1['MO-OOA_seas']+f1['LO-OOA_seas']
f1['OA_app_Rolling'] = f1['OOA_Rolling']+f1['HOA_Rolling']+f1['LOA_Rolling']+f1['COA_Rolling'] + \
    f1['BBOA_Rolling']+f1['SHINDOA_Rolling'] + \
        f1['Wood_Rolling']+f1['Coal_Rolling']+f1['Peat_Rolling']
f1['OA_app_seas'] = f1['OOA_seas']+f1['HOA_seas']+f1['LOA_seas']+f1['COA_seas'] + \
    f1['BBOA_seas']+f1['SHINDOA_seas'] + \
        f1['Wood_seas']+f1['Coal_seas']+f1['Peat_seas']

dr_all = pd.date_range("2013/08/28 00:00", end="2018/10/03")  # BCN
# %%  R2 function


def R2(a, b):
    c = pd.DataFrame({"a": a, "b": b})
    cm = c.corr(method='pearson')
    r = cm.iloc[0, 1]
    return r**2


def R_log(a, b):
    c = pd.DataFrame({"a": -1/np.log(a.astype('float')),
                     "b": -1/np.log(b.astype('float'))})
    cm = c.corr(method='pearson')
    r = cm.iloc[0, 1]
    return r**2


def slope(b, a):
    c = pd.DataFrame({"a": a, "b": b})
    mask = ~np.isnan(a) & ~np.isnan(b)
    a1 = a[mask]
    b1 = b[mask]
    if (a1.empty) or (b1.empty):
        s = np.nan
    else:
        s, intercept, r_value, p_value, std_err = linregress(a1, b1)
    return s


def f(B, x):
    return B[0]*x + B[1]


def orthoregress(x, y):
    linreg = linregress(x, y)
    linear = Model(f)
    dat = Data(x, y)
    od = ODR(dat, model=linear, beta0=linreg[0:2])
    out = od.run()
    return list(out.beta)


# %% Abs and Rel TS
# ************************* ABSOLUTE AND FRACTION FACTORS PLOT ******************
#
# It will ask whether if you prefer to plot it in Absolute or Fraction concentrations, a well as the city name.
#
#       Red:Rolling     Blue: Seasonal
#
city = input("City name?")
choose = input("Enter -Abs- or -Rel-: ")
if choose == "Abs":
    if city == 'BCN':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling'], 'HOA': f1['HOA_Rolling'],
                         'BBOA': f1['BBOA_Rolling'], 'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                         'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas'], 'HOA': f1['HOA_seas'],
                          'BBOA': f1['BBOA_seas'], 'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'],
                          'OOA': f1['OOA_seas']})
        cityname = "BARCELONA"
    if city == 'MAG':
        R = pd.DataFrame({'datetime': f1.Time, 'LOA': f1['LOA_Rolling'], 'HOA': f1['HOA_Rolling'],
                         'BBOA': f1['BBOA_Rolling'], 'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                         'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'LOA': f1['LOA_seas'], 'HOA': f1['HOA_seas'],
                          'BBOA': f1['BBOA_seas'], 'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'MAGADINO'
    if city == 'LIL':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                          'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'LILLE'
    if city == 'DUB':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'Coal': f1['Coal_Rolling'], 'Wood': f1['Wood_Rolling'], 'Peat': f1['Peat_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                          'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'Coal': f1['Coal_seas'], 'Wood': f1['Wood_seas'], 'Peat': f1['Peat_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'DUBLIN'

    if city == 'BUC':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                          'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'BUCHAREST'
    if city == 'MAR':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling'], 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'], 'SHINDOA': f1['SHINDOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'],
                          'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas'], 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'], 'SHINDOA': f1['SHINDOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'MARSEILLE'
    if city == 'TAR':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'], 'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'TARTU'
    if city == 'CYP':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'], 'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'CYPRUS'
    if city == 'SIR':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling'], 'BBOA': f1['BBOA_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'], 'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 'BBOA': f1['BBOA_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = 'SIRTA'
    if city == 'WHOLE':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling'], 'LOA': f1['LOA_Rolling'], 'HOA': f1['HOA_Rolling'], 'SHINDOA': f1['SHINDOA_Rolling'],
                         'BBOA': f1['BBOA_Rolling'], 'Coal': f1['Coal_Rolling'], 'Wood': f1['Wood_Rolling'], 'Peat': f1['Peat_Rolling'],
                         'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'], 'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas'], 'LOA': f1['LOA_seas'], 'HOA': f1['HOA_seas'], 'SHINDOA': f1['SHINDOA_Rolling'],
                          'BBOA': f1['BBOA_seas'], 'Coal': f1['Coal_seas'], 'Wood': f1['Wood_seas'], 'Peat': f1['Peat_seas'],
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = "Whole dataset"
    if city == 'SYN':
        R = pd.DataFrame({'datetime': f1.Time,  'HOA': f1['HOA_Rolling'],
                         'BBOA': f1['BBOA_Rolling'], 
                         'LO-OOA': f1['LO-OOA_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling'], 'OOA': f1['OOA_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas'], 
                          'BBOA': f1['BBOA_seas'], 
                          'LO-OOA': f1['LO-OOA_seas'], 'MO-OOA': f1['MO-OOA_seas'], 'OOA': f1['OOA_seas']})
        cityname = "Synthetic Dataset"
if choose == "Rel":
    if city == 'BCN':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling']/f1['OA_app_R'], 'HOA': f1['HOA_Rolling']/f1['OA_app_R'],
                          'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'], 'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas']/f1['OA_app_s'], 'HOA': f1['HOA_seas']/f1['OA_app_s'],
                          'BBOA': f1['BBOA_seas']/f1['OA_app_s'], 'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'],
                          'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'], 'OOA': f1['OOA_seas']/f1['OA_app_S']})
    if city == 'MAG':
        R = pd.DataFrame({'datetime': f1.Time, 'LOA': f1['LOA_Rolling']/f1['OA_app_Rolling'], 'HOA': f1['HOA_Rolling']/f1['OA_app_Rolling'],
                          'BBOA': f1['BBOA_Rolling']/f1['OA_app_Rolling'], 'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_Rolling'],
                          'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_Rolling'], 'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'LOA': f1['LOA_seas']/f1['OA_app_s'], 'HOA': f1['HOA_seas']/f1['OA_app_s'],
                          'BBOA': f1['BBOA_seas']/f1['OA_app_s'], 'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'],
                          'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'], 'OOA': f1['OOA_seas']/f1['OA_app_s']})
    if city == 'LIL':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_S']})
    if city == 'DUB':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'Coal': f1['Coal_Rolling']/f1['OA_app_R'], 'Wood': f1['Wood_Rolling']/f1['OA_app_R'], 'Peat': f1['Peat_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'Coal': f1['Coal_seas']/f1['OA_app_s'], 'Wood': f1['Wood_seas']/f1['OA_app_s'], 'Peat': f1['Peat_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_seas']})
    if city == 'KOS':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_S']})
    if city == 'BUC':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_S']})
        cityname = 'BUCHAREST'
    if city == 'MAR':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling']/f1['OA_app_R'], 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'SHINDOA': f1['SHINDOA_Rolling']/f1['OA_app_R'], 'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'],
                          'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'], 'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas']/f1['OA_app_s'], 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'], 'SHINDOA': f1['SHINDOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_S']})
        cityname = 'MARSEILLE'
    if city == 'TAR':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_s']})
        cityname = 'TARTU'
    if city == 'CYP':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_s'], 'BBOA': f1['BBOA_seas']/f1['OA_app_s'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_s'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_s'],
                          'OOA': f1['OOA_seas']/f1['OA_app_S']})
        cityname = 'CYPRUS'
    if city == 'SIR':
        R = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_Rolling']/f1['OA_app_R'], 'BBOA': f1['BBOA_Rolling']/f1['OA_app_R'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_R'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_R'],
                          'OOA': f1['OOA_Rolling']/f1['OA_app_R']})
        S = pd.DataFrame({'datetime': f1.Time, 'HOA': f1['HOA_seas']/f1['OA_app_seas'], 'BBOA': f1['BBOA_seas']/f1['OA_app_seas'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_seas'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_seas'],
                          'OOA': f1['OOA_seas']/f1['OA_app_seas']})
        cityname='SIRTA'
    if city == 'WHOLE':
        R = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_Rolling']/f1['OA_app_Rolling'], 'LOA': f1['LOA_Rolling']/f1['OA_app_Rolling'],
                          'HOA': f1['HOA_Rolling']/f1['OA_app_Rolling'], 'SHINDOA': f1['SHINDOA_Rolling']/f1['OA_app_Rolling'],
                          'BBOA': f1['BBOA_Rolling']/f1['OA_app_Rolling'],  'Coal': f1['Coal_Rolling']/f1['OA_app_Rolling'], 'Wood': f1['Wood_Rolling']/f1['OA_app_Rolling'], 'Peat': f1['Peat_Rolling']/f1['OA_app_Rolling'],
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_Rolling'], 'OOA': f1['OOA_Rolling']/f1['OA_app_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 'COA': f1['COA_seas']/f1['OA_app_seas'], 'LOA': f1['LOA_seas']/f1['OA_app_seas'],
                          'HOA': f1['HOA_seas']/f1['OA_app_seas'], 'SHINDOA': f1['SHINDOA_seas']/f1['OA_app_seas'],
                          'BBOA': f1['BBOA_seas']/f1['OA_app_seas'], 'Coal': f1['Coal_seas']/f1['OA_app_seas'], 'Wood': f1['Wood_seas']/f1['OA_app_seas'], 'Peat': f1['Peat_seas']/f1['OA_app_seas'],
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_seas'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_seas'],
                          'OOA': f1['OOA_seas']/f1['OA_app_seas']})
        cityname = 'Whole Dataset'
    if city == 'SYN':
        R = pd.DataFrame({'datetime': f1.Time,
                          'HOA': f1['HOA_Rolling']/f1['OA_app_Rolling'], 
                          'BBOA': f1['BBOA_Rolling']/f1['OA_app_Rolling'],  
                          'LO-OOA': f1['LO-OOA_Rolling']/f1['OA_app_Rolling'], 'MO-OOA': f1['MO-OOA_Rolling']/f1['OA_app_Rolling'], 'OOA': f1['OOA_Rolling']/f1['OA_app_Rolling']})
        S = pd.DataFrame({'datetime': f1.Time, 
                          'HOA': f1['HOA_seas']/f1['OA_app_seas'], 
                          'BBOA': f1['BBOA_seas']/f1['OA_app_seas'], 
                          'LO-OOA': f1['LO-OOA_seas']/f1['OA_app_seas'], 'MO-OOA': f1['MO-OOA_seas']/f1['OA_app_seas'],
                          'OOA': f1['OOA_seas']/f1['OA_app_seas']})
        cityname = 'Synthetic Dataset'
R = R.set_index('datetime')
S = S.set_index('datetime')
if city == 'BCN' or city == 'MAG' or city=='SYN':
    num = 6
if city == 'LIL' or city == 'KOS' or city == 'BUC' or city == 'TAR' or city == 'CYP' or city=='SIR' or city=='SYN':
    num = 5
if city == 'DUB' or city == 'MAR':
    num = 7
if city == 'WHOLE':
    num = 11
fig, axes = plt.subplots(num, 1, figsize=(28, 26), constrained_layout=True)
fig.canvas.set_window_title('Comparison')
plt.rcParams.update({'font.size': 22})
fig.suptitle(cityname+" ("+choose+")", fontsize=28)
count = 0
for c in range(num):
    name1 = R.columns[c]
    name2 = name1
    axes[c].plot(R.index, R[name1], marker='o', color='red')
    ax2 = axes[c].twinx()
    ax2.plot(S.index, S[name2], marker='o', color='blue')
    axes[c].grid(axis='x')
    axes[c].grid(axis='y')
    axes[c].set_axisbelow(True)
    axes[c].set_title(name1)
    count = count+1
plotname_PRO = "Factors_"+choose+".png"
# plt.savefig(plotname_PRO)
# %%
# PIE OF APPORTIONMENT
#
my_labels = 'HOA', 'BBOA', 'LO-OOA', 'MO-OOA'
# f1['OA_app_Rolling'] = f1['OA_app_R']
# f1['OA_app_seas'] = f1['OA_app_s']
fracs_R = [(f1['HOA_Rolling']/f1['OA_app_Rolling']).mean(),
           # (f1['LOA_Rolling']/f1['OA_app_Rolling']).mean(),
           # (f1['COA_Rolling']/f1['OA_app_Rolling']).mean(),
           # (f1['SHINDOA_Rolling']/f1['OA_app_Rolling']).mean(),
           # (f1['CCOA_Rolling']/f1['OA_app_Rolling']).mean(),
           (f1['BBOA_Rolling']/f1['OA_app_Rolling']).mean(),
           # (f1['Wood_Rolling'] / f1['OA_app_Rolling']).mean(), (f1['Coal_Rolling'] / \
            # f1['OA_app_Rolling']).mean(), (f1['Peat_Rolling']/f1['OA_app_Rolling']).mean(),
           (f1['LO-OOA_Rolling'] / f1['OA_app_Rolling']).mean(), (f1['MO-OOA_Rolling']/f1['OA_app_Rolling']).mean()]
fracs_S = [(f1['HOA_seas']/f1['OA_app_seas']).mean(),
           # (f1['LOA_seas']/f1['OA_app_seas']).mean(),
           # (f1['COA_seas']/f1['OA_app_seas']).mean(),
           # (f1['SHINDOA_seas']/f1['OA_app_seas']).mean(),
            # (f1['CCOA_seas']/f1['OA_app_seas']).mean(),
            (f1['BBOA_seas']/f1['OA_app_seas']).mean(),
           # (f1['Wood_seas']/f1['OA_app_seas']).mean(), (f1['Coal_seas'] / \
            # f1['OA_app_seas']).mean(), (f1['Peat_seas']/f1['OA_app_seas']).mean(),
           (f1['LO-OOA_seas'] / f1['OA_app_seas']).mean(), (f1['MO-OOA_seas']/f1['OA_app_seas']).mean()]

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(cityname, fontsize=25, y=0.75)
axs[0].pie(fracs_R, labels=my_labels, autopct='%1.0f%%',textprops={'fontsize': 18}, shadow=True, colors=['grey', #'firebrick', #darkkhaki', 'sienna','mediumorchid',
           'sienna', 'limegreen', 'darkgreen', ])  # 'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan'
axs[0].set_title('Rolling')
axs[1].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%',textprops={'fontsize': 18}, shadow=True, colors=['grey',#'firebrick', #'darkkhaki', 'sienna','mediumorchid'
           'sienna', 'limegreen', 'darkgreen', ])  # 'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1].set_title('Seasonal')
# %%   DIEL
columns_R = ['HOA_Rolling', 'BBOA_Rolling', 'LO-OOA_Rolling', 'MO-OOA_Rolling', 'OOA_Rolling',# 'COA_Rolling', 'CCOA_Rolling', 
           'HOA_seas',  'BBOA_seas', 'LO-OOA_seas', 'MO-OOA_seas', 'OOA_seas']#'COA_seas', 'CCOA_seas',
# columns_S=['HOA_seas', 'COA_seas', 'SHINDOA_seas', 'BBOA_seas', 'LO-OOA_seas', 'MO-OOA_seas', 'OOA_seas']
diel = pd.DataFrame(f1[columns_R])
diel['Hour'] = f1['Time'].dt.hour
diel_h = diel.groupby('Hour', axis=0).mean()
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(30, 8), gridspec_kw={
                       'height_ratios': [0.5], 'width_ratios': [2, 2, 2, 2, 2]})
fig.suptitle(cityname)
ax[0].plot(diel_h.iloc[:, 0], color='grey', lw=2)
ax[0].plot(diel_h.iloc[:, 5], color='grey', lw=2, ls='--')
ax[0].set_title('HOA')
ax[0].grid()
# ax[1].plot(diel_h.iloc[:, 1], color='mediumorchid', lw=2)
# ax[1].plot(diel_h.iloc[:, 8], color='mediumorchid', lw=2, ls='--')
# ax[1].set_title('COA')
# ax[1].grid()
# ax[2].plot(diel_h.iloc[:, 2], color='firebrick', lw=2)
# ax[2].plot(diel_h.iloc[:, 9], color='firebrick', lw=2, ls='--')
# ax[2].set_title('CCOA')
# ax[2].grid()
ax[1].plot(diel_h.iloc[:, 1], color='brown', lw=2)
ax[1].plot(diel_h.iloc[:, 6], color='brown', lw=2, ls='--')
ax[1].set_title('BBOA')
ax[1].grid()
ax[2].plot(diel_h.iloc[:, 2], color='limegreen', lw=2)
ax[2].plot(diel_h.iloc[:, 7], color='limegreen', lw=2, ls='--')
ax[2].set_title('LO-OOA')
ax[2].grid()
ax[3].plot(diel_h.iloc[:, 3], color='darkgreen', lw=2)
ax[3].plot(diel_h.iloc[:, 7], color='darkgreen', lw=2, ls='--')
ax[3].set_title('MO-OOA')
ax[3].grid()
ax[4].plot(diel_h.iloc[:, 4], color='green', lw=2)
ax[4].plot(diel_h.iloc[:, 8], color='green', lw=2, ls='--')
ax[4].set_title('OOA')
ax[4].grid()

# %% MONTHLY
columns_R = ['HOA_Rolling', 'BBOA_Rolling', 'LO-OOA_Rolling', 'MO-OOA_Rolling', 'OOA_Rolling',
           'HOA_seas', 'BBOA_seas', 'LO-OOA_seas', 'MO-OOA_seas', 'OOA_seas']
monthly = pd.DataFrame(f1[columns_R])
monthly['Month'] = f1['Time'].dt.month
monthly_m = monthly.groupby('Month', axis=0).mean()
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(50, 8), gridspec_kw={
                       'height_ratios': [0.5], 'width_ratios': [2, 2, 2, 2, 2]})
fig.suptitle(cityname)
x = np.arange(0, 11)
x_label = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N']
ax[0].bar(x, height=monthly_m.iloc[:, 0], width=0.3,
          color='black', tick_label=x_label, lw=2)
ax[0].bar(x+0.5, height=monthly_m.iloc[:, 5], width=0.3,
          edgecolor='black', color='white', linewidth=2)
plt.xticks(x, x_label)
ax[0].set_title('HOA')
ax[0].grid()
ax[1].bar(x, monthly_m.iloc[:, 1], width=0.3,
          color='black', tick_label=x_label, lw=2)
ax[1].bar(x+0.5, monthly_m.iloc[:, 6], color='white',
          edgecolor='black', width=0.3)
ax[1].set_title('COA')
ax[1].grid()
# ax[2].bar(x, monthly_m.iloc[:, 2], color='black',
#           width=0.3, tick_label=x_label, lw=2)
# ax[2].bar(x+0.5, monthly_m.iloc[:, 9], color='white',
#           edgecolor='black', width=0.3)
# ax[2].set_title('SHINDOA')
# ax[2].grid()
ax[2].bar(x, monthly_m.iloc[:, 2], color='black',
          width=0.3, tick_label=x_label, lw=2)
ax[2].bar(x+0.5, monthly_m.iloc[:, 7], color='white',
          edgecolor='black', width=0.3)
ax[2].set_title('BBOA')
ax[2].grid()
ax[3].bar(x, monthly_m.iloc[:, 3], color='black',
          width=0.3, tick_label=x_label, lw=2)
ax[3].bar(x+0.5, monthly_m.iloc[:, 8],
          color='white', edgecolor='black', width=0.3)
ax[3].set_title('LO-OOA')
ax[3].grid()
ax[4].bar(x, monthly_m.iloc[:, 4], color='black',
          width=0.3, tick_label=x_label)
ax[4].bar(x+0.5, monthly_m.iloc[:, 9],
          color='white', edgecolor='black', width=0.3)
ax[4].set_title('MO-OOA')
ax[4].grid()
# ax[5].bar(x, monthly_m.iloc[:, 5], color='black',
          # width=0.3, tick_label=x_label)
# ax[5].bar(x+0.5, monthly_m.iloc[:, 12],
#           color='white', edgecolor='black', width=0.3)
# ax[5].set_title('MO-OOA')
# ax[5].grid()
# ax[6].bar(x, monthly_m.iloc[:, 6], color='black',
#           width=0.3, tick_label=x_label)
# ax[6].bar(x+0.5, monthly_m.iloc[:, 13],
#           color='white', edgecolor='black', width=0.3)
# ax[6].set_title('OOA')
# ax[6].grid()

# %% Differences are a function of OA?
per = 14
dif = pd.DataFrame()
dif['datetime'], dif['Org'] = f1.Time, f1.Org
dif['COA'], dif['LOA'], dif['HOA'], dif['SHINDOA'], dif['BBOA'], dif['Wood'], dif['Coal'], dif['Peat'], dif['LO-OOA'], dif['MO-OOA'], dif['OOA'] = f1['COA_Rolling']-f1['COA_seas'], f1['LOA_Rolling']-f1['LOA_seas'], f1['HOA_Rolling']-f1['HOA_seas'], f1['SHINDOA_Rolling'] - \
    f1['SHINDOA_seas'], f1['BBOA_Rolling']-f1['BBOA_seas'], f1['Wood_Rolling']-f1['Wood_seas'], f1['Coal_Rolling']-f1['Coal_seas'], f1['Peat_Rolling'] - \
        f1['Peat_seas'], f1['LO-OOA_Rolling']-f1['LO-OOA_seas'], f1['MO-OOA_Rolling'] - \
            f1['MO-OOA_seas'], f1['OOA_Rolling']-f1['OOA_seas']
plt.scatter(y=dif['OOA'], x=dif['Org'])
for i in dif.columns:
    if per == 'Period':
        print(R2(dif[i], dif['Org']))
        continue
    if i == 'datetime':
        continue
    a = (data_averager(dif[i], dif['datetime'], dr_all, per, 1).outdata)
    b = (data_averager(dif['Org'], dif['datetime'], dr_all, per, 1).outdata)
    print(i, R2(a, b))
# %% Differences are a function of other differences?
# a=(dif.iloc[:,2:].corr())**2
f = plt.figure(figsize=(19, 15))
plt.matshow(a, fignum=f.number)
plt.xticks(range(0, len(dif.columns[2:])),
           dif.columns[2:], fontsize=14, rotation=45)
plt.yticks(range(0, len(dif.columns[2:])), dif.columns[2:], fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Absolute difference $R^2$. Rolling - Seasonal.', fontsize=20);
rel_dif = pd.DataFrame()
rel_dif['HOA'] = dif['HOA']/((f1['HOA_Rolling']+f1['HOA_seas'])/2.0)*100.0
rel_dif['BBOA'] = dif['BBOA']/((f1['BBOA_Rolling']+f1['BBOA_seas'])/2.0)*100.0
rel_dif['LO-OOA'] = dif['LO-OOA'] / \
    ((f1['LO-OOA_Rolling']+f1['LO-OOA_seas'])/2.0)*100.0
rel_dif['MO-OOA'] = dif['MO-OOA'] / \
    ((f1['MO-OOA_Rolling']+f1['MO-OOA_seas'])/2.0)*100.0
rel_dif['OOA'] = dif['OOA']/((f1['OOA_Rolling']+f1['OOA_seas'])/2.0)*100.0
rel_dif['COA'], rel_dif['LOA'], = dif['COA'] / \
    ((f1['COA_Rolling']+f1['COA_seas'])/2.0)*100.0, dif['LOA'] / \
     ((f1['LOA_Rolling']+f1['LOA_seas'])/2.0)*100.0
rel_dif['WCOA'] = dif['Wood']/((f1['Wood_Rolling']+f1['Wood_seas'])/2.0)*100.0
rel_dif['CCOA'], rel_dif['PCOA'] = dif['Coal'] / \
    ((f1['Coal_Rolling']+f1['Coal_seas'])/2.0)*100.0, dif['Peat'] / \
     ((f1['Peat_Rolling']+f1['Peat_seas'])/2.0)*100.0
rel_dif['SHINDOA'] = dif['SHINDOA'] / \
    ((f1['SHINDOA_Rolling']+f1['SHINDOA_seas'])/2.0)*100.0
a = (rel_dif.iloc[:, 2:].corr())**2
f, ax = plt.subplots(figsize=(19, 15))
boxprops = dict(linestyle='-', linewidth=0.6)
meanprops = dict(marker='o', linewidth=0.6,
                 markeredgecolor='black', markerfacecolor='black')
rbp = rel_dif.boxplot(showfliers=False, showmeans=True, boxprops=boxprops,
                      meanprops=meanprops, color='black', fontsize=18)
rbp.set_ylabel('Relative Errors (%)', fontsize=22)
per = 1
for i in dif.columns:
    if per == 'Period':
        print(i, R2(dif[i], (f1[i+'_Rolling']+f1[i+'_seas'])/2.0))
        continue
    if i == 'datetime' or i == 'Org':
        continue
    a = (data_averager(dif[i], dif['datetime'], dr_all, per, 1).outdata)
    b = (data_averager((f1[i+'_seas']+f1[i+'_Rolling']) /
         2.0, f1.Time, dr_all, per, 1).outdata)
    print(i, R2(a, b))
    plt.scatter(y=dif['SHINDOA'], x=(
        f1['SHINDOA_seas']+f1['SHINDOA_Rolling'])/2.0)
# print(i, R2(dif['LO-OOA'],(f1[i+'_Rolling']+f1[i+'_seas'])/2.0))
# plt.scatter(y=dif['BBOA'],x=(f1['LO-OOA_seas']+f1['LO-OOA_Rolling'])/2.0)
# plt.scatter(y=f1['BBOA_Rolling'],x=f1['LO-OOA_Rolling'], c=dif['LO-OOA'])
plt.scatter(y=dif['BBOA'], x=(dif['LO-OOA']),
            c=(f1['LO-OOA_seas']+f1['LO-OOA_Rolling'])/2.0)
# %% NOx
# f1['Nox'] = f1['NOx']
f1['BCwb'] = f1['BCbb']
# %%
# ******* ROLLING R2 PLOT FOR EXTERNAL CORRELATIONS*******
#
# The outcoming graphs are the correlation timelines for R, S and their intercomparison
#
#       Red:Rolling     Blue: Seasonal
#


def External_Corr(wdw_l, shift):  # wdw_l is the window length
    lR = []
    lS = []
    lT=[]
    l_dates = []
    for i in range(0, len(dr_all), shift):
        st_d = dr_all[i]
        dr_14 = pd.date_range(st_d, periods=wdw_l)
        en_d = dr_14[-1]
        mask_i = (f1['Time'] > st_d) & (f1['Time'] <= en_d)
        f3 = f1.loc[mask_i]
        l_dates.append(dr_14[0])
        if city == 'DUB':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['HOA_Rolling'], f3['NOx']), R2(f3['Coal_Rolling'], f3['BCwb']),
                     R2(f3['Wood_Rolling'], f3['BCwb']), R2(
                         f3['Peat_Rolling'], f3['BCwb']),
                     R2(f3['OOA_Rolling'], f3['NH4']), R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BCff']), R2(f3['HOA_seas'], f3['NOx']), R2(f3['Coal_seas'], f3['BCwb']),
                     R2(f3['Wood_seas'], f3['BCwb']), R2(
                         f3['Peat_seas'], f3['BCwb']),
                     R2(f3['OOA_seas'], f3['NH4']), R2(f3['MO-OOA_seas'], f3['SO4'])]
        if city == 'MAR':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['HOA_Rolling'], f3['NOx']), R2(f3['BBOA_Rolling'], f3['BCwb']),
                     R2(f3['SHINDOA_Rolling'], f3['N2_1020nm']),
                     R2(f3['OOA_Rolling'], f3['NH4']), R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BCff']), R2(f3['HOA_seas'], f3['NOx']), R2(f3['BBOA_seas'], f3['BCwb']),
                     R2(f3['SHINDOA_seas'], f3['N2_1020nm']),
                     R2(f3['OOA_seas'], f3['NH4']), R2(f3['MO-OOA_seas'], f3['SO4'])]
        if city == 'LIL' or city=='SIR':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['BBOA_Rolling'], f3['BCwb']),
                     R2(f3['OOA_Rolling'], f3['NH4']), R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BCff']),  R2(f3['BBOA_seas'], f3['BCwb']),
                     R2(f3['OOA_seas'], f3['NH4']), R2(f3['MO-OOA_seas'], f3['SO4'])]
        if city == 'CYP':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['HOA_Rolling'], f3['NOx']), R2(f3['BBOA_Rolling'], f3['BCwb']),
                      R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BCff']),  R2(f3['HOA_seas'], f3['NOx']), R2(f3['BBOA_seas'], f3['BCwb']),
                    R2(f3['MO-OOA_seas'], f3['SO4'])]
        if city != 'DUB' and city != 'MAR' and city != 'LIL' and city != 'CYP' and city!= 'SYN' and city !='WHO' and city!='SIR':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['HOA_Rolling'], f3['NOx']), R2(f3['BBOA_Rolling'], f3['BCwb']),
                     R2(f3['OOA_Rolling'], f3['NH4']),R2(f3['MO-OOA_Rolling'], f3['SO4'])] # 
            rsq_S = [R2(f3['HOA_seas'], f3['BCff']),  R2(f3['HOA_seas'], f3['NOx']), R2(f3['BBOA_seas'], f3['BCwb']),#
                     R2(f3['OOA_seas'], f3['NH4']),R2(f3['MO-OOA_seas'], f3['SO4'])]
        if city =='SYN':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BC']), R2(f3['HOA_Rolling'], f3['NOx']),
                     R2(f3['OOA_Rolling'], f3['NH4']),R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BC']),  R2(f3['HOA_seas'], f3['NOx']),
                     R2(f3['OOA_seas'], f3['NH4']),R2(f3['MO-OOA_seas'], f3['SO4'])]  
            rsq_T = [R2(f3['HOA_T'], f3['BC']),  R2(f3['HOA_T'], f3['NOx']),
                     R2(f3['SOA_T'], f3['NH4']),R2(f3['SOA_T'], f3['SO4'])]  
        if city =='WHO':
            rsq_R = [R2(f3['HOA_Rolling'], f3['BCff']), R2(f3['BBOA_Rolling'], f3['BCwb']),
                    R2(f3['MO-OOA_Rolling'], f3['SO4'])]
            rsq_S = [R2(f3['HOA_seas'], f3['BC']),R2(f3['BBOA_seas'], f3['BCwb']),
                    R2(f3['MO-OOA_seas'], f3['SO4'])]       
        lR.append(rsq_R)
        lS.append(rsq_S)
        lT.append(rsq_T)
    if city == 'DUB':
        R = pd.DataFrame(lR, columns=['HOA vs. BCff', 'HOA vs. NOx', 'Coal vs. BCwb', 'Wood vs. BCwb', 'Peat vs. BCwb',  # DUB
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff', 'HOA vs. NOx', 'Coal vs. BCwb', 'Wood vs. BCwb', 'Peat vs. BCwb',  # DUB
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
    if city == 'MAR':
        R = pd.DataFrame(lR, columns=['HOA vs. BCff', 'HOA vs. NOx', 'BBOA vs. BCwb', 'SHINDOA vs. UF Industry',  # MAR
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff', 'HOA vs. NOx', 'BBOA vs. BCwb', 'SHINDOA vs. UF Industry',
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
    if city == 'LIL' or city=='SIR':
        R = pd.DataFrame(lR, columns=['HOA vs. BCff', 'BBOA vs. BCwb',
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff',  'BBOA vs. BCwb',
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
    if city == 'CYP':
        R = pd.DataFrame(lR, columns=['HOA vs. BCff', 'HOA vs. NOx', 'BBOA vs. BCwb',
                                      'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff', 'HOA vs. NOx',  'BBOA vs. BCwb',
                                      'MO-OOA vs. SO4'])
    if city != 'DUB' and city != 'MAR' and city != 'LIL' and city != 'CYP' and city!='SYN' and city!='WHO' and city!='SIR' :
        R = pd.DataFrame(lR, columns=['HOA vs. BCff', 'HOA vs. NOx', 'BBOA vs. BCwb',
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff', 'HOA vs. NOx',  'BBOA vs. BCwb',
                                      'OOA vs. NH4', 'MO-OOA vs. SO4'])
    if city == 'SYN':
        R = pd.DataFrame(lR, columns=['HOA vs. BC', 'HOA vs. NOx','OOA vs. NH4', 'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BC', 'HOA vs. NOx','OOA vs. NH4', 'MO-OOA vs. SO4'])
        T= pd.DataFrame(lT, columns=['HOA vs. BC', 'HOA vs. NOx','OOA vs. NH4', 'MO-OOA vs. SO4'])
    if city=='WHO':
        R = pd.DataFrame(lR, columns=['HOA vs. BCff',  'BBOA vs. BCwb',
                                       'MO-OOA vs. SO4'])
        S = pd.DataFrame(lS, columns=['HOA vs. BCff', 'BBOA vs. BCwb',
                                       'MO-OOA vs. SO4'])    
    R['datetime'] = l_dates
    R = R.set_index('datetime')
 #   R.to_csv('Rolling_R2_'+str(wdw_l)+'_R_'+city+'.txt')
    fig_R = R.plot(subplots=True, figsize=(25, 20), grid=True)[1].get_figure()
    fig_R.savefig('Rolling_R2_'+str(wdw_l)+'_R_'+city+'.png')
    S['datetime'] = l_dates
    S = S.set_index('datetime')
#    S.to_csv('Rolling_R2_'+str(wdw_l)+'_S_'+city+'.txt')
    fig_S = S.plot(subplots=True, figsize=(25, 20), grid=True)[1].get_figure()
    fig_S.savefig('Rolling_R2_'+str(wdw_l)+'_S_'+city+'.png')
#
#       ROLLING-SEASONAL ROLLING R2 comparison
#
    num = len(S.columns)
    fig, axes = plt.subplots(num, 1, figsize=(28, 26), constrained_layout=True)
    fig.canvas.set_window_title('Comparison')
    plt.rcParams.update({'font.size': 22})
    fig.suptitle(cityname + "\n "+str(wdw_l)+"days window", fontsize=28)
    for c in range(num):
        name1 = R.columns[c-1]
        name2 = name1
        axes[c].plot(R.index, R[name1], marker='o', color='red')
        ax2 = axes[c].twinx()
        ax2.plot(S.index, S[name2], marker='o', color='blue')
        axes[c].grid(axis='x')
        axes[c].grid(axis='y')
        axes[c].set_axisbelow(True)
        axes[c].set_title(name1)
        plotname_PRO = "Mobile_R2_Comparison_"+str(wdw_l)+".png"
        plt.savefig(plotname_PRO)
    return R-S, R, S,T


# %% Calculation of externals
wdw_l = 14
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic")#"+cityname)
Dif_14, R14, S14, T14 = External_Corr(14, 1) #, T14
R14.to_csv('Rolling_R2_'+str(wdw_l)+'_R_'+city+'.txt', sep='\t')
S14.to_csv('Rolling_R2_'+str(wdw_l)+'_S_'+city+'.txt', sep='\t')

# %% By months
Dif_14['Month'] = Dif_14.index.month
Dif_14.boxplot(by='Month', figsize=(20, 20), title='Monthly')
# %% Mean and median of correlation difference
# Calculation of the median and mean of each correlation difference
print('Median:\n', Dif_14.median().round(4))
print('Mean:\n', Dif_14.mean().round(4))
print(Dif_14.quantile(0.25).round(4))
print(Dif_14.quantile(0.75).round(4))
# print(abs(Dif_14.mean() - Dif_14.quantile(0.25)).round(4))
# print(abs(Dif_14.mean() -Dif_14.quantile(0.75)).round(4))

# %% For period long R2
print('ROLLING:')
print('HOAvs.BCff', R2(f1['HOA_Rolling'], f1['BCff']))
print('HOA vs. NOx:', R2(f1['HOA_Rolling'], f1['Nox']))
print('BBOA vs. BCwb:', R2(f1['BBOA_Rolling'], f1['BCwb']))
# print('SHINDOA vs. UF:',R2(f1['SHINDOA_Rolling'], f1['BCwb']))
# print('Wood vs. BCwb:',R2(f1['Wood_Rolling'], f1['BCwb']))
# print('Coal vs. BCwb:',R2(f1['Coal_Rolling'], f1['BCwb']))
# print('Peat vs. BCwb:',R2(f1['Peat_Rolling'], f1['BCwb']))
print('MO-OOA vs. SO4:', R2(f1['MO-OOA_Rolling'], f1['SO4']))
print('OOA vs. NH4:', R2(f1['OOA_Rolling'], f1['NH4']))
print('Seasonal:')
print('HOAvs.BCff', R2(f1['HOA_seas'], f1['BCff']))
print('HOA vs. NOx:', R2(f1['HOA_seas'], f1['Nox']))
print('BBOA vs. BCwb:', R2(f1['BBOA_seas'], f1['BCwb']))
# print('SHINDOA vs. UF:',R2(f1['SHINDOA_seas'], f1['BCwb']))
# print('Wood vs. BCwb:',R2(f1['Wood_seas'], f1['BCwb'])
)
# print('Coal vs. BCwb:',R2(f1['Coal_seas'], f1['BCwb']))
# print('Peat vs. BCwb:',R2(f1['Peat_seas'], f1['BCwb']))
print('MO-OOA vs. SO4:', R2(f1['MO-OOA_seas'], f1['SO4']))
print('OOA vs. NH4:', R2(f1['OOA_seas'], f1['NH4']))

# %% Caluclation of the area below 0
for i in range(0, len(Dif_14.columns)):
    values, bins, patches=plt.hist(Dif_14.iloc[:, i], 100, density = True)
    zero=np.where(bins > 0.0)
    Neg_Area=sum(values[0:zero[0][0]])/sum(values)
    print(Dif_14.columns[i], Neg_Area.round(2)*100)
#%%
# isneg_R = R14['OOA vs. NH4']<0 or  R14['OOA vs. NH4']>1
R14['OOA vs. NH4']=R14['OOA vs. NH4'][R14['OOA vs. NH4']<0]
#%% Hist-KDE of the R2 difference RS
# del Dif_14['Month']
fig, ax=plt.subplots(figsize = (8, 5))
plt.tick_params(labelsize = 14)
if city !='SYN':
    fig, ax=plt.subplots(figsize = (10,8))
    plt.tick_params(labelsize = 16)
    Dif_14.plot.kde(ax = ax, grid = True, linewidth = 2.0, color = ['black', 'grey', 'brown', 'limegreen', 'darkgreen'])  # 'darkkhaki','sienna', 'tan','lightskyblue'
    plt.legend(['HOA vs. BC$_{ff}$','HOA vs. NO$_X$', 'OOA vs. NH$_4$', 'MO-OOA vs. SO$_4$'],fontsize = 14)# 'BBOA vs. BC$_{wb}$'
    ax.set_xlabel('R$^2_{Rolling}$ - R$^2_{Seasonal}$', fontsize = 18)
    plt.title(cityname+'\n'+" 14-days R$^2$", fontsize = 18)
    ax.set_ylabel('Frequency', fontsize = 16)

if city=='SYN':
    fig, ax=plt.subplots(1,3,figsize = (12,4), sharey=True, sharex=True)
    T14.plot.kde(ax = ax[0], grid = True, color = ['black', 'grey', 'limegreen', 'darkgreen'],linewidth = 2.0)     
    R14.plot.kde(ax = ax[1], grid = True, color = ['black', 'grey', 'limegreen', 'darkgreen'],linewidth = 2.0) 
    S14.plot.kde(ax = ax[2], grid = True, color = ['black', 'grey', 'limegreen', 'darkgreen'],linewidth = 2.0) 
    ax[0].legend('')
    ax[1].legend('')
    ax[2].legend(fontsize=14, loc='upper left')
    ax[0].set_ylabel('Frequency (adim.)',fontsize=16)
    ax[1].set_xlabel('R$^2$ between method and ancillary measurements',fontsize=16)
    ax[0].set_title('Truth',fontsize=16)
    ax[1].set_title('Rolling',fontsize=16)
    ax[2].set_title('Seasonal',fontsize=16)

    plt.xlim([0,1])
# %% HISTOGRAMS WITH FIT
# hist, bins, nose= plt.hist(Dif_14.iloc[:,i], 100, density=1, alpha=0.5)
fig, axes=plt.subplots(ncols = len(Dif_14.columns),
                       figsize = (25, 4), sharey = True)
fig.text(0.5, 1, cityname, ha = 'center', fontsize = 25)
for i in range(0, len(Dif_14.columns)):
    axes[i].hist(Dif_14.iloc[:, i], 50, density = 1, alpha = 0.5, color = 'k')
    axes[i].grid()
    axes[i].set_title(Dif_14.columns[i], fontsize = 20)
    mu, sigma=scipy.stats.norm.fit(Dif_14.iloc[:, i].dropna())
    best_fit_line=scipy.stats.norm.pdf(bins, mu, sigma)
    axes[i].text(-0.28, 22, '$\mu$='+str(mu.round(2)) + \
                 '\n$\sigma$='+str(sigma.round(2)), fontsize = 16)
    axes[i].plot(bins, best_fit_line, color = 'black')
    axes[i].set_xlim(-0.3, 0.3)
fig.text(0.5, -0.1, 'Difference Rolling vs. Seasonal R$^2$ *', ha = 'center')
fig.text(0.90, -0.08, '* Period = 14 days\n bins = 50',
         ha = 'right', fontsize = 14)


# %%
#       ROLLING R2 for the Rolling Seasonal Dataset Comparison
#
def Rolling_vs_Seasonal_Factors(per):
    per=90
    l=[]
    l2=[]
    l_dt=[]
    for i in range(0, len(dr_all)):
        st_d=dr_all[i]
        dr_14=pd.date_range(st_d, periods = per)
        en_d=dr_14[-1]
        mask_i=(f1['Time'] > st_d) & (f1['Time'] <= en_d)
        f3=f1.loc[mask_i]
        # if f3['HOA_Rolling'].empty:
        #     continue
        l_dt.append(st_d)
        if city == 'BCN':
            rsq=[R2(f3['COA_Rolling'], f3['COA_seas']), R2(f3['HOA_Rolling'], f3['HOA_seas']),
                   R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
    #        slp = [orthoregress(f3['COA_Rolling'], f3['COA_seas'])[0], orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0],
    #               orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
    #               orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        if city == 'MAG':
            rsq= [R2(f3['LOA_Rolling'], f3['LOA_seas']), R2(f3['HOA_Rolling'], f3['HOA_seas']),
                   R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
        #     slp = [orthoregress(f3['LOA_Rolling'], f3['LOA_seas'])[0], orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0],
        #             orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
        #             orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        # if city == 'LIL':
            rsq= [R2(f3['HOA_Rolling'], f3['HOA_seas']),
                   R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
            slp= [orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0],
                   orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])]
        if city == 'DUB':
            rsq= [R2(f3['HOA_Rolling'], f3['HOA_seas']),
                   R2(f3['Wood_Rolling'], f3['Wood_seas']), R2(
                       f3['Coal_Rolling'], f3['Coal_seas']), R2(f3['Peat_Rolling'], f3['Peat_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
            slp= [orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0],
                   orthoregress(f3['Coal_Rolling'], f3['Coal_seas'])[0], orthoregress(
                       f3['Wood_Rolling'], f3['Wood_seas'])[0], orthoregress(f3['Peat_Rolling'], f3['Peat_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        if city == 'KOS':
            rsq= [R2(f3['HOA_Rolling'], f3['HOA_seas']), R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
            slp= [orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0], orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        if city == 'BUC':
            rsq= [R2(f3['HOA_Rolling'], f3['HOA_seas']), R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])]
            slp= [orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0], orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        if city == 'MAR':
            rsq= [R2(f3['COA_Rolling'], f3['COA_seas']), R2(f3['HOA_Rolling'], f3['HOA_seas']),
                   R2(f3['BBOA_Rolling'], f3['BBOA_seas']), R2(
                       f3['SHINDOA_Rolling'], f3['SHINDOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
            slp= [orthoregress(f3['COA_Rolling'], f3['COA_seas'])[0], orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0],
                   orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0], orthoregress(
                       f3['SHINDOA_Rolling'], f3['SHINDOA_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        if city == 'TAR' or city == 'CYP':
            rsq= [R2(f3['HOA_Rolling'], f3['HOA_seas']), R2(f3['BBOA_Rolling'], f3['BBOA_seas']),
                   R2(f3['LO-OOA_Rolling'], f3['LO-OOA_seas']), R2(f3['MO-OOA_Rolling'], f3['MO-OOA_seas']), R2(f3['OOA_Rolling'], f3['OOA_seas'])]
            slp= [orthoregress(f3['HOA_Rolling'], f3['HOA_seas'])[0], orthoregress(f3['BBOA_Rolling'], f3['BBOA_seas'])[0],
                   orthoregress(f3['LO-OOA_Rolling'], f3['LO-OOA_seas'])[0], orthoregress(f3['MO-OOA_Rolling'], f3['MO-OOA_seas'])[0], orthoregress(f3['OOA_Rolling'], f3['OOA_seas'])[0]]
        l.append(rsq)
    #    l2.append(slp)

    if city == 'BCN':
        col= ['COA R vs. COA S', 'HOA R vs. HOA S', 'BBOA R vs. BBOA S', 'LO-OOA R vs. LO-OOA S', 'MO-OOA R vs. MO-OOA S', 'OOA R vs. OOA S']
    if city == 'MAG':
        col= ['LOA R vs. LOA S', 'HOA R vs. HOA S', 'BBOA R vs. BBOA S',
            'LO-OOA R vs. LO-OOA S', 'MO-OOA R vs. MO-OOA S', 'OOA R vs. OOA S']
    if city == 'LIL' or city == 'KOS' or city =='BUC' or city=='TAR' or city=='CYP':    
        col = ['HOA R vs. HOA S', 'BBOA R vs. BBOA S',
            'LO-OOA R vs. LO-OOA S', 'MO-OOA R vs. MO-OOA S', 'OOA R vs. OOA S']
    if city == 'DUB':
        col = ['HOA R vs. HOA S', 'Wood R vs. Wood S', 'Coal R vs. Coal S', 'Peat R vs. Peat S', 'LO-OOA R vs. LO-OOA S','MO-OOA R vs. MO-OOA S','OOA R vs. OOA S']
    if city == 'MAR':
        col = ['COA R vs. COA S', 'HOA R vs. HOA S', 'BBOA R vs. BBOA S', 'SHINDOA R vs. SHINDOA S', 'LO-OOA R vs. LO-OOA S', 'MO-OOA R vs. MO-OOA S','OOA R vs. OOA S'] 
    R = pd.DataFrame(l, columns=col)
    S = pd.DataFrame(l2, columns=col)
    R['datetime'] = l_dt
    S['datetime'] = l_dt
    R.to_csv('Rolling_R2_Rolling_vs_Seas_f_'+str(per)+'.txt', sep="\t")
    S.to_csv('Rolling_slope_Rolling_vs_Seas_f_'+str(per)+'.txt')
    R = R.set_index('datetime')
    S = S.set_index('datetime')

    # PLot rolling R2 for the Rolling dataset
    fig_R, axes = plt.subplots(nrows=num, ncols=1, sharex=True, figsize=(25, 20), constrained_layout=True)
    fig_R.canvas.set_window_title('Comparison')
    fig_R.suptitle(cityname+" (Absolute)", fontsize=28)
    plt.rcParams.update({'font.size': 22})
    count = 0
    for c in range(num):
        name1 = R.columns[c]
        name2 = name1
        axes[c].plot(R.index, R[name1], marker='o', color='black')
        ax2 = axes[c].twinx()
        axes[c].set_ylabel('R²')
        ax2.set_ylabel('Slope (x=S, y=R)', color='grey')
        ax2.plot(S.index, S[name2], marker='o', color='grey')
        axes[c].grid(axis='x')
        axes[c].grid(axis='y')
        axes[c].set_axisbelow(True)
        axes[c].set_title(name1)
        count = count+1
    fig_R.savefig('Rolling_R2slope_Rolling_vs_Seas_'+str(per)+'_f.png')
    return R, S
# %%
R, S=Rolling_vs_Seasonal_Factors(1)
print('R2:','\n',R.mean())
print('Slope:','\n',S.mean())
# %% Overall R2 and slopes (whole period)
# R2
print(('R2'))
if city=='BCN' or city=='MAR':
    print('COA:',R2(f1['COA_Rolling'], f1['COA_seas']))
if city=='MAG':
    print('LOA:',R2(f1['LOA_Rolling'], f1['LOA_seas']))
if city=='DUB':  
    print('Coal burning:',R2(f1['Coal_Rolling'], f1['Coal_seas']))
    print('Wood burning:',R2(f1['Wood_Rolling'], f1['Wood_seas']))
    print('Peat burning',R2(f1['Peat_Rolling'], f1['Peat_seas']))
if city=='MAR':
    print('SHINDOA:',R2(f1['SHINDOA_Rolling'], f1['SHINDOA_seas']))
print('HOA:',R2(f1['HOA_Rolling'], f1['HOA_seas']))
if city != 'DUB':
    print('BBOA:',R2(f1['BBOA_Rolling'], f1['BBOA_seas']))
print('LO-OOA',R2(f1['LO-OOA_Rolling'], f1['LO-OOA_seas']))
print('MO-OOA',R2(f1['MO-OOA_Rolling'], f1['MO-OOA_seas']))
print('OOA',R2(f1['OOA_Rolling'], f1['OOA_seas']))
# SLOPES
print('slopes')
if city=='BCN' or city=='MAR':
    print('COA:',slope(f1['COA_Rolling'], f1['COA_seas']))
if city=='MAG':
    print('LOA:',slope(f1['LOA_Rolling'], f1['LOA_seas']))
if city=='DUB':  
    print('Coal burning:',slope(f1['Coal_Rolling'], f1['Coal_seas']))
    print('Wood burning:',slope(f1['Wood_Rolling'], f1['Wood_seas']))
    print('Peat burning',slope(f1['Peat_Rolling'], f1['Peat_seas']))
if city=='MAR':
    print('SHINDOA:',slope(f1['SHINDOA_Rolling'], f1['SHINDOA_seas']))
print('HOA:',slope(f1['HOA_Rolling'], f1['HOA_seas']))
if city != 'DUB':
    print('BBOA:',slope(f1['BBOA_Rolling'], f1['BBOA_seas']))
print('LO-OOA',slope(f1['LO-OOA_Rolling'], f1['LO-OOA_seas']))
print('MO-OOA',slope(f1['MO-OOA_Rolling'], f1['MO-OOA_seas']))
print('OOA',slope(f1['OOA_Rolling'], f1['OOA_seas']))
# %%
def data_averager(indata, intime, outtime, nod, shift):  # nod
    # Average of a nod-day period in a rolling fashion with 1 day shifts .
    l = []
    lt = []
    for i in range(0, len(outtime), shift):
        st_d = outtime[i]
        dr = pd.date_range(start=outtime[i], periods=nod+1)
        en_d = dr[-1]
        mask_i = (intime > st_d) & (intime <= en_d)
        fmask = indata.loc[mask_i]
        if len(fmask)<100 and nod>2:
             continue
        l.append(fmask.mean(skipna=True))
        lt.append(st_d)
    outdata = pd.DataFrame(l, columns=['outdata'])
    outdata['datetime'] = lt
#    outdata['dt']=outtime
    return outdata

# %%
# 1 day resolution
# Absolute
def Factors_averaging(method, per): #method=(Abs, Rel)_seas
    RS_resol=pd.DataFrame()
    if method == 'Abs':
        if city == 'BCN' or city=='WHOLE':
            RS_resol = pd.DataFrame({#'datetime': dr_all,
                                     'COA_Rolling': data_averager(f1['COA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'COA_Seasonal': data_averager(f1['COA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'MAG':
            RS_resol = pd.DataFrame({'datetime': dr_all,
                                   'LOA_Rolling': data_averager(f1['LOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LOA_Seasonal': data_averager(f1['LOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas'], f1['Time'], dr_all, per, 1).outdata})
      
        if city == 'DUB':
            RS_resol = pd.DataFrame({'datetime': dr_all,
                                     'Coal_Rolling': data_averager(f1['Coal_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'Coal_Seasonal': data_averager(f1['Coal_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'Wood_Rolling': data_averager(f1['Wood_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'Wood_Seasonal': data_averager(f1['Wood_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'Peat_Rolling': data_averager(f1['Peat_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'Peat_Seasonal': data_averager(f1['Peat_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas'], f1['Time'], dr_all, per, 1).outdata})
    
        if city == 'MAR':
            RS_resol = pd.DataFrame({'datetime': f1['datetime'],
                                     'COA_Rolling': data_averager(f1['COA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'COA_Seasonal': data_averager(f1['COA_seas'], f1['Time'], dr_all, per, 1).outdata,                                     
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'SHINDOA_Rolling': data_averager(f1['SHINDOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'SHINDOA_Seasonal': data_averager(f1['SHINDOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas'], f1['Time'], dr_all, per, 1).outdata})    

        if city == 'KOS' or city == 'TAR' or city == 'BUC' or city=='CYP' or city=='LIL' or city=='SYN':
            RS_resol = pd.DataFrame({'datetime': f1['datetime'],
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas'], f1['Time'], dr_all, per, 1).outdata, 
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas'], f1['Time'], dr_all, per, 1).outdata})
            
    if method == 'Rel':
        if city == 'BCN':
            RS_resol = pd.DataFrame({'datetime': dr_all,
                                     'COA_Rolling': data_averager(f1['COA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'COA_Seasonal': data_averager(f1['COA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'MAG':
            RS_resol = pd.DataFrame({'datetime': dr_all,
                                   'LOA_Rolling': data_averager(f1['LOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LOA_Seasonal': data_averager(f1['LOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'DUB':
            RS_resol = pd.DataFrame({'datetime': dr_all,
                                     'Coal_Rolling': data_averager(f1['Coal_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'Coal_Seasonal': data_averager(f1['Coal_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'Wood_Rolling': data_averager(f1['Wood_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'Wood_Seasonal': data_averager(f1['Wood_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'Peat_Rolling': data_averager(f1['Peat_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'Peat_Seasonal': data_averager(f1['Peat_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'MAR':
            RS_resol = pd.DataFrame({'datetime': f1['datetime'],
                                     'COA_Rolling': data_averager(f1['COA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'COA_Seasonal': data_averager(f1['COA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'SHINDOA_Rolling': data_averager(f1['SHINDOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'SHINDOA_Seasonal': data_averager(f1['SHINDOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_R'], f11['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'LIL' or city == 'TAR' or city == 'BUC' or city=='CYP':
            RS_resol = pd.DataFrame({'datetime': f1['datetime'],
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_R'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_s'], f1['Time'], dr_all, per, 1).outdata})
        if city == 'SYN':
            RS_resol = pd.DataFrame({'datetime': f1['datetime'],
                                     'HOA_Rolling': data_averager(f1['HOA_Rolling']/f1['OA_app_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'HOA_Seasonal': data_averager(f1['HOA_seas']/f1['OA_app_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'BBOA_Rolling': data_averager(f1['BBOA_Rolling']/f1['OA_app_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'BBOA_Seasonal': data_averager(f1['BBOA_seas']/f1['OA_app_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'LO-OOA_Rolling': data_averager(f1['LO-OOA_Rolling']/f1['OA_app_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'LO-OOA_Seasonal': data_averager(f1['LO-OOA_seas']/f1['OA_app_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'MO-OOA_Rolling': data_averager(f1['MO-OOA_Rolling']/f1['OA_app_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'MO-OOA_Seasonal': data_averager(f1['MO-OOA_seas']/f1['OA_app_seas'], f1['Time'], dr_all, per, 1).outdata,
                                     'OOA_Rolling': data_averager(f1['OOA_Rolling']/f1['OA_app_Rolling'], f1['Time'], dr_all, per, 1).outdata, 'OOA_Seasonal': data_averager(f1['OOA_seas']/f1['OA_app_seas'], f1['Time'], dr_all, per, 1).outdata})
    return RS_resol
# %%
# ***************** t-test for Rolling and Seasonal time series ***********
#
# If we observe a large p-value, for example larger than 0.05 or 0.1,
# then we cannot reject the null hypothesis of identical average scores.
# If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
# then we reject the null hypothesis of equal averages.
#
# PRINT t-test RESULTS averaged
method='Abs'
# city='LIL'
per=90
RS_resol=Factors_averaging(method, per) #method=(Abs, Rel)_seas
RS_resol['OA_app_Rolling']=RS_resol['HOA_Rolling']+RS_resol['LO-OOA_Rolling']+RS_resol['MO-OOA_Rolling']+RS_resol['BBOA_Rolling']#+RS_resol['COA_Rolling']#+RS_resol['SHINDOA_Rolling']  #+RS_resol['Peat_Rolling']+RS_resol['Wood_Rolling']+RS_resol['Coal_Rolling']#
RS_resol['OA_app_Seasonal']=RS_resol['HOA_Seasonal']+RS_resol['LO-OOA_Seasonal']+RS_resol['MO-OOA_Seasonal']+RS_resol['BBOA_Seasonal']#+RS_resol['COA_Seasonal']#+RS_resol['SHINDOA_Seasonal']#+RS_resol['Peat_Seasonal']+RS_resol['Wood_Seasonal']+RS_resol['Coal_Seasonal']#+RS_resol['COA_Rolling'] +RS_resol['BBOA_Seasonal']
print(method, per)
if city=='BCN' or city=='MAR' or city=='WHOLE':
    print('COA_T:',ttest_ind(RS_resol['COA_Rolling'], RS_resol['COA_Seasonal'], nan_policy='omit', equal_var=False))
if city=='MAG':
    print('LOA:',ttest_ind(RS_resol['LOA_Rolling'], RS_resol['LOA_Seasonal'], nan_policy='omit', equal_var=False))
print('HOA:',ttest_ind(RS_resol['HOA_Rolling'],RS_resol['HOA_Seasonal'], nan_policy='omit', equal_var=False))
if city!='DUB':
    print('BBOA:',ttest_ind(RS_resol['BBOA_Rolling'],RS_resol['BBOA_Seasonal'], nan_policy='omit', equal_var=False))
if city=='DUB':
    print('Wood:',ttest_ind(RS_resol['Wood_Rolling'], RS_resol['Wood_Seasonal'], nan_policy='omit', equal_var=False))
    print('Coal:',ttest_ind(RS_resol['Coal_Rolling'], RS_resol['Coal_Seasonal'], nan_policy='omit', equal_var=False))
    print('Peat:',ttest_ind(RS_resol['Peat_Rolling'], RS_resol['Peat_Seasonal'], nan_policy='omit', equal_var=False))
if city=='MAR':
    print('SHINDOA:',ttest_ind(RS_resol['SHINDOA_Rolling'], RS_resol['SHINDOA_Seasonal'], nan_policy='omit', equal_var=False))

print('LO-OOA:',ttest_ind(RS_resol['LO-OOA_Rolling'],
                RS_resol['LO-OOA_Seasonal'], nan_policy='omit', equal_var=False))
print('MO-OOA:',ttest_ind(RS_resol['MO-OOA_Rolling'],
                RS_resol['MO-OOA_Seasonal'], nan_policy='omit', equal_var=False))
print('OOA:',ttest_ind(RS_resol['OOA_Rolling'],
                RS_resol['OOA_Seasonal'], nan_policy='omit', equal_var=False))
print('OA:',ttest_ind(RS_resol['OA_app_Rolling'],
                RS_resol['OA_app_Seasonal'], nan_policy='omit', equal_var=False))


# %% aLL PERIOD
method='Abs'
print(method)
if method=='Abs':
    print('HOA:',ttest_ind(f1['HOA_Rolling'], f1['HOA_seas'],nan_policy='omit', equal_var=False))
    if city !='DUB':
        print('BBOA:',ttest_ind(f1['BBOA_Rolling'], f1['BBOA_seas'],nan_policy='omit', equal_var=False))
    if city=='BCN' or city=='MAR' or city=='WHOLE':
        print('COA:',ttest_ind(f1['COA_Rolling'], f1['COA_seas'], nan_policy='omit', equal_var=False))#BCN
    if city=='MAG':
        print('LOA:',ttest_ind(f1['LOA_Rolling'], f1['LOA_seas'], nan_policy='omit', equal_var=False))
    if city=='DUB':
        print('Wood:',ttest_ind(f1['Wood_Rolling'], f1['Wood_seas'], nan_policy='omit', equal_var=False))
        print('Coal:',ttest_ind(f1['Coal_Rolling'], f1['Coal_seas'], nan_policy='omit', equal_var=False))
        print('Peat:',ttest_ind(f1['Peat_Rolling'], f1['Peat_seas'], nan_policy='omit', equal_var=False))
    if city=='MAR':
        print('SHINDOA:',ttest_ind(f1['SHINDOA_Rolling'], f1['SHINDOA_seas'], nan_policy='omit', equal_var=False))
    print('LO-OOA:',ttest_ind(f1['LO-OOA_Rolling'], f1['LO-OOA_seas'],nan_policy='omit', equal_var=False))
    print('MO-OOA:',ttest_ind(f1['MO-OOA_Rolling'], f1['MO-OOA_seas'], nan_policy='omit', equal_var=False))
    print('OOA:',ttest_ind(f1['OOA_Rolling'], f1['OOA_seas'], nan_policy='omit', equal_var=False))
    print('OA:',ttest_ind(f1['OA_app_Rolling'], f1['OA_app_seas'], nan_policy='omit', equal_var=False))

if method=='Rel':
    print('HOA:',ttest_ind(f1['HOA_Rolling']/f1['OA_app_Rolling'], f1['HOA_seas'] /f1['OA_app_seas'], nan_policy='omit', equal_var=False))
    if city !='DUB':
        print('BBOA:',ttest_ind(f1['BBOA_Rolling']/f1['OA_app_Rolling'], f1['BBOA_seas'] /
                        f1['OA_app_seas'], nan_policy='omit', equal_var=False))
    if city=='BCN' or city=='MAR':
        print('COA:',ttest_ind(f1['COA_Rolling']/f1['OA_app_Rolling'], f1['COA_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))#BCN
    if city=='MAG':
        print('LOA:',ttest_ind(f1['LOA_Rolling']/f1['OA_app_Rolling'], f1['LOA_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))
    if city=='DUB':
        print('Wood:',ttest_ind(f1['Wood_Rolling']/f1['OA_app_Rolling'], f1['Wood_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))
        print('Coal:',ttest_ind(f1['Coal_Rolling']/f1['OA_app_Rolling'], f1['Coal_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))
        print('Peat:',ttest_ind(f1['Peat_Rolling']/f1['OA_app_Rolling'], f1['Peat_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))
    if city=='MAR':
        print('SHINDOA:',ttest_ind(f1['SHINDOA_Rolling']/f1['OA_app_Rolling'], f1['SHINDOA_seas'], nan_policy='omit', equal_var=False))
    print('LO-OOA:',ttest_ind(f1['LO-OOA_Rolling']/f1['OA_app_Rolling'], f1['LO-OOA_seas']/f1['OA_app_seas'],nan_policy='omit', equal_var=False))
    print('MO-OOA:',ttest_ind(f1['MO-OOA_Rolling']/f1['OA_app_Rolling'], f1['MO-OOA_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))
    print('OOA:',ttest_ind(f1['OOA_Rolling']/f1['OA_app_Rolling'], f1['OOA_seas']/f1['OA_app_seas'], nan_policy='omit', equal_var=False))


# %% S
per = 90
print(per)
f1['OA_app_R']=f1['OA_app_Rolling']
f1['OA_app_S']=f1['OA_app_seas']
mask = ~np.isnan(f1['Org']) & ~ np.isnan(f1['OA_app_R']) & ~np.isnan(f1['OA_app_S']) &~(f1['OA_app_R']==0)& ~(f1['OA_app_S']==0)
f11=f1[mask]
RS = pd.DataFrame({'Time':data_averager(f11['Org'], f11['Time'], dr_all, per,1).datetime, 
                   'Org': data_averager(f11['Org'], f11['Time'], dr_all, per,1).outdata, 
                   'OA_app_R':data_averager(f11['OA_app_R'],f11['Time'],dr_all,per,1).outdata,
                   'OA_app_S':data_averager(f11['OA_app_s'],f11['Time'],dr_all,per,1).outdata})
mask=~np.isnan(RS['Org']) & ~ np.isnan(RS['OA_app_R']) & ~np.isnan(RS['OA_app_S'])
print('Rolling:', R2(RS['Org'][mask], RS['OA_app_R'][mask]),orthoregress(RS['Org'][mask], RS['OA_app_R'][mask]))  # )

print('Seasonal:', R2(RS['Org'][mask], RS['OA_app_S'][mask]), orthoregress(RS['Org'][mask], RS['OA_app_S'][mask]))  # )

# %%
per=14
f_plot = pd.DataFrame({'Org': RS['Org'], 'OA_app_R': RS['OA_app_R'], 'OA_app_S': RS['OA_app_S']})
f_plot.index = (data_averager(f11['Org'], f11['Time'], dr_all, per, 1)['datetime'])
f_plot.plot(figsize=(18, 8), title=city+'  ('+str(per)+' days)',grid=True)
# %%OA vs. OA app plots!
# V for different resolutions!
    fig, ax = plt.subplots(figsize=(8, 8))
    mask = ~np.isnan(RS['Org']) & ~np.isnan(
        RS['OA_app_R']) & ~(RS['OA_app_R'] == 0)
    mask2 = ~np.isnan(RS['Org']) & ~np.isnan(
        RS['OA_app_S']) & ~(RS['OA_app_S'] == 0)
    plt.scatter(x=RS["Org"][mask2], y=RS['OA_app_S'][mask2], color='blue')
    plt.scatter(x=RS["Org"][mask], y=RS['OA_app_R'][mask], color='red')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    mr = orthoregress(RS['Org'][mask], RS['OA_app_R'][mask])[0]
    br = orthoregress(RS['Org'][mask], RS['OA_app_R'][mask])[1]
    ms = orthoregress(RS['Org'][mask2], RS['OA_app_S'][mask2])[0]
    bs = orthoregress(RS['Org'][mask2], RS['OA_app_S'][mask2])[1]
    plt.legend(['Seasonal', 'Rolling',  'Reg R', 'Reg S'])  # BCN
    plt.plot(RS["Org"], mr*RS["Org"]+br, color='darkred')  # BCN
    plt.plot(RS["Org"], ms*RS["Org"]+bs, color='darkblue')  # BCN
    ax.set_xlabel('OA ($ µg·m^{-3}$)')
    ax.set_ylabel('Apportioned OA ($µg·m^{-3}$)')
    plt.title(cityname+' ('+str(per)+' days resolution)')
    plt.grid(10)
    plt.text(15, 1, 'Rolling: R$^2$ = ' + str(R2(f1["Org"], f1['OA_app_R']))[:4]+'\n'+'y='+str(mr)[:4]+'·x + '+str(br)[:4] + '\n' +
             'Seasonal: R$^2$ = ' + str(R2(RS["Org"], RS['OA_app_S']))[:4]+'\n' + 'y=' + str(ms)[:4]+'·x + ' + str(bs)[:4] + '\n', fontsize=14)
# %%
# Period OAapp vs. Org
mask = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_R']) & ~(f1['OA_app_R'] == 0)
mask2 = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_s']) & ~(f1['OA_app_s'] == 0)
print('OA vs OA Rolling:', R2(f1['Org'][mask], f1['OA_app_R'][mask]), orthoregress(
    f1['Org'][mask], f1['OA_app_R'][mask]))
print('OA vs. OA Seasonal:', R2(f1['Org'], f1['OA_app_s']), orthoregress(
    f1['Org'][mask2], f1['OA_app_s'][mask2]))

# %%#%%
# Period OAapp vs. Org
mask = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_R']) & ~(f1['OA_app_R'] == 0)
mask2 = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_s']) & ~(f1['OA_app_s'] == 0)
print('OA vs OA Rolling:', R2(f1['Org'][mask], f1['OA_app_R'][mask]), orthoregress(
    f1['Org'][mask], f1['OA_app_R'][mask]))
print('OA vs. OA Seasonal:', R2(f1['Org'], f1['OA_app_s']), orthoregress(
    f1['Org'][mask2], f1['OA_app_s'][mask2]))
# plt.scatter(x=f1["Org"], y=f1['OA_app_s'], color='blue')
fig, ax = plt.subplots(figsize=(8, 8))

mask = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_R']) & ~(f1['OA_app_R'] == 0)
mask2 = ~np.isnan(f1['Org']) & ~np.isnan(
    f1['OA_app_s']) & ~(f1['OA_app_s'] == 0)
plt.scatter(x=f1["Org"][mask2], y=f1['OA_app_s'][mask2], color='blue')
plt.scatter(x=f1["Org"][mask], y=f1['OA_app_R'][mask], color='red')
plt.xlim(0, 40)  # MAG
plt.ylim(0, 40)  # MAG
mr = orthoregress(f1['Org'][mask], f1['OA_app_R'][mask])[0]
br = orthoregress(f1['Org'][mask], f1['OA_app_R'][mask])[1]
ms = orthoregress(f1['Org'][mask2], f1['OA_app_s'][mask2])[0]
bs = orthoregress(f1['Org'][mask2], f1['OA_app_s'][mask2])[1]
plt.legend(['Seasonal', 'Rolling',  'Reg R', 'Reg S'])  # BCN
plt.plot(f1["Org"], mr*f1["Org"]+br, color='darkred')  # BCN
plt.plot(f1["Org"], ms*f1["Org"]+bs, color='darkblue')  # BCN

ax.set_xlabel('OA ($ µg·m^{-3}$)')
ax.set_ylabel('Apportioned OA ($µg·m^{-3}$)')
plt.title(cityname)
plt.grid(9)
plt.text(25, 2, 'Rolling: R$^2$ = ' + str(R2(f1["Org"], f1['OA_app_R']))[:4]+'\n'+'y='+str(mr)[:4]+'·x + '+str(br)[:4] + '\n')

# %%
'''SCALED RESIDUALS HISTOGRAMS'''

# %% RESOLUTION AVERAGER ERRORS

def Averager_Roll_E(df, freq_days, Sc_or_Q, city, avg_or_sum):  # SC_or_Q = Sc Res or Q_Res, avg_or_sum = "avg" or "sum" 
    l = []
    dt = []
    for i in range(0, len(dr_all), 1):  # 1,7
        st_d = dr_all[i]
        dr_14 = pd.date_range(st_d, periods=freq_days+1)
        en_d = dr_14[-1]
        mask_i = (df['Time'] > st_d) & (df['Time'] <= en_d)  #
        f3 = df.loc[mask_i]
        if avg_or_sum == "avg":
            value = [f3[Sc_or_Q+'_R'].mean(skipna=True), f3[Sc_or_Q+'_S'].mean(skipna=True)]
        if avg_or_sum == "sum":
            value = [
                f3[Sc_or_Q+'_R'].sum(skipna=True), f3[Sc_or_Q+'_S'].sum(skipna=True)]
        l.append(value)
        dt.append(dr_all[i])
    R_week = pd.DataFrame(l, columns=[Sc_or_Q+'_R', Sc_or_Q+'_S'])
    R_week['datetime'] = dt
    R_week = R_week.set_index('datetime')
    return R_week
# %%#SCR=pd.DataFrame({'ScRes R': pd.to_numeric(f1['Sc Res_R']), 'ScRes S':pd.to_numeric(f1['Sc Res_S'])})
mask= (f1['Sc Res_R']<f1['Sc Res_R'].quantile(0.95)) & (f1['Sc Res_R']>f1['Sc Res_R'].quantile(0.05))&(f1['Sc Res_S']<f1['Sc Res_S'].quantile(0.95)) & (f1['Sc Res_S']>f1['Sc Res_S'].quantile(0.05))
SCR = pd.DataFrame({'Sc Res_R': (f1['Sc Res_R'][mask]), 'Sc Res_S': (f1['Sc Res_S'][mask])})
SCR.plot()
#%%
print(SCR.mean())
SCR['Time']=f1['Time']
SCR.boxplot(showmeans=True)
#%%
# **** SCALED RESIDUALS HISTOGRAM KDE ***************
fig, ax = plt.subplots(figsize=(8,6))
kde_scres=SCR.plot.kde(ax=ax, legend=True, color=(['r', 'b']))
ax.set_xlabel("Scaled Residuals", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.legend(fontsize=18)
plt.plot()
# ax.set_xlim(-0.8,0.8)
ax.legend(['Rolling', 'Seasonal'], loc='upper right',fontsize=16)
plt.tick_params(labelsize=16)
ax.grid(True)
plt.suptitle(cityname, fontsize=18, y=1)
plt.title('Window length = Period', fontsize=14)
#
# %%
SCR_d = Averager_Roll_E(SCR, 1, 'Sc Res', city, "avg")
SCR_w = Averager_Roll_E(SCR, 7, 'Sc Res', city, "avg")
SCR_14 = Averager_Roll_E(SCR, 14, 'Sc Res', city, "avg")
SCR_m = Averager_Roll_E(SCR, 30, 'Sc Res', city, "avg")
SCR_90 = Averager_Roll_E(SCR, 90, 'Sc Res', city, "avg")
SCR.to_csv("Sc_Res_orig_res.txt", sep="\t")
SCR_d.to_csv("Sc_Res_daily_res.txt", sep="\t")
SCR_w.to_csv("Sc_Res_weekly_res.txt", sep="\t")
SCR_14.to_csv("Sc_Res_14d_res.txt", sep="\t")
# %% HISt
fig, axs= plt.subplots(ncols=4, figsize=(25,6))#, sharey=True)
fig.text(0.5, 1,cityname, ha='center', fontsize=28)
# del SCR['Time']
SCR.plot.hist(ax=axs[0],bins=50,title='Period', color=(['r', 'b']),legend=False, grid=True, alpha=0.5,xlim=(-0.01, 0.01))
SCR_90.plot.hist(ax=axs[1],bins=50, title='Season',color=(['r', 'b']),legend=False, grid=True,alpha=0.5,xlim=(-0.01,0.01))
SCR_14.plot.hist(ax=axs[2],bins=50,title='Fortnight', color=(['r', 'b']),legend=False, grid=True,alpha=0.5,xlim=(-0.01,0.01))
SCR_d.plot.hist(ax=axs[3],bins=50,title='Day', color=(['r', 'b']),legend=False, grid=True,alpha=0.5,xlim=(-0.01,0.01))
fig.legend(['Rolling', 'Seasonal'],loc=(0.72,0.890), fontsize=16)
fig.text(0.5, -0.05, 'Scaled residuals (adim.)', ha='center',fontsize=26)
#%% KDE
fig, axs= plt.subplots(ncols=4,figsize=(25,6))#, sharey=True)
fig.text(0.5, 1,cityname, ha='center', fontsize=28)
SCR.plot.kde(ax=axs[0],title='WL=Period', color=(['r', 'b']),legend=False, grid=True)#,xlim=(-20,20))
SCR_90.plot.kde(ax=axs[1], title='WL=Season',color=(['r', 'b']),legend=False, grid=True)#,xlim=(-0.25,0.25))
SCR_14.plot.kde(ax=axs[2],title='WL=14 days', color=(['r', 'b']),legend=False, grid=True)#,xlim=(-0.5,0.5))
SCR_d.plot.kde(ax=axs[3],title='WL=1 day', color=(['r', 'b']),legend=False, grid=True)#0,xlim=(-3,3))
fig.legend(['Rolling', 'Seasonal'],loc=(0.72,0.850), fontsize=16)
fig.text(0.5, -0.05, 'Scaled residuals (adim.)', ha='center',fontsize=26)
# %% HIST CALCULATIONS
nbins=100
SCR=SCR.dropna()
hist_p_R, bins_p_R, nose_p_R= plt.hist(SCR['Sc Res_R'],  nbins,density=1,  alpha=0.5)
hist_p_S, bins_p_S, nose_p_S= plt.hist(SCR['Sc Res_S'],  nbins, density=1, alpha=0.5)
mupR, sigmapR = scipy.stats.norm.fit(SCR['Sc Res_R'])
mupS, sigmapS = scipy.stats.norm.fit(SCR['Sc Res_S'])
best_fit_line_pR = scipy.stats.norm.pdf(bins_p_R, mupR, sigmapR)
best_fit_line_pS = scipy.stats.norm.pdf(bins_p_S, mupS, sigmapS)

SCR_90=SCR_90.dropna()
hist_90_R, bins_90_R, nose_90_R= plt.hist(SCR_90['Sc Res_R'],nbins, density=1,   alpha=0.5)
hist_90_S, bins_90_S, nose_90_S= plt.hist(SCR_90['Sc Res_S'], nbins, density=1,  alpha=0.5)
mu_90_R, sigma_90_R = scipy.stats.norm.fit(SCR_90['Sc Res_R'])
mu_90_S, sigma_90_S = scipy.stats.norm.fit(SCR_90['Sc Res_S'])
best_fit_line_90_R = scipy.stats.norm.pdf(bins_90_R, mu_90_R, sigma_90_R)
best_fit_line_90_S = scipy.stats.norm.pdf(bins_90_S, mu_90_S, sigma_90_S)

SCR_14=SCR_14.dropna()
hist_14_R, bins_14_R, nose_14_R= plt.hist(SCR_14['Sc Res_R'],nbins, density=1,   alpha=0.5)
hist_14_S, bins_14_S, nose_14_S= plt.hist(SCR_14['Sc Res_S'], nbins, density=1,  alpha=0.5)
mu_14_R, sigma_14_R = scipy.stats.norm.fit(SCR_14['Sc Res_R'])
mu_14_S, sigma_14_S = scipy.stats.norm.fit(SCR_14['Sc Res_S'])
best_fit_line_14_R = scipy.stats.norm.pdf(bins_14_R, mu_14_R, sigma_14_R)
best_fit_line_14_S = scipy.stats.norm.pdf(bins_14_S, mu_14_S, sigma_14_S)

SCR_d=SCR_d.dropna()
hist_d_R, bins_d_R, nose_d_R= plt.hist(SCR_d['Sc Res_R'],nbins, density=1,   alpha=0.5)
hist_d_S, bins_d_S, nose_d_S= plt.hist(SCR_d['Sc Res_S'], nbins, density=1,  alpha=0.5)
mu_d_R, sigma_d_R = scipy.stats.norm.fit(SCR_d['Sc Res_R'])
mu_d_S, sigma_d_S = scipy.stats.norm.fit(SCR_d['Sc Res_S'])
best_fit_line_d_R = scipy.stats.norm.pdf(bins_d_R, mu_d_R, sigma_d_R)
best_fit_line_d_S = scipy.stats.norm.pdf(bins_d_S, mu_d_S, sigma_d_S)

# %%
fig, axs= plt.subplots(ncols=4,figsize=(20,6))
fig.text(0.5, 1,cityname, ha='center', fontsize=28)
nbins=10
axs[0].hist(SCR['Sc Res_R'], bins=50,density=1, color='r', alpha=0.5)
axs[0].hist(SCR['Sc Res_S'], bins=50,density=1, color='b', alpha=0.5)
axs[0].plot(bins_p_R, best_fit_line_pR, color='red',lw=3)
axs[0].plot(bins_p_S, best_fit_line_pS, color='blue',lw=3)
axs[0].grid()
axs[0].set_xlim(-1,0.1)
# axs[0].text(-0.02,30,'Rolling:\n$\mu$='+str(mupR.round(2))+'\n$\sigma$='+str(sigmapR.round(2)), fontsize=16)
# axs[0].text(-0.02,50,'Seasonal:\n$\mu$='+str(mupS.round(2))+'\n$\sigma$='+str(sigmapS.round(2)), fontsize=16)
axs[0].set_ylabel('Density')
axs[0].set_title('WL = Period')
nbins=50

axs[1].hist(SCR_90['Sc Res_R'], bins=50,density=1, color='r', alpha=0.5)
axs[1].hist(SCR_90['Sc Res_S'], bins=50,density=1, color='b', alpha=0.5)
axs[1].plot(bins_90_R, best_fit_line_90_R, color='red',lw=3)
axs[1].plot(bins_90_S, best_fit_line_90_S, color='blue',lw=3)
axs[1].set_title('WL = Season')
axs[1].set_xlim(-0.5,0.1)
# axs[1].set_ylim(0,4000)
# axs[1].text(0.25, 7,'Rolling:\n$\mu$='+str(mu_90_R.round(2))+'\n$\sigma$='+str(sigma_90_R.round(2)), fontsize=16)
# axs[1].text(0.25,5,'Seasonal:\n$\mu$='+str(mu_90_S.round(2))+'\n$\sigma$='+str(sigma_90_S.round(2)), fontsize=16)
axs[1].grid()

axs[2].hist(SCR_14['Sc Res_R'], bins=50,density=1, color='r', alpha=0.5)
axs[2].hist(SCR_14['Sc Res_S'], bins=50,density=1, color='b', alpha=0.5)
axs[2].plot(bins_14_R, best_fit_line_14_R, color='red',lw=3)
axs[2].plot(bins_14_S, best_fit_line_14_S, color='blue',lw=3)
axs[2].grid()
axs[2].set_xlim(-1,1)
# axs[1].set_ylim(0,4000)
axs[2].set_title('WL = 14 days')
# axs[2].text(-0.8, 2.3,'Rolling:\n$\mu$='+str(mu_14_R.round(2))+'\n$\sigma$='+str(sigma_14_R.round(2)), fontsize=16)
# axs[2].text(-0.8,1.8,'Seasonal:\n$\mu$='+str(mu_14_S.round(2))+'\n$\sigma$='+str(sigma_14_S.round(2)), fontsize=16)

axs[3].hist(SCR_d['Sc Res_R'], bins=10,density=1, color='r', alpha=0.5)
axs[3].hist(SCR_d['Sc Res_S'], bins=10,density=1, color='b', alpha=0.5)
axs[3].plot(bins_d_R, best_fit_line_d_R, color='red',lw=3)
axs[3].plot(bins_d_S, best_fit_line_d_S, color='blue',lw=3)
# axs[3].set_xlim(-1,1)
axs[3].grid()
axs[3].set_title('WL = 1 day')
# axs[3].text(-2,1.1,'Rolling:\n$\mu$='+str(mu_d_R.round(2))+'\n$\sigma$='+str(sigma_d_R.round(2)), fontsize=16)
# axs[3].text(-2,0.75,'Seasonal:\n$\mu$='+str(mu_d_S.round(2))+'\n$\sigma$='+str(sigma_d_S.round(2)), fontsize=16)


# fig.legend(['Rolling', 'Seasonal'],loc=(0.078,0.68))
fig.text(0.5, -0.05, 'Scaled residuals (adim.)', ha='center',fontsize=26)
fig.legend(['Rolling', 'Seasonal'], fontsize=15, loc=(0.1,0.02))
fig.text(0.8, -0.02, '(Number of bins = 10)', ha='center',fontsize=16)

os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/'+cityname+'/Profiles')
HOA_R = pd.read_csv("TDP_HOA_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
HOA_S = pd.read_csv("TDP_HOA_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
LO_R = pd.read_csv("TDP_LO_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
LO_S = pd.read_csv("TDP_LO_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
MO_R = pd.read_csv("TDP_MO_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
MO_S = pd.read_csv("TDP_MO_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)

# OOA_R = pd.read_csv("TDP_OOA_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
# OOA_S = pd.read_csv("TDP_OOA_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
if city=='BCN' or city=='MAR' or city=='SYN':
    COA_R=pd.read_csv("TDP_COA_Rolling.txt", sep="\t", keep_date_col=True)#), infer_datetime_format=True)
    COA_S=pd.read_csv("TDP_COA_Seasonal.txt", sep="\t", infer_datetime_format=True)
if city=='MAR':
    SHINDOA_R=pd.read_csv("TDP_ShINDOA_Rolling.txt", sep="\t", infer_datetime_format=True)
    SHINDOA_S=pd.read_csv("TDP_ShINDOA_Seasonal.txt", sep="\t", infer_datetime_format=True)
if city=='MAG':
    LOA_R=pd.read_csv("TDP_LOA_Rolling.txt", sep="\t", infer_datetime_format=True)
    LOA_S=pd.read_csv("TDP_LOA_Seasonal.txt", sep="\t", infer_datetime_format=True)
    OOA_R.drop(['0'],inplace=True,axis=1)
    OOA_S.drop(['0'],inplace=True,axis=1)
if city!='DUB':
    BBOA_R = pd.read_csv("TDP_BBOA_Rolling.txt", sep="\t", infer_datetime_format=True)
    BBOA_S = pd.read_csv("TDP_BBOA_Seasonal.txt", sep="\t", infer_datetime_format=True)  
if city=='DUB':
    Coal_R=pd.read_csv("TDP_Coal_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
    Coal_S=pd.read_csv("TDP_Coal_Seasonal.txt", sep="\t", infer_datetime_format=True)
    Wood_R=pd.read_csv("TDP_Wood_Rolling.txt", sep="\t", infer_datetime_format=True)
    Wood_S=pd.read_csv("TDP_Wood_Seasonal.txt", sep="\t", infer_datetime_format=True)
    Peat_R=pd.read_csv("TDP_Peat_Rolling.txt", sep="\t", infer_datetime_format=True)
    Peat_S=pd.read_csv("TDP_Peat_Seasonal.txt", sep="\t", infer_datetime_format=True)

d = pd.DataFrame()
d['Time'] = pd.to_datetime(LO_R['datetime'], dayfirst=True, errors='coerce')
# MO_S['Time'] = d['Time']
# %%
print(MO_S.iloc[9],'\t',MO_R.iloc[9])
# %% Time dependent profiles comparison R, S
l = []
lt = []
per=90
method='Rlog'#input('R2' or 'Rlog')
for i in range(0, len(dr_all)-per):
    st_d = dr_all[i]
    dr_14 = pd.date_range(st_d, periods=per)
    en_d = dr_14[-1]
    mask_i = (d['Time'] > st_d) & (d['Time'] <= en_d)
    if city=='BCN':
        fRC = COA_R.loc[mask_i]
        fSC = COA_S.loc[mask_i]
        names_columns=['HOA', 'BBOA','COA','LO-OOA', 'MO-OOA', 'OOA']
    if city=='MAG':
        fRLOA = LOA_R.loc[mask_i]
        fSLOA = LOA_S.loc[mask_i]
        names_columns=[ 'LOA','HOA', 'BBOA', 'LO-OOA', 'MO-OOA', 'OOA']
    if city=='MAR':
        fRC = COA_R.loc[mask_i]
        fSC = COA_S.loc[mask_i]
        fRS = SHINDOA_R.loc[mask_i]
        fSS = SHINDOA_S.loc[mask_i]
        names_columns=[ 'COA','HOA', 'BBOA', 'SHINDOA','LO-OOA', 'MO-OOA', 'OOA']
    if city!='DUB':  
        fRB = BBOA_R.loc[mask_i]
        fSB = BBOA_S.loc[mask_i]
    if city=='DUB':
        fRCoal=Coal_R.loc[mask_i]
        fSCoal=Coal_S.loc[mask_i]
        fRWood=Wood_R.loc[mask_i]
        fSWood=Wood_S.loc[mask_i]
        fRPeat=Peat_R.loc[mask_i]
        fSPeat=Peat_S.loc[mask_i]
        names_columns=[ 'COA','HOA', 'Wood', 'Coal', 'Peat', 'LO-OOA', 'MO-OOA', 'OOA']
    else:
        names_columns=['HOA', 'BBOA', 'LO-OOA', 'MO-OOA', 'OOA']
    fRH = HOA_R.loc[mask_i]
    fSH = HOA_S.loc[mask_i]
    fRL = LO_R.loc[mask_i]
    fSL = LO_S.loc[mask_i]
    fRM = MO_R.loc[mask_i]
    fSM = MO_S.loc[mask_i]
    fRO = OOA_R.loc[mask_i]
    fSO = OOA_S.loc[mask_i]
    rsq=[]
    if fRL.empty:
        continue
    if method == 'R':
        rsq = [R2(fRH.mean(axis=0), fSH.mean(axis=0)), 
  #             R2(fRCoal.mean(axis=0), fSCoal.mean(axis=0)), R2(fRPeat.mean(axis=0), fSPeat.mean(axis=0)),R2(frPeat.mean(axis=0), fSCoal.mean(axis=0)),
              R2(fRC.mean(axis=0), fSC.mean(axis=0)),               
               R2(fRB.mean(axis=0), fSB.mean(axis=0)), 
               R2(fRC.mean(axis=0), fSC.mean(axis=0)), 
 #              R2(fRLOA.mean(axis=0), fSLOA.mean(axis=0)),  
               R2(fRL.mean(axis=0), fSL.mean(axis=0)), R2(fRM.mean(axis=0), fSM.mean(axis=0)),
               R2(fRO.mean(axis=0), fSO.mean(axis=0))]
    if method == 'Rlog':
        rsq = [R_log(fRH.mean(axis=0), fSH.mean(axis=0)),R_log(fRB.mean(axis=0), fSB.mean(axis=0)), 
               # R_log(fRLOA.mean(axis=0), fSLOA.mean(axis=0)), 
               R_log(fRC.mean(axis=0), fSC.mean(axis=0)), 
               R_log(fRS.mean(axis=0), fSS.mean(axis=0)), 
               R_log(fRL.mean(axis=0), fSL.mean(axis=0)), R_log(fRM.mean(axis=0), fSM.mean(axis=0)),
               R_log(fRO.mean(axis=0), fSO.mean(axis=0))]
#               R_log(fRC.mean(axis=0), fSC.mean(axis=0))]
 #              R_log(fRCoal.mean(axis=0), fSCoal.mean(axis=0)), R_log(fRWood.mean(axis=1),fSWood.mean(axis=1)),R_log(fRPeat.mean(axis=1),fSPeat.mean(axis=1))]
        names_columns=['HOA', 'BBOA','COA', 'SHINDOA', 'LO-OOA','MO-OOA', 'OOA']#, 'Coal', 'Wood', 'Peat']'COA', 'SHINDOA',
    l.append(rsq)
    lt.append(dr_14[0])
# names_columns=[ 'HOA', 'BBOA', 'LO-OOA', 'MO-OOA', 'OOA']
namefile='Rolling Correlation R vs. S ('
R = pd.DataFrame(l, columns=names_columns)
R['datetime'] = lt
R.to_csv(namefile +str(per)+'d.txt')
R = R.set_index('datetime')
print(per,'\n',R.mean()) 

# %%
# %%
print(R_log(COA_R.mean(axis=0), COA_S.mean(axis=0)))
print(R_log(SHINDOA_R.mean(axis=0), SHINDOA_S.mean(axis=0)))
# print(R_log(LOA_R.mean(axis=0), LOA_S.mean(axis=0)))
print(R_log(HOA_R.mean(axis=0), HOA_S.mean(axis=0)))
print(R_log(BBOA_R.mean(axis=0), BBOA_S.mean(axis=0)))
print(R_log(LO_R.mean(axis=0), LO_S.mean(axis=0)))
print(R_log(MO_R.mean(axis=0), MO_S.mean(axis=0)))
print(R_log(OOA_R.mean(axis=0), OOA_S.mean(axis=0)))
# print(R_log(Coal_R.mean(axis=0), Coal_S.mean(axis=0)))
# print(R_log(Wood_R.mean(axis=0), Wood_S.mean(axis=0)))
# print(R_log(Peat_R.mean(axis=0), Peat_S.mean(axis=0)))
# %%
ar=(R.groupby(by=[R.index.year, R.index.month]).mean())
ar.plot(figsize=(20,7), marker='o', grid=True, title=cityname)
plt.ylabel("R$^2$ log Rvs.S (14D)")


# %%
fig_R, axes = plt.subplots(nrows=len(R.columns), ncols=1, sharex=True, figsize=(25, 20), constrained_layout=True)
fig_R.suptitle( cityname+" (Absolute)\n Correlation time-dependent profiles ("+str(per)+'d)', fontsize=28)
plt.rcParams.update({'font.size': 22})
count = 0
for c in range(len(R.columns)):
    name1 = R.columns[c]
    axes[c].plot(R.index, R[name1], marker='o', color='black')
    axes[c].set_ylabel('R²')
    axes[c].grid(axis='x')
    axes[c].grid(axis='y')
    axes[c].set_axisbelow(True)
    axes[c].set_title(name1)
    count = count+1
fig_R.savefig('Rolling_R2slope_Rolling_vs_Seas_'+str(per)+'_f.png')
# %%
def QQexp_Seasonal(df, m, p_list, date_in, date_out):
    q=[]
    for i in range(0, len(p_list)):
        mask_i = (f1['Time'] > date_in[i]) & (f1['Time'] <= date_out[i])
        df_masked = df.loc[mask_i]
        ndays=len(pd.date_range(date_in[i], end=date_out[i]))
        ns=2.0*24.0*ndays
        qq=df_masked['Q_Res_S'].mean()/(m*ns-p_list[i]*(m+ns))
        q.append(qq)
    return sum(q)

# %% Q/Qexp determination and weighting by 
# city='BUC'
Q_14 = Averager_Roll_E(f1, 14, 'Q Res', city, 'avg') #averages Q every 14 days (a rolling window)
# Q_90 = Averager_Roll_E(f1, 90, 'Q Res', city, 'avg')# Averages Q every 90 days (aprox a season)
# Q_14.to_csv("Q_14_res.txt", sep="\t")
# Q_90.to_csv("Q_90_res.txt", sep="\t")
# Q_14=pd.read_csv('Q_14_res.txt', sep='\t')
# Q_90=pd.read_csv('Q_90_res.txt', sep='\t')

if city=='BCN':
    m=92.0
    p_BCN=[4.0,5.0,4.0,4.0,4.0]
    in_BCN=[pd.to_datetime("2017/09/01 00:00"),pd.to_datetime("2017/11/01 00:00"),pd.to_datetime("2018/04/01 00:00"),pd.to_datetime("2018/06/01 00:00"),pd.to_datetime("2018/09/01 00:00") ]
    out_BCN=[pd.to_datetime("2017/10/31 23:59"),pd.to_datetime("2018/03/31 23:59"),pd.to_datetime("2018/05/31 23:59"),pd.to_datetime("2018/08/31 23:59"),pd.to_datetime("2018/10/31 23:59") ]
    QQ_S = QQexp_Seasonal(f1, m, p_BCN, in_BCN, out_BCN)
if city=='BUC':
    m=92.0
    p=4.0
    p_BUC=[4.0,4.0,4.0,4.0]
    f1['Q_R']=f1['Q_Res_R']
    # f1['Q_Res_R']=f1['Q_R']/m
    f1['Q_S']=f1['Q_Res_S']
    # f1['Q_Res_S']=f1['Q_S']/m
    in_BUC=[pd.to_datetime("2016/09/01 00:00"),pd.to_datetime("2016/12/01 00:00"),pd.to_datetime("2017/03/01 00:00"),pd.to_datetime("2017/06/01 00:00")]
    out_BUC=[pd.to_datetime("2017/11/30 23:59"),pd.to_datetime("2018/02/28 23:59"),pd.to_datetime("2018/05/31 23:59"),pd.to_datetime("2018/08/31 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_BUC, in_BUC, out_BUC)
if city=='CAO':
    m=73.0
    f1['Q_S']=f1['Q Res_R']
    f1['Q_S']=f1['Q Res_R']
    p_CAO=[4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0]
    in_CAO=[pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00"),pd.to_datetime("2015/03/01 00:00")]
    out_CAO=[pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59"),pd.to_datetime("2017/01/16 23:59")]
    # QQ_S = QQexp_Seasonal(f1, m, p_CAO, in_CAO, out_CAO)
    p=4.0
if city=='LIL':
    m=72.0
    p=4.0
    p_LIL=[p,p,p,p]
    in_LIL=[pd.to_datetime("2016/10/11 00:00"),pd.to_datetime("2016/10/11 00:00"),pd.to_datetime("2016/10/11 00:00"),pd.to_datetime("2016/10/11 00:00")]
    out_LIL=[pd.to_datetime("2017/08/18 23:59"),pd.to_datetime("2017/08/18 23:59"),pd.to_datetime("2017/08/18 23:59"),pd.to_datetime("2017/08/18 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_LIL, in_LIL, out_LIL)
if city=='DUB':
    m=72.0
    p=6.0
    f1['Q_R']=f1['Q_Res_R']
    f1['Q_Res_R']=f1['Q_R']/m
    f1['Q_S']=f1['Q_Res_S']
    f1['Q_Res_S']=f1['Q_S']
    p_DUB=[p,p,p,p]
    in_DUB=[pd.to_datetime("2016/09/01 00:00"),pd.to_datetime("2016/09/01 00:00"),pd.to_datetime("2016/09/01 00:00"),pd.to_datetime("2016/09/01 00:00")]
    out_DUB=[pd.to_datetime("2017/08/31 23:59"),pd.to_datetime("2017/08/31 23:59"),pd.to_datetime("2017/08/31 23:59"),pd.to_datetime("2017/08/31 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_DUB, in_DUB, out_DUB)
if city=='MAG':
    m=70.0
    p_MAG=[5.0,5.0,5.0,5.0]
    in_MAG=[pd.to_datetime("2013/08/28 00:00"),pd.to_datetime("2013/08/28 00:00"),pd.to_datetime("2013/08/28 00:00"),pd.to_datetime("2013/08/28 00:00")]
    out_MAG=[pd.to_datetime("2014/10/30 00:00"),pd.to_datetime("2014/10/30 00:00"),pd.to_datetime("2014/10/30 00:00"),pd.to_datetime("2014/10/30 00:00")]
    QQ_S = QQexp_Seasonal(f1, m, p_MAG, in_MAG, out_MAG)
if city=='MAR':
    m=185.0
    p=6.0
    print(city)
    p_MAR=[p,p,p,p,p]
    # f1['Q_Res_R']=f1['Q_Res_R']/m
    # f1['Q_Res_S']=f1['Q_Res_S']/m
    in_MAR=[pd.to_datetime("2017/01/31 00:00"),pd.to_datetime("2017/01/31 00:00"),pd.to_datetime("2017/01/31 00:00"),pd.to_datetime("2017/01/31 00:00"),pd.to_datetime("2017/01/31 00:00")]
    out_MAR=[pd.to_datetime("2018/04/13 23:59"),pd.to_datetime("2018/04/13 23:59"),pd.to_datetime("2018/04/13 23:59"),pd.to_datetime("2018/04/13 23:59"),pd.to_datetime("2018/04/13 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_MAR, in_MAR, out_MAR)
if city=='TAR':
    m=73.0
    p=4.0   
    f1['Q_R']=f1['Q_Res_R']
    f1['Q_Res_R']=f1['Q_R']*m
    f1['Q_S']=f1['Q_Res_S']
    f1['Q_Res_S']=f1['Q_S']*m
    p_TRT=[4.0,4.0,4.0,4.0]
    in_TRT=[pd.to_datetime("2016/09/01 00:00"),pd.to_datetime("2016/12/01 00:00"),pd.to_datetime("2017/03/01 00:00"),pd.to_datetime("2017/06/01 00:00")]
    out_TRT=[pd.to_datetime("2017/11/30 23:59"),pd.to_datetime("2018/02/28 23:59"),pd.to_datetime("2018/05/31 23:59"),pd.to_datetime("2018/08/31 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_TRT, in_TRT, out_TRT)
if city=='SIR':
    m=72.0
    p=4.0
    p_SIR=[4.0,4.0,4.0,4.0]
    #•FIX WHEN I GET THE REAL DATAAAAA
    in_SIR=[pd.to_datetime("2017/09/01 00:00"),pd.to_datetime("2017/11/01 00:00"),pd.to_datetime("2018/04/01 00:00"),pd.to_datetime("2018/06/01 00:00"),pd.to_datetime("2018/09/01 00:00") ]
    out_SIR=[pd.to_datetime("2017/10/31 23:59"),pd.to_datetime("2018/03/31 23:59"),pd.to_datetime("2018/05/31 23:59"),pd.to_datetime("2018/08/31 23:59"),pd.to_datetime("2018/10/31 23:59") ]
    
    QQ_S = QQexp_Seasonal(f1, m, p_SIR, in_SIR, out_SIR)
    
if city=='SYN':
    m=85.0
    p=4.0 
    f1['Q_R']=f1['Q_Res_R']
    f1['Q_Res_R']=f1['Q_R']*m
    f1['Q_S']=f1['Q_Res_S']
    f1['Q_Res_S']=f1['Q_S']*m
    p_SYN=[6.0,6.0,6.0]
    in_SYN=[pd.to_datetime("2011/02/01 00:00"),pd.to_datetime("2011/06/01 00:00"),pd.to_datetime("2011/09/01 00:00")]
    out_SYN=[pd.to_datetime("2011/05/31 23:59"),pd.to_datetime("2011/08/31 23:59"),pd.to_datetime("2018/12/31 23:59")]
    QQ_S = QQexp_Seasonal(f1, m, p_SYN, in_SYN, out_SYN)
n = len(f1)
nr = 2.0*24.0*14.0
nb_seas=5.0
Q_theoric = n*m
Q_exp = m*n-p*(m+n)
QQ_R_num = Q_14['Q Res_R'].sum()/(14.0*(m*nr- p*(m+nr)))
print('QQ_T= ', Q_theoric/Q_exp, "QQexp_R=", QQ_R_num)# #"QQexp_S=", QQ_S)
#%%
f1['Q Res_R']=f1['Q Res_R']
f1['Q Res_S']=f1['Q Res_S']
print('Q_R= ', f1['Q Res_R'].mean(),'Q_S= ', f1['Q Res_S'].mean() )

#%%
f1['Q Res_R']=f1['Q_Res_R']/m
f1['Q Res_S']=f1['Q_Res_S']/m
Qexp=m*n-p*(m+n)
print('Rolling:', f1['Q Res_R'].sum()/Qexp)
print('Seasonal:', f1['Q Res_S'].sum()/Qexp)


# %%# Transition period - Scaled Residuals
# f1=f0
#Transition Period!
df=pd.DataFrame()
if city=='BCN':
    date_ini=[pd.to_datetime('22/10/2017',dayfirst=True), pd.to_datetime('22/03/2018',dayfirst=True),pd.to_datetime('22/05/2018',dayfirst=True),pd.to_datetime('22/08/2018',dayfirst=True)]
    date_end=[pd.to_datetime('07/11/2017',dayfirst=True), pd.to_datetime('07/04/2018',dayfirst=True),pd.to_datetime('07/06/2018',dayfirst=True), pd.to_datetime('07/09/2018',dayfirst=True)]
if city=='BUC':
    date_ini=[pd.to_datetime('22/11/2016',dayfirst=True), pd.to_datetime('22/02/2017',dayfirst=True),pd.to_datetime('22/05/2017',dayfirst=True),pd.to_datetime('22/08/2017',dayfirst=True)]
    date_end=[pd.to_datetime('07/12/2016',dayfirst=True), pd.to_datetime('07/03/2017',dayfirst=True),pd.to_datetime('07/06/2017',dayfirst=True), pd.to_datetime('07/09/2017',dayfirst=True)]
if city=='CAO':
    date_ini=[pd.to_datetime('22/05/2015',dayfirst=True), pd.to_datetime('22/08/2015',dayfirst=True),pd.to_datetime('22/11/2015',dayfirst=True), pd.to_datetime('22/02/2016',dayfirst=True),pd.to_datetime('22/05/2016',dayfirst=True),pd.to_datetime('22/08/2016',dayfirst=True),pd.to_datetime('22/11/2016',dayfirst=True)]
    date_end=[pd.to_datetime('07/06/2015',dayfirst=True), pd.to_datetime('07/09/2015',dayfirst=True),pd.to_datetime('07/12/2015',dayfirst=True), pd.to_datetime('07/03/2016',dayfirst=True),pd.to_datetime('07/06/2016',dayfirst=True),pd.to_datetime('07/09/2016',dayfirst=True),pd.to_datetime('07/12/2016',dayfirst=True)]
if city=='DUB':
    date_ini=[pd.to_datetime('22/11/2016',dayfirst=True), pd.to_datetime('22/02/2017',dayfirst=True),pd.to_datetime('22/05/2017',dayfirst=True)]
    date_end=[pd.to_datetime('07/12/2016',dayfirst=True), pd.to_datetime('07/03/2017',dayfirst=True),pd.to_datetime('07/06/2017',dayfirst=True) ]
if city=='LIL':
    date_ini=[pd.to_datetime('22/11/2016',dayfirst=True), pd.to_datetime('22/02/2017',dayfirst=True),pd.to_datetime('22/05/2017',dayfirst=True)]
    date_end=[pd.to_datetime('07/12/2016',dayfirst=True), pd.to_datetime('07/03/2017',dayfirst=True),pd.to_datetime('07/06/2017',dayfirst=True)]
if city=='TAR':
    date_ini=[pd.to_datetime('22/11/2016',dayfirst=True), pd.to_datetime('22/02/2017',dayfirst=True),pd.to_datetime('22/05/2017',dayfirst=True),pd.to_datetime('22/08/2017',dayfirst=True)]
    date_end=[pd.to_datetime('07/12/2016',dayfirst=True), pd.to_datetime('07/03/2017',dayfirst=True),pd.to_datetime('07/06/2017',dayfirst=True), pd.to_datetime('07/09/2017',dayfirst=True)]
if city=='MAG':
    date_ini=[pd.to_datetime('22/08/2013',dayfirst=True),pd.to_datetime('22/11/2013',dayfirst=True), pd.to_datetime('22/02/2014',dayfirst=True),pd.to_datetime('22/05/2014',dayfirst=True),pd.to_datetime('22/08/2014',dayfirst=True)]
    date_end=[pd.to_datetime('07/09/2013',dayfirst=True),pd.to_datetime('07/12/2013',dayfirst=True), pd.to_datetime('07/03/2014',dayfirst=True),pd.to_datetime('07/06/2014',dayfirst=True), pd.to_datetime('07/09/2014',dayfirst=True)]
if city=='MAR':
    date_ini=[pd.to_datetime('22/02/2017',dayfirst=True),pd.to_datetime('22/05/2017',dayfirst=True),pd.to_datetime('22/08/2017',dayfirst=True), pd.to_datetime('22/11/2017',dayfirst=True),pd.to_datetime('22/02/2018',dayfirst=True)]
    date_end=[pd.to_datetime('07/03/2017',dayfirst=True),pd.to_datetime('07/06/2017',dayfirst=True), pd.to_datetime('07/09/2017',dayfirst=True),pd.to_datetime('07/12/2017',dayfirst=True),pd.to_datetime('07/03/2018',dayfirst=True)]
if city=='SIR':
    date_ini=[pd.to_datetime('22/02/2016',dayfirst=True),pd.to_datetime('22/05/2016',dayfirst=True),pd.to_datetime('22/08/2016',dayfirst=True), pd.to_datetime('22/11/2016',dayfirst=True),pd.to_datetime('22/02/2017',dayfirst=True)]
    date_end=[pd.to_datetime('07/03/2016',dayfirst=True),pd.to_datetime('07/06/2016',dayfirst=True), pd.to_datetime('07/09/2016',dayfirst=True),pd.to_datetime('07/12/2016',dayfirst=True),pd.to_datetime('07/03/2017',dayfirst=True)]    
if city=='SYN':
    date_ini=[pd.to_datetime('22/05/2011',dayfirst=True),pd.to_datetime('22/08/2011',dayfirst=True)]
    date_end=[pd.to_datetime('07/06/2011',dayfirst=True),pd.to_datetime('07/09/2011',dayfirst=True)]  
mask_i=[]
for i in range(0,len(date_ini)):
    mask_ii = (f1['Time'] >= date_ini[i]) & (f1['Time'] <= date_end[i])
    print(date_ini[i], date_end[i])
    mask_i.append(mask_ii)
mask=mask_i[0]|mask_i[1]#|mask_i[2]|mask_i[3]#|mask_i[4]#|mask_i[5]|mask_i[6]
f1_masked = f1.loc[mask]
f1_masked.to_csv('f1_transition.txt',sep='\t')

boxprops = dict(linestyle='-', linewidth=0.6)
whiskerprops = dict(linestyle='-', linewidth=0.6)
medianprops = dict(linestyle='-', linewidth=0.6,color='black')
meanprops = dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
fig,ax=plt.subplots(figsize=(6,6))
f1_masked[['Sc Res_R','Sc Res_S']].boxplot(showfliers=False,showmeans=True,ax=ax,meanprops=meanprops,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops)
ax.set_title(cityname + '\n'+ 'TRANSITION PERIODS')
ax.set_ylabel('Scaled residuals $(μg·m^{-3})$')
ax.set_xticklabels(['Rolling', 'Seasonal'])
#%%
fig, axs=plt.subplots(figsize=(5,5))
f1_masked[['Sc Res_R','Sc Res_S']].plot.hist(bins=50, color=['red', 'blue'], alpha=0.5, grid=True, ax=axs)
axs.legend(['Rolling', 'Seasonal'], fontsize=13, loc='upper right')
axs.set_xlabel('Scaled Residuals', fontsize=15)
axs.set_ylabel('Frequency', fontsize=15)
axs.set_title(cityname+'\n'+'Transition period', fontsize=17)
#%%
f1_masked['unc_R']=f1_masked['Res_R']/f1_masked['Sc Res_R']
f1_masked['unc_S']=f1_masked['Res_S']/f1_masked['Sc Res_S']
f1_masked[['Res_R','Sc Res_R','Res_S','Sc Res_S']].boxplot(showfliers=False,grid=True,showmeans=True)
# f1_masked[['unc_R','unc_S']].boxplot(showfliers=False)
#%%Correlations transition period
f1['Sc Res_R']= f1
#%%

if city!='SYN':
# f1_masked['NOx']=f1_masked['Nox']
    print(R2(f1_masked['HOA_Rolling'], f1_masked['BCff']).round(2), R2(f1_masked['HOA_seas'], f1_masked['BCff']).round(2))
    print(R2(f1_masked['HOA_Rolling'], f1_masked['NOx']).round(2), R2(f1_masked['HOA_seas'], f1_masked['NOx']).round(2))
    print(R2(f1_masked['BBOA_Rolling'], f1_masked['BCwb']).round(2),R2(f1_masked['BBOA_seas'], f1_masked['BCwb']).round(2))
    print(R2(f1_masked['LO-OOA_Rolling'], f1_masked['NO3']).round(2),R2(f1_masked['LO-OOA_seas'], f1_masked['NO3']).round(2))
    print(R2(f1_masked['MO-OOA_Rolling'], f1_masked['SO4']).round(2),R2(f1_masked['MO-OOA_seas'], f1_masked['SO4']).round(2))
    print(R2(f1_masked['OOA_Rolling'], f1_masked['NH4']).round(2),R2(f1_masked['OOA_seas'], f1_masked['NH4']).round(2))
if city=='SYN':
    print(R2(f1_masked['HOA_Rolling'], f1_masked['BC']).round(2), R2(f1_masked['HOA_seas'], f1_masked['BC']).round(2))
    print(R2(f1_masked['HOA_Rolling'], f1_masked['NOx']).round(2), R2(f1_masked['HOA_seas'], f1_masked['NOx']).round(2))
    print(R2(f1_masked['MO-OOA_Rolling'], f1_masked['SO4']).round(2),R2(f1_masked['MO-OOA_seas'], f1_masked['SO4']).round(2))
    print(R2(f1_masked['OOA_Rolling'], f1_masked['NH4']).round(2),R2(f1_masked['OOA_seas'], f1_masked['NH4']).round(2))










# %% In the case of BCN, these ions have to be casted as numeric first so they become numbers
f1['f60_R'] = pd.to_numeric(f1['f60_R'], errors='coerce')
f1['f73_R'] = pd.to_numeric(f1['f73_R'], errors='coerce')
f1['f43_R'] = pd.to_numeric(f1['f43_R'], errors='coerce')
f1['f44_R'] = pd.to_numeric(f1['f44_R'], errors='coerce')
f1['f60_S'] = pd.to_numeric(f1['f60_S'], errors='coerce')
f1['f73_S'] = pd.to_numeric(f1['f73_S'], errors='coerce')
f1['f43_S'] = pd.to_numeric(f1['f43_S'], errors='coerce')
f1['f44_S'] = pd.to_numeric(f1['f44_S'], errors='coerce')
f1['f43'] = pd.to_numeric(f1['f43'], errors='coerce')
f1['f44'] = pd.to_numeric(f1['f44'], errors='coerce')
f1['f60'] = pd.to_numeric(f1['f60'], errors='coerce')
f1['f73'] = pd.to_numeric(f1['f73'], errors='coerce')
# %%
f1['mz60_R'] = pd.to_numeric(f1['mz60_R'], errors='coerce')
f1['mz73_R'] = pd.to_numeric(f1['mz73_R'], errors='coerce')
f1['mz43_R'] = pd.to_numeric(f1['mz43_R'], errors='coerce')
f1['mz44_R'] = pd.to_numeric(f1['mz44_R'], errors='coerce')
f1['mz60_S'] = pd.to_numeric(f1['mz60_S'], errors='coerce')
f1['mz73_S'] = pd.to_numeric(f1['mz73_S'], errors='coerce')
f1['mz43_S'] = pd.to_numeric(f1['mz43_S'], errors='coerce')
f1['mz44_S'] = pd.to_numeric(f1['mz44_S'], errors='coerce')
f1['mz43'] = pd.to_numeric(f1['mz43'], errors='coerce')
f1['mz44'] = pd.to_numeric(f1['mz44'], errors='coerce')
f1['mz60'] = pd.to_numeric(f1['mz60'], errors='coerce')
f1['mz73'] = pd.to_numeric(f1['mz73'], errors='coerce')

# %%
f1['mz60_R'] = f1['f60_R']*f1['OA_app_R']
f1['mz73_R'] = f1['f73_R']*f1['OA_app_R']
f1['mz43_R'] = f1['f43_R']*f1['OA_app_R']
f1['mz44_R'] = f1['f44_R']*f1['OA_app_R']
f1['mz60_S'] = f1['f60_S']*f1['OA_app_s']
f1['mz73_S'] = f1['f73_S']*f1['OA_app_s']
f1['mz43_S'] = f1['f43_S']*f1['OA_app_s']
f1['mz44_S'] = f1['f44_S']*f1['OA_app_s']
f1['mz43'] = f1['f43_S']*f1['Org']
f1['mz44'] = f1['f44_S']*f1['Org']
f1['mz60'] = f1['f60_S']*f1['Org']
f1['mz73'] = f1['f73_S']*f1['Org']
# %%
# *************** Weekly averages of original an modeled ions*************
#
# Averager_Roll generates two df (R_w, S_w) to flot ions histogram afterwards


def Averager_Roll(df, name_R, name_S):
    lR = []
    lS = []
    for i in range(0, len(dr_all)):
        st_d = dr_all[i]
        # You can change the length of rolling R2 window here (periods=14)
        dr_14 = pd.date_range(st_d, periods=14)
        en_d = dr_14[-1]
        mask_i = (f1['Time'] > st_d) & (f1['Time'] <= en_d)
        f3 = df.loc[mask_i]
        avg_R = [f3['mz43_R'].mean(skipna=True), f3['mz44_R'].mean(skipna=True), f3['mz60_R'].mean(skipna=True),
                 f3['mz73_R'].mean(skipna=True), f3['mz43'].mean(skipna=True), f3['mz44'].mean(skipna=True),
                 f3['mz60'].mean(skipna=True), f3['mz73'].mean(skipna=True)]
        avg_S = [f3['mz43_S'].mean(skipna=True), f3['mz44_S'].mean(skipna=True), f3['mz60_S'].mean(skipna=True),
                 f3['mz73_S'].mean(skipna=True), f3['mz43'].mean(skipna=True), f3['mz44'].mean(skipna=True),
                 f3['mz60'].mean(skipna=True), f3['mz73'].mean(skipna=True)]
        lR.append(avg_R)
        lS.append(avg_S)
    R_week = pd.DataFrame(lR, columns=['mz43_R', 'mz44_R', 'mz60_R', 'mz73_R','mz43','mz44','mz60','mz73'])  
    S_week = pd.DataFrame(lS, columns=['mz43_S', 'mz44_S', 'mz60_S', 'mz73_S','mz43','mz44','mz60','mz73'])  
    R_week['datetime'] = dr_all
    S_week['datetime'] = dr_all
    R_week = R_week.set_index('datetime')
    S_week = S_week.set_index('datetime')
    R_week.to_csv(name_R+'_Ions.csv')
    S_week.to_csv(name_S+'_Ions.csv')
    return R_week, S_week


# %%
# We call the function and save R_week and S_week under the R_w, S_w names.
R_w, S_w = Averager_Roll(f1, "Roll", "Seas")
# %%
R_w['f44f43']=R_w['mz44']/R_w['mz43']
R_w['f44f43_R']=R_w['mz44_R']/R_w['mz43_R']
# R_w['f44f43_R']=R_w['f44f43_R'].loc(abs(R_w['f44f43_R'])<20.0)
S_w['f44f43_S']=S_w['mz44_S']/S_w['mz43_S']

# %%
R_w['f44f43_R'].plot()
S_w['f44f43_S'].plot()
R_w['f44f43'].plot(figsize=(20,10))

plt.legend()
# %% 
ion='f44f43'
li = []


for i in range(0, len(R_w)-14, 1):
    calc = [R_w[ion].iloc[i+14]-R_w[ion].iloc[i], R_w[ion+'_R'].iloc[i+14]-R_w[ion+'_R'].iloc[i],
            S_w[ion+'_S'].iloc[i+14]-S_w[ion+'_S'].iloc[i]]  # R_w['f44'].iloc[i+7]-R_w['f44'].iloc[i],
    li.append(calc)
RS_w_shift = pd.DataFrame(li, columns=[ion, ion+'_Rolling', ion+'_Seasonal'])
RS_w_shift[ion]=RS_w_shift[ion].loc[abs(RS_w_shift[ion])<20.0]
RS_w_shift[ion+'_Rolling']=RS_w_shift[ion+'_Rolling'].loc[abs(RS_w_shift[ion+'_Rolling'])<20.0]
RS_w_shift[ion+'_Seasonal']=RS_w_shift[ion+'_Seasonal'].loc[abs(RS_w_shift[ion+'_Seasonal'])<20.0]
RS_w_shift['Rolling'] = RS_w_shift[ion]-RS_w_shift[ion+'_Rolling']
RS_w_shift['Seasonal'] = RS_w_shift[ion]-RS_w_shift[ion+'_Seasonal']
RS_w_shift['r_s']=RS_w_shift['Rolling']-RS_w_shift['Seasonal']
# %%
RS_w_shift['Rolling'].plot(figsize=(20,10),legend=True,title=cityname)
RS_w_shift['Seasonal'].plot(legend=True)
plt.grid()
# %%HISTOGRAM F44[I]-F44[I-1]
fig_1=plt.figure(figsize=(10,10))
RS_w_shift['Rolling'].plot.kde(alpha=0.5, linewidth=5).get_figure()
RS_w_shift['Seasonal'].plot.kde(alpha=0.5,linewidth=5).get_figure()
# plt.xlabel("$\Delta f_{44}  - \Delta f_{434)_{Rolling/Seasonal} $ ")
plt.ylabel("Frequency (14 days increment)")
plt.legend()
plt.grid(True)
plt.suptitle(cityname)
plt.title('14-day adaptability of '+ion)
fig_1.savefig('Adaptability_1_f44f43.png')
# %%
RS_w_shift['r_s'].plot.kde(alpha=0.5).get_figure()
plt.grid()
# %%
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/B. SoFi/B. SoFi/reference profiles/Comp_R_S')
ref = pd.read_csv("Reference_Profiles.txt", sep="\t", infer_datetime_format=True)
# %%
def R_log(a,b):
    c=pd.DataFrame({"a":-1/np.log(a), "b":-1/np.log(b)})
    cm=c.corr(method='pearson')
    r=cm.iloc[0,1]
    return r**2

# %%
