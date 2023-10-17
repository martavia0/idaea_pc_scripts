# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 07:57:48 2020

@author: Marta Via
"""
# ************ DATE AVERAGER **********
#
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import math
#%%
pd.options.display.precision = 0
#
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR")
s=pd.read_csv("Specs.txt", sep="\t") 
e=pd.read_csv("Errors.txt", sep="\t")
#
s['dt']=pd.to_datetime(s['utc_time'])
s['date']=s['dt'].dt.date
e["dt"]=pd.to_datetime(e['utc_end_time'])
e['date']=e['dt'].dt.date
#%% HOURLY AVERAGE
s2=pd.DataFrame(columns=s.columns)
ii=pd.DataFrame()
ii['dt'] = pd.date_range(start='21/9/2017', end='31/10/2018',freq='H')
ii['d']= pd.to_datetime(ii['dt'])
ii['date']=ii['dt'].dt.date
df=pd.DataFrame()
#%%
#%%
for i in range(0, len(ii)): #Resolució desitjada
    entra=0
    ac=pd.DataFrame(columns=s.columns)
    if (s['date'].isin([ii.date.iloc[i]])).any():
        print(i,j)
        for j in range(0,len(s)): #Resolució per defecte
            print(i,j)
            if(s.dt.iloc[j]>ii.d.iloc[i] and s.dt.iloc[j]<=ii.d.iloc[i+1]):
                print("Hola")
                ac=ac.append(s.iloc[j], ignore_index=True)
                print(i,j,ac)
                entra=1
#            break
        row=ac.mean(axis=0)
        row_T=row.transpose()
        df=df.append(row, ignore_index=True)
#    if entra==0
                 
#%%
