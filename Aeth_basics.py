# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:32:13 2020

@author: Marta Via
"""

import pandas as pd
import os as os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/Other Data Instruments/AE33_BCN")
prov1= pd.read_csv("aeth_proves.txt", sep="\t", low_memory=False, parse_dates=True, infer_datetime_format=True)
#%%
prov2=prov1
prov2['date2']=pd.to_datetime(prov2["date"],infer_datetime_format=True)
prov2=prov2.set_index(prov2.date2)
table1=prov2[:84322]
table1.tail()
table1=table1.drop(columns=["Pressure(Pa)", "Temperature(C)", "FlowC"])
table1=table1.drop(columns=["BC11", "BC12", "BC21","BC22", "BC31", "BC32", "BC41", "BC32"])
table1=table1.drop(columns=[ "BC42", "BC51","BC52", "BC61", "BC62", "BC71", "BC72"]) #"BC41;",
table1['Weekday Name'] = table1.index.weekday_name
#%%
file_out=table1.to_csv('C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/Other Data Instruments/AE33_BCN/aeth_t.txt',sep="\t",na_rep='NaN')
#%%
#%%
sns.set(rc={'figure.figsize':(11, 4)})
table1['BC2'].plot(linewidth=0.5);
#%%
cols_plot = ['BC1', 'BC2', 'BC3','BC4', 'BC5', 'BC6','BC7']
axes = table1[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
i=0
for ax in axes:
    ax.set_ylabel(cols_plot[i])
    i=i+1
#%%
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.plot(table1.loc['01/01/2018':'28/02/2018', 'BC7'], marker='o', linestyle='-')
ax.set_ylabel('Black Carbon in WL 1')
ax.set_title('Jan-Feb 2018 BC1')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
#ax.Locator()
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));
#%%
table2=pd.DataFrame(columns=["date","BC1", "BC2", "BC3", "BC4", "BC5", "BC6", "BC7"])
date=table1.date()
#%%
matplotlib.pyplot.plot(table1.date, table1.BC1) 
#%%
plt.plot(table1.date, table1.BC1) 

#%%
table2 = pd.DataFrame(columns=["date","BC1", "BC2", "BC3", "BC4", "BC5", "BC6", "BC7"])
table2.date=table1.date
#%%
l=[table1.BC1.quantile(0.1), table1.BC2.quantile(0.1),table1.BC3.quantile(0.1),table1.BC4.quantile(0.1),
   table1.BC5.quantile(0.1),table1.BC6.quantile(0.1),table1.BC7.quantile(0.1)]   
print(table1.BC1.quantile(0.1))
print(0.05*table1.BC1.iloc[5]+l[0])
#%%
for i in range(0,len(table1)):
    table2.BC1.iloc[i] = 0.05*table1.BC1.iloc[i]+l[0]
    table2.BC2.iloc[i] = 0.05*table1.BC2.iloc[i]+l[1]
    table2.BC3.iloc[i] = 0.05*table1.BC3.iloc[i]+l[2]
    table2.BC4.iloc[i] = 0.05*table1.BC4.iloc[i]+l[3]
    table2.BC5.iloc[i] = 0.05*table1.BC5.iloc[i]+l[4]
    table2.BC6.iloc[i] = 0.05*table1.BC6.iloc[i]+l[5]
    table2.BC7.iloc[i] = 0.05*table1.BC7.iloc[i]+l[6]
#%%
file_out=table2.to_csv('C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/Other Data Instruments/AE33_BCN/BC17.txt',sep="\t",na_rep='NaN')    





