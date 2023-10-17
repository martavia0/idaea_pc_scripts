# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:53:41 2019

@author: Marta Via
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as d
import datetime as dt
import numpy as np
#%%
os.chdir('C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/Other Data Instruments/SMPS')
f1= pd.read_csv("SMPS_5min_2017_MVIA_.txt", sep=";", low_memory=False, skip_blank_lines=False)
f2= pd.read_csv("SMPS_5min_2018_MVIA__.txt", sep=";", low_memory=False, skip_blank_lines=False)
#%%
#%%
f1=f1.drop(['Sample Temp (C)','Start Time', 'Sample #','Sample Pressure (kPa)', 'Mean Free Path (m)' ,
             'Gas Viscosity (Pa*s)', 'Diameter Midpoint','Unnamed: 2','Scan Up Time(s)', 'Retrace Time(s)', 
             'Down Scan First','Scans Per Sample','Impactor Type(cm)', 'Sheath Flow(lpm)','Aerosol Flow(lpm)',
             'CPC Inlet Flow(lpm)','CPC Sample Flow(lpm)', 'Low Voltage', 'High Voltage', "Lower Size(nm)", 
             'Upper Size(nm)', 'Title', 'Status Flag', 'td(s)', 'tf(s)', 'D50(nm)', 'Median(nm)',
             'Mean(nm)','Geo. Mean(nm)','Mode(nm)','Geo. Std. Dev.', 'Total Conc.(#/cm3)','Comment','Month'], axis=1)
f2=f2.drop(['Sample Temp (C)','Start Time', 'Sample #','Sample Pressure (kPa)', 'Mean Free Path (m)' ,'Unnamed: 2',
             'Gas Viscosity (Pa*s)', 'Diameter Midpoint','Scan Up Time(s)', 'Retrace Time(s)', 'Down Scan First','Scans Per Sample',
             'Impactor Type(cm)', 'Sheath Flow(lpm)', 'Aerosol Flow(lpm)', 'CPC Inlet Flow(lpm)', 
             'CPC Sample Flow(lpm)', 'Low Voltage', 'High Voltage', "Lower Size(nm)", 'Upper Size(nm)',
             'Title', 'Status Flag', 'td(s)', 'tf(s)', 'D50(nm)', 'Median(nm)', 'Mean(nm)',
             'Geo. Mean(nm)','Mode(nm)','Geo. Std. Dev.', 'Total Conc.(#/cm3)','Comment','Month'], axis=1) 
             #%%
d=pd.concat([f1['Density(g/cc)'], f2['Density(g/cc)']])
f=pd.concat([f1, f2])
f=f.drop(['Density(g/cc)'], axis=1)

#%%
f=f.drop(['10.9','11.3','11.8','12.2','12.6', '13.1','13.60','14.1','14.6','15.1','15.7','16.3','16.8',
          '101.8','105.5','109.4','113.4','117.6','121.9','126.3','131','135.8','140.7','145.9','151.2',
          '156.8','162.5','168.5','174.7','181.1','187.7','194.6','201.7','209.1','216.7','224.7','232.9',
          '241.4','250.3','259.5','269','278.8','289','299.6','310.6','322','333.8','346','358.7','371.8',
          '385.4','399.5','414.2','429.4','445.1','461.4','478.3'], axis=1)
#%%
diam=list(f.columns)
del diam[0]
print type(diam)
l = [[x] for x in diam]
#%%
#%%
for j in range(0,len(diam)):
    for i in range(0,len(f)):
        if f.iloc[i][j+1] != np.NaN:
            a=f.iloc[i][j+1]*((float(diam[j])**3)*np.pi/(6*64))*(d.iloc[i]/(10**9))
            l[j].append(a)
        print "i= ",i,"j= ", j, "a= ", a
#%%
ff=pd.DataFrame(l)        
ff2=ff.transpose()
#%%
ff2.columns=ff2.iloc[0]
ff2=ff2.drop(0)
#%%
ff2['Timestamp']=list(f.date) 
#%%
ff2['Conc']=ff2['17.5']+ff2['18.1']+ff2['18.8']+ff2['19.5']+ff2['17.5']+ff2['18.1']+ff2['18.8']+ff2['19.5']+ff2['20.2']+ff2['20.9']+ff2['21.7']+ff2['22.5']+ff2['23.3']+ff2['24.1']+ff2['25']+ff2['26.9']+ff2['27.9']+ff2['28.9']+ff2['30']+ff2['31.1']+ff2['32.2']+ff2['33.4']+ff2['34.6']+ff2['35.9']+ff2['37.2']+ff2['38.5']+ff2['40']+ff2['41.4']+ff2['42.9']+ff2['44.5']+ff2['46.1']+ff2['47.8']+ff2['49.6']+ff2['51.4']+ff2['53.3']+ff2['55.2']+ff2['57.3']+ff2['59.4']+ff2['61.5']+ff2['63.8']+ff2['66.1']+ff2['68.5']+ff2[ '71']+ff2['73.7']+ff2['76.4']+ff2['79.1']+ff2['82']+ff2['85.1']+ff2['88.2']+ff2['91.4']+ff2['94.7']+ff2['98.2']

#%%
ff2['Conc_ug']=ff2['Conc']/1000000
#%%
ff2=ff2.drop(['17.5','18.1','18.8','19.5','17.5','18.1','18.8','19.5','20.2','20.9','21.7','22.5','23.3','24.1','25','25.9','26.9','27.9','28.9','30','31.1','32.2','33.4','34.6','35.9','37.2','38.5','40','41.4','42.9','44.5','46.1','47.8','49.6','51.4','53.3','55.2','57.3','59.4','61.5','63.8','66.1','68.5','71','73.7','76.4','79.1','82','85.1','88.2','91.4','94.7','98.2'], axis=1)
#%%
df=ff2.to_csv('SMPS.csv', sep=";")
