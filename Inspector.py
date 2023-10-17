# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:36:30 2020

@author: Marta Via
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import numpy as np
import scipy
import glob
import math
#%%
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Org_BC_NR_Chem/Data_Treatment/Org_BCs_p6")
ext =  pd.read_csv("Externals.txt", sep="\t",skip_blank_lines=False, keep_default_na=True, parse_dates=True,infer_datetime_format=False)
#%%
path = r"C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Org_BC_NR_Chem/Data_Treatment/Org_BCs_p6" # use your path
filenames = glob.glob(path + "/TimeSeries_*.txt")
df_TS = []
for filename in filenames:
    df_TS.append(pd.read_csv(filename, skip_blank_lines=False, parse_dates=True, infer_datetime_format=True, sep="\t"))
#    
filenames = glob.glob(path + "/Profiles*.txt")
df_PRO = []
for filename in filenames:
    df_PRO.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))
#    
filenames = glob.glob(path + "/Residuals_TS*.txt")
df_RESTS = []
for filename in filenames:
    df_RESTS.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))
#
filenames = glob.glob(path + "/Residuals_PR*.txt")
df_RESPR = []
for filename in filenames:
    df_RESPR.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))

filenames = glob.glob(path + "/Datetime_*.txt")
df_t = []
for filename in filenames:
    df_t.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))    
#%%
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Org_BC_NR_Chem/Data_Treatment/Org_BCs_p6")    
mz=pd.read_csv("mz.txt")
ts=pd.read_csv("time.txt", skip_blank_lines=False, sep="\t", infer_datetime_format=True)
ts['Time']=pd.to_datetime(ts['timestamp'], dayfirst=True, errors='coerce')
#%%#%%    
#********* ARRANGEMENT  TS-DATE  *******
# Here we append to the TS dataframes their timestamp.    
for k in range(0,len(df_TS)):
    a=pd.to_datetime(df_t[k]['acsm_utc_end_time'], dayfirst=True)
    print(k)
    df_TS[k]=pd.concat([df_TS[k],a], axis=1)
#    df_RESTS[i]=pd.concat([df_RESTS[i],df_t[i]], axis=1)
#%% 
#  ARRANGEMENT TO SAME TIMESTAMPS
#df_TS4=llista de dataframes.     
df_TS4=range(0,len(df_TS))  #df_TS: serie temporal a la que li falten punts. 
for k in range(78,len(df_TS)):
#    df_TS4[k]=pd.DataFrame()
    print(k, len(df_TS))
    list_k=[None] * len(ts)
#    list_k=[] # per cada punts del temps que volem
    for i in range(0,len(ts)): #TS: resolució bona
        entra=0
        for j in range(0,len(df_TS[k])): # j recorre els temps incomplets
            if ((df_TS[k]['acsm_utc_end_time'].iloc[j]== ts.Time.iloc[i])):
                print(df_TS[k]['acsm_utc_end_time'].iloc[j],ts.Time.iloc[i])
                list_k[i]=(df_TS[k].iloc[j])
                entra=1
                break
        if entra==0:
            list_k[i]=(pd.Series([np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
#        print(list_k)
#        sys.exit()
    a=pd.DataFrame(list_k, columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'acsm_utc_end_time', 'd']) 
           
#    df_TS4[k]=a
#    print(list_k)    
#%%

#%%    
df34223=pd.DataFrame(list_k[:1326])
#%% 
#  ARRANGEMENT TO SAME TIMESTAMPS 2
df_TS4=range(0,len(df_TS))  
for k in range(78,len(df_TS)):
    df_TS4[k]=pd.DataFrame()
    list_k= [None] * (21+len(df_TS[k]))
    for i in range(0,len(ts)):
#        entra=0
        ac_F1=0.0
        ac_F2=0.0
        ac_F3=0.0
        ac_F4=0.0
        ac_F5=0.0
        ac_F6=0.0
        count=0.0000001
        for j in range(0,len(df_TS[k])):
            if ((df_TS[k]['acsm_utc_end_time'].iloc[j]>ts.Time.iloc[i-1]) and (df_TS[k]['acsm_utc_end_time'].iloc[j]<=ts.Time.iloc[i])):
#                print(df_TS[k]['acsm_utc_end_time'].iloc[j], ts.Time.iloc[i])
                ac_F1=ac_F1+df_TS[k]['F1'].iloc[j]
                ac_F2=ac_F2+df_TS[k]['F2'].iloc[j]
                ac_F3=ac_F3+df_TS[k]['F3'].iloc[j]
                ac_F4=ac_F4+df_TS[k]['F4'].iloc[j]
                ac_F5=ac_F5+df_TS[k]['F5'].iloc[j]
                ac_F6=ac_F6+df_TS[k]['F6'].iloc[j]
                count=count+1.0
                entra=1
        row=[ac_F1/count,ac_F2/count, ac_F3/count, ac_F4/count,ac_F5/count,ac_F6/count]
#        print(row)
        list_k[i]=row
        print(i)
#        if entra==0:
#            list_k.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])#pd.Series([np.nan, np.nan,np.nan, np.nan,np.nan, np.nan,np.nan]))
#%%
    a=pd.DataFrame(list_k, columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'])#, 'acsm_utc_end_time']) 
    print(k)           
    df_TS4[k]=a
#    print(list_k)
#%%
for k in range(0,len(df_TS4)):
    name="df_TS4_2_" + str(k) + ".txt"
    df_TS4[k].to_csv(name, sep="\t")    
#%%
for i in range(0,len(df_PRO)):
    df_PRO[i]=df_PRO[i].drop(0)
    df_PRO[i]=df_PRO[i].drop(len(df_PRO[0]))
#%% 
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Org_BC_NR_Chem/Data_Treatment/Org_BCs_p5/Plots")

for i in range(0,len(df_PRO)):       
    Profiles_O(df_PRO[i])
#    Profiles_F(df_PRF[i])
    Time_Series(df_TS[i])
    Pie(df_TS[i])
    Hourly(df_TS[i])
    Weekly(df_TS[i])
    Monthly(df_TS[i])

#%%********** PROFILES  ORGANICS *******
def Profiles_O(df_O):
    
    df_O['mz']=mz
    fig_1=plt.figure()
    pro=df_O.plot.bar(subplots=True, figsize=(10,10), x="mz")
    plotname_PRO = "Run_" + str(i)+ "_PR_O.png"
    fig_1 = plt.gcf()
    plt.savefig(plotname_PRO)

#********** PROFILES  FILTERS *******
    """
def Profiles_F(df_F):
    fig_2=plt.figure()
    prf=df_F.plot.bar(subplots=True, figsize=(10,10), x="F")
    plotname_PRF = "Run_" + str(i)+ "_PR_F.png"
    plt.savefig(plotname_PRF)
    plt.show()
    """
# ******* TIME SERIES *********    
def Time_Series(df_TS):
    df_TS['d']=pd.to_datetime(df_TS["acsm_utc_end_time"], dayfirst=True)
    df_TS=df_TS.set_index(df_TS.d)
    del df_TS['d']
    fig_ts=plt.figure()
    ts_p=df_TS.plot(subplots=True, figsize=(10,10))
    plotname_TS = "Run_" + str(i)+ "_TS.png"
    plt.savefig(plotname_TS)
    plt.show()
# ************** PIE PLOT ************
def Pie(df_TS):
    avg=df_TS.mean()
    fig_pie=plt.figure()
    avg.plot.pie(figsize=(5,5), autopct='%1.1f%%',startangle=90)
    plotname_pie="Run_"+str(i)+"_Pie.png"
    fig_pie.savefig(plotname_pie)
# ****** HOURLY ******
def Hourly(df_TS):   
    df_TS['d']=pd.to_datetime(df_TS["acsm_utc_end_time"], dayfirst=True)
    df_TS['Hour']=df_TS['d'].dt.hour
    Hour=df_TS.groupby(df_TS['Hour']).mean()
    figh=plt.figure()
    Hour.plot(subplots=True, figsize=(6,10))
    plotname_hour = "Run_"+str(i)+"_Diel.png"
    plt.savefig(plotname_hour)
    del df_TS['Hour']
# ****** WEEKLY ******
def Weekly(df_TS):    
    df_TS['Week']=df_TS['d'].dt.dayofweek
    Week=df_TS.groupby(df_TS['Week']).mean()
    figw=plt.figure()
    Week.plot(subplots=True, figsize=(6,10))
    plotname_week = "Run_"+str(i)+"_Weekly.png"
    plt.savefig("plotname_week")
    del df_TS['Week']
# ******** MONTHLY ********
def Monthly(df_TS):
    df_TS['Month']=df_TS['d'].dt.month
    Month=df_TS.groupby(df_TS['Month']).mean()
    figm=plt.figure()
    Month.plot(subplots=True, figsize=(6,10))
    plotname_month="Run_"+str(i)+"_Monthly.png"
    plt.savefig("plotname_month")
    del df_TS['Month']

#%%
#Step1: Definition and sorting of OOAs:
for k in range(0,len(df_PRO)):
    df_PRO[k]['mz']=mz['mz']
    df_PRO[k]=df_PRO[i].set_index("mz")
    if ((df_PRO[k]['F5'].iloc[43]/df_PRO[k]['F5'].loc[44])>(df_PRO[i]['F6'].loc[43]/df_PRO[i]['F6'].loc[44])):
        df_PRO[i]=df_PRO[i].rename(columns={"F5":"LO-OOA","F6":"MO-OOA"})
    else:
        df_PRO[i]=df_PRO[i].rename(columns={"F5":"MO-OOA","F6":"LO-OOA"})   
#%%
#Treating externals
extern=ext        
extern['d']=pd.to_datetime(extern.date, dayfirst=True, errors="coerce")
extern['NOx']=ext.NO+ext.No2
extern['OX']=ext.No2+ext.O3
extern=extern.set_index('d')
eee=pd.DataFrame(extern)
eee['ind']=range(0,len(eee))
eee.set_index('ind', drop=True, inplace=True)
eee=eee.replace(0,np.nan)   
#%%
def R2(a,b):
    c=pd.DataFrame({"a":a, "b":b})
    cm=c.corr(method='pearson')
    r=cm.iloc[0,1]
    return r**2
#%%
corre=[]
for k in range(0,len(df_TS4)):
    fff=pd.DataFrame(df_TS4[k])
    fff['ind']=range(0,len(fff))
    fff.set_index('ind', drop=True, inplace=True)

    row=[k, R2(fff['F1'],fff['F2']),R2(fff['F1'],eee.mz55),R2(fff['F2'],eee['NOx']),
         R2(fff['F2'],eee['BC']),R2(fff['F2'],eee['NO']),R2(fff['F2'],eee['No2']),
         R2(fff['F2'],eee['mz57']),R2(fff['F3'],eee['mz60']),
         R2(fff['F3'],eee['mz73']),R2(fff['F3'],eee['chl'])]  
    print(k, row)  
    corre.append(row)    
corr=pd.DataFrame(data=corre, columns=['Run', 'COAvsHOA', 'COAvs55','HOAvsBC','HOAvsNOx','HOAvsNO','HOAvsNO2',
                           'HOAvs57','BBOAvs60','BBOAvs73', 'BBOAvsChl'])
corr.to_csv("Corr_table.txt", sep='\t')   

#%%
print(R2(fff.F2, fff.F1))
eee.mz55.plot()
fff.F1.plot()
#%%
# Step 2: Run_correlation R2 colormap:
#We create a list with as many dataframes as factors there are

fact_df=[pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
li_f1=[]
li_f2=[]
li_f3=[]
li_f4=[]
li_f5=[]
li_f6=[]
for k in range(0,len(df_PRO)):
    df_t=df_PRO[k].transpose()
    for j in range(0,len(df_t)):
        if j==0:
            li_f1.append(df_PRO[k]['F1'])
        if j==1:
            li_f2.append(df_PRO[k]['F2'])
        if j==2:
            li_f3.append(df_PRO[k]['F3'])
        if j==3:
            li_f4.append(df_PRO[k]['F4'])        
        if j==4:
            li_f5.append(df_PRO[k]['F5'])
        if j==5:
            li_f6.append(df_PRO[k]['F6'])
# MAtrixes with profiles for each run and factor (i.e.: F1 is for COA, F2 for HOA)            
F1=pd.DataFrame(li_f1, index=range(0,27))
F1=F1.transpose()
F2=pd.DataFrame(li_f2, index=range(0,27))  
F2=F2.transpose()      
F3=pd.DataFrame(li_f3, index=range(0,27))  
F3=F3.transpose()      
F4=pd.DataFrame(li_f4, index=range(0,27)) 
F4=F4.transpose()       
F5=pd.DataFrame(li_f5, index=range(0,27))        
F5=F5.transpose()
F6=pd.DataFrame(li_f6, index=range(0,27))        
F6=F6.transpose()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
# Displaying correlations for each factor (do manually)
corr_f1=F1.corr()
mask = np.triu(np.ones_like(corr_f1, dtype=bool))
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr_f1, mask=mask, annot=True,cmap="YlGnBu",center=0.5,square=True)

#%%
          
            
            