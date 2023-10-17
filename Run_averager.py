# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:32:14 2021

@author: Marta Via
"""
import pandas as pd
import os as os
import glob
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
#                   RUN AVERAGER
#%%
path = r"C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/24hF/6F" # use your path
filenames = glob.glob(path + "/TimeSeries_*.txt")
print(filenames)
df_TS = []
for filename in filenames:
    df_TS.append(pd.read_csv(filename, skip_blank_lines=False, parse_dates=True, infer_datetime_format=True, sep="\t"))
#    
filenames = glob.glob(path + "/Profiles_*.txt")
df_PRO = []
for filename in filenames:
    df_PRO.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))
#
filenames = glob.glob(path + "/Rel_Prof_*.txt")
df_RELP = []
for filename in filenames:
    df_RELP.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))
#    
filenames = glob.glob(path + "/Residuals_TS_*.txt")
df_RESTS = []
for filename in filenames:
    df_RESTS.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))
#
filenames = glob.glob(path + "/Residuals_PR_*.txt")
df_RESPR = []
for filename in filenames:
    df_RESPR.append(pd.read_csv(filename, skip_blank_lines=False, sep="\t"))

#%%
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/orig")    
mz=pd.read_csv("mz.txt")
mzlab=pd.read_csv("mz_lab.txt", skip_blank_lines=(False))
class_pr=pd.read_csv("Class_PR.txt")
class_pr2=pd.read_csv("Class_PR2.txt", sep="\t") 
ts=pd.read_csv("utc_end_time.txt", skip_blank_lines=False, sep="\t", infer_datetime_format=True)
ts['Time']=pd.to_datetime(ts['utc_start_time'], dayfirst=True, errors='coerce')
#%%
def Reorder_Factors(df,neworder): #df is a run, neworder is a list with:
#  4F->  #1: HOA   #2: AN+BBOA   #3:AS+MO   #4:LO+metals
#  5F->  #1: HOA   #2: AN+BBOA   #3:AS+MO   #4:LO+industry  #5: >100
#  6F->  #1: HOA   #2: AN+BBOA   #3:AS+MO   #4:LO+ind+AN,AC  #5: LO+SN #6: 58,LO #7:%>100+K+BB
#  7F->  #1: HOA   #2: AN+BBOA   #3:AS+MO   #4:LO+ind+AN,AC  #5: LO+SN #%%>100+K+BB
#  8F-> #1: HOA  #2: BBOA  #3:AS+MO   #4:MO+AN  #5: LO+SN #6:>80 Cd  #7:%%>100+Fe,Mn.. #8: 58 Cl,Ni
#  9F-> #1:HOA #2:BBOA #3:AS+MO #4:LO+ind+Cl #5:LO+SN #6:COA+Cr #7: AN+SOA  #8:MO+NO3 #F9: >100
    df=df.dropna(axis=1, how='all')
    df2 = df[neworder]
    df2.columns = [''] * len(df2.columns)
    return df2
#%%   TIME SERIES AND RELPFILES:
a=Reorder_Factors(df_PRO[0], ['F1','F2','F3','F4','F5','F6'])#]'F1','F6','F3','F8'])#,'F2'])
b=Reorder_Factors(df_PRO[1], ['F3','F1','F5','F4','F6','F2'])#,'F8','F3','F7','F5'])#,'F6'])
#c=Reorder_Factors(df_TS[2], ['F7','F6','F2','F1']#,'F5','F4','F3','F8'])
#d=Reorder_Factors(df_PRO[3], ['F4','F7','F8','F1','F2','F5']#,'F3','F6'])
#e=Reorder_Factors(df_PRO[4], ['F4','F1','F8','F2','F5','F3']#,'F7','F6'])
#f=Reorder_Factors(df_PRO[5], ['F5','F4','F8','F1','F2','F3']#,'F6','F7'])
#%%
a=Reorder_Factors(df_RELP[0], ['RF1','RF2','RF3','RF4','RF5','RF6'])#,'RF3','RF8'])
b=Reorder_Factors(df_RELP[1], ['RF3','RF1','RF5','RF4','RF6','RF2'])#,'RF7','RF5'])
#c=Reorder_Factors(df_RELP[2], ['RF7','RF6','RF2','RF1','RF5','RF4','RF3','RF8'])
#d=Reorder_Factors(df_RELP[1], ['RF7','RF5','RF4','RF6','RF3','RF2','RF1'])
#e=Reorder_Factors(df_RELP[4], ['RF4','RF1','RF8','RF2','RF5','RF3','RF7','RF6'])
#f=Reorder_Factors(df_RELP[5], ['RF5','RF4','RF8','RF1','RF2','RF3','RF6','RF7'])
#%%     RESIDUALS
a=df_RESPR[0]
b=df_RESPR[1]
#c=df_RESTS[2]
#d=df_RESPR[3]
#e=df_RESTS[4]
#f=df_RESTS[5]

#%%
avg=(a+b)/2
#%%
avg_r=(a+b)/2
#%%
path = r"C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/24hF/6F/" # use your path
avg_r.to_csv(path+"REL_PR_Avg.txt", sep="\t")
#%% Res TS
Time_Series_R(avg)
#Hourly_R(avg)
#%% RES PR
Profiles_O(avg)
#%% Time Series
Time_Series_R(avg)
Pie_R(avg)
#Hourly_R(avg)
#%% PROFILES
avg.columns=['F1','F2','F3','F4','F5','F6']#,'F9']
avg_r.columns=['RF1','RF2','RF3','RF4','RF5','RF6']#,'RF9']
#%%
Profiles_NRBCFRP(avg,avg_r)
Profiles_OARP(avg,avg_r)
#%%
avg=avg.drop(116)
avg_r=avg_r.drop(116)
#%%
for i in range(0,4):
    df_PRO[i]=df_PRO[i].drop(116)#len(df_PRO[0]))
    df_RELP[i]=df_RELP[i].drop(116)#len(df_PRO[0]))
    df_RESPR[i]=df_RESPR[i].drop(116)
#%%
def Time_Series_R(df_TS):
    df_TS['dt']=pd.to_datetime(ts["Time"], dayfirst=True)
    df_TS=df_TS.set_index(df_TS.dt)
    del df_TS['dt']
    print(df_TS.head(5))
    df_TS=df_TS.dropna(axis=1, how='all')
    fig_ts=plt.figure()
    ts_p=df_TS.plot(subplots=True, figsize=(25,10), grid=True)
    plotname_TS = "RES_TS.png"
    plt.savefig(plotname_TS)
    plt.show() 
# ************** PIE PLOT ************
def Pie_R(df_TS):
    avg=df_TS.mean()
    fig_pie=plt.figure()
    avg.plot.pie(figsize=(5,5), autopct='%1.1f%%',startangle=90)
    plotname_pie="_Pie.png"
    fig_pie.savefig(plotname_pie)
# ****** HOURLY ******
def Hourly_R(df_TS):   
    df_TS['d']=pd.to_datetime(ts["Time"], dayfirst=True)
    df_TS['Hour']=df_TS['d'].dt.hour
    Hour=df_TS.groupby(df_TS['Hour']).mean()
    Hour=Hour.dropna(axis=1, how='all')
    #del Hour['Hour']
    figh=plt.figure()
    Hour.plot(subplots=True, figsize=(6,10))
    plotname_hour ="RES_Diel_.png"
    plt.savefig(plotname_hour)
    del df_TS['Hour']
def Profiles_O(df_O):    
    df_O['mz']=mzlab['mz_lab']
    fig_1=plt.figure()
    print(df_O.columns)
    pro=df_O.plot.bar(subplots=True, figsize=(25,15), x="mz", logy=True)
    plotname_PRO =  "PR_RES.png"
    fig_1 = plt.gcf()
    plt.savefig(plotname_PRO)      
#%%
#%% PROFILES
Profiles_NRBCFRP_R(avg,avg_rp)
Profiles_OARP_R(avg,avg_rp)

#%%








