# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:59:19 2020

@author: Marta Via
"""

#%%
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import numpy as np
import scipy
import glob
import math
import seaborn as sns
#%%
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/MTR/MT_20210202/24hF")
#print("Tell the byte at which the file cursor is:",my_file.tell())
os.listdir() 
#%%
l=[]
with open('REL_PR.txt') as f:   #Datetime, PR, TS, RES_ts, RES_pr, EXPL_var_ts, EXPL_var_pr
    lines=f.read().splitlines()
f.close()    
for i in range(0,len(lines)):
    if '_mean' in lines[i]:    #_mean for all except for datetime (_nb)
        l.append(i)
d=[]
len_l=len(l)
for i in range(0,len_l-1):
    d.append(lines[l[i]:l[i+1]])
d.append(lines[l[len_l-1]:])

#%%
#*************** EXPLAINED VARIATION TS*******
s="F1"+"\t"+"F2"+"\t"+"F3"+"\t"+"F4"+"\t"+"F5"+"\t"+"F6"+"\t"+"Unexpl1"+"\t"+"Unexpl2"+"\n"
for k in range(0,len_l):
    name_m="Expl_Var_TS"+str(k)+".txt"
    fw=open(name_m, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item+"\n")
#%%
#************* EXPLAINED VARIATION PROFILES ************ 
s="F1"+"\t"+"F2"+"\t"+"F3"+"\t"+"F4"+"\t"+"F5"+"\t"+"F6"+"\t"+"Unexpl1"+"\t"+"Unexpl2"+"\n"
for k in range(0,len_l):
    name_n="Expl_Var_PR_"+str(k)+".txt"
    fw=open(name_n, 'w')
    fw.write(s)
    for item2 in d[k][1:]:
        fw.write(item2+"\n")
fw.close()
#%%
# **************** TIME SERIES ***************
s="F1"+"\t"+"F2"+"\t"+"F3"+"\t"+"F4"+"\t"+"F5"+"\t"+"F6"+"\t"+"F7"+"\t"+"F8"+"\t"+"F9"+"\n"            
for k in range(0,len_l):
    name="TimeSeries_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()            
#%%     
#**************** DATETIME ****************** 
s="acsm_utc_end_time"+"\n"                 
for k in range(0,len(d)):
    print(k)
    name="Datetime_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()
#%%     
#**************** PROFILES ****************** 
s="F1"+"\t"+"F2"+"\t"+"F3"+"\t"+"F4"+"\t"+"F5"+"\t"+"F6"+"\t"+"F7"+"\t"+"F8"+"\t"+"F9"+"\n"                 
for k in range(0,len_l):
    name="Profiles_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()
#%%     
#**************** RELATIVE PROFILES ****************** 
s="RF1"+"\t"+"RF2"+"\t"+"RF3"+"\t"+"RF4"+"\t"+"RF5"+"\t"+"RF6"+"\t"+"RF7"+"\t"+"RF8"+"\t"+"RF9"+"\n"                 
for k in range(0,len_l):
    name="Rel_Prof_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()
#%%
#**************** RESIDUALS TS ****************** 
 s="Res"+"\t"+"Abs_Res"+"\t"+"Sc_Res"+"\t"+"Abs_Sc_Res"+"\t"+"QQexp"+"\n"                 
for k in range(0,len_l):
    name="Residuals_TS_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()
#%%
#**************** RESIDUALS PR ****************** 
 s="Res"+"\t"+"Abs_Res"+"\t"+"Sc_Res"+"\t"+"Abs_Sc_Res"+"\t"+"QQexp"+"\n"                 
for k in range(0,len_l):
    name="Residuals_PR_"+str(k)+".txt"
    fw=open(name, 'w')
    fw.write(s)
    for item in d[k][1:]:
        fw.write(item)
        fw.write("\n")
fw.close()




    