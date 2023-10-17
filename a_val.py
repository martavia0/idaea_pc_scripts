# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:27:38 2020

@author: Marta Via
"""
import pandas as pd
import os as os
import numpy as np
#%%
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2014_05_2015_05/Peaks")
ext= pd.read_csv("externals.txt", sep="\t", low_memory=False)
#BC= pd.read_csv("Errors.txt", sep="\t", low_memory=False, skip_blank_lines=False, parse_dates=True,infer_datetime_format=True)
#%%
low_lim = 0.0
up_lim = float (input("Up to which limit do you want to do the Sensitive Analysis? "))
num_fact = int(input("How many factors have you run? "))
#%% MATRIX PREPARED
runs = pd.DataFrame(columns=["Run", "aCOA", "aHOA", "aBBOA"])
num_runs = int((((up_lim - low_lim)*10)+1)**2)
rang=np.sqrt(num_runs)
runs["Run"]=range(0,num_runs,1)
runs["aCOA"]=[input("Value of fixed COA: ")]*(num_runs)
runs["aHOA"]=(list(np.arange(low_lim,up_lim+0.1,0.1))*int(rang))
l=[]
li=[]
for i in range(0,int(rang),1):
    l.append([float(i)]*int(rang))
    li = np.concatenate(l)
runs["aBBOA"]= li/10.0
#%%
os.chdir("C:/Users/Marta Via/Desktop")
sol1= pd.read_csv("prova.txt", sep="\t", low_memory=False, skip_blank_lines=False)
sol2 = pd.read_csv("prova2.txt", sep="\t", low_memory=False, skip_blank_lines=False)
sol2.columns=["COA", "HOA", "BBOA","OOA"]
#%%
lim=(sol2.index[sol2['COA'] == "sol_ts_fact_mean"].tolist()
#%%   OKKKK UNTIL HERE
spl=[int(i) for i in 'lim']
#h_letters = [ letter for letter in 'human' ]




for i in range(0,len(lim)):
    df=sol2[i):]
#%%
df=sol[]

k=0

    #%%
    for j in range(0,len(sol2)):
        name="sol_"+str(k)
        df=pd.DataFrame(sol[lim.iloc[i]:lim.iloc[i+1]])
        df.rename()
    k=k+1    
#%%
