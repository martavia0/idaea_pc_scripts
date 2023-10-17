# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:49:49 2019

@author: Marta Via
"""
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
#%% FOR PERIOD 2014 - 2015
#We set as default the directory where we have our files.
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2014_05_2015_05/Variables/Summer_2014/SynopticPattern")
# Open all files under a recognosible name
synpat = pd.read_csv("SP_summer.txt", skip_blank_lines=False, sep=";", parse_dates=True,infer_datetime_format=True)
acsmbc = pd.read_csv("Summer_14.txt", skip_blank_lines=False,sep="/t",parse_dates=True, infer_datetime_format=True)
acsm_time = pd.read_csv("ACSM_time.txt", sep=';',dtype = int,keep_default_na=True,parse_dates=True, infer_datetime_format=True)

#%% FOR PERIOD 2017 - 2018
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Synoptic Pattern/Seasonal_20200309")
synpat = pd.read_csv("Retro_MVIA.txt", skip_blank_lines=False, sep=";", parse_dates=True,infer_datetime_format=True)
acsmbc = pd.read_csv("Seasonal.txt", skip_blank_lines=False,sep="\t",parse_dates=True, infer_datetime_format=True, engine="python")
acsm_time = pd.read_csv("ACSM_Time.txt", sep=';',keep_default_na=True,parse_dates=True, infer_datetime_format=True)
#%%
acsmbc.insert(4,"ACSM", value = acsmbc.COA + acsmbc.HOA  + acsmbc.LOOA + acsmbc.MOOA)#+acsmbc.BBOA
acsmbc.insert(5,"SynPatt", value= None)
acsmbc["SynPatt"] =acsmbc["SynPatt"]
acsmbc.insert(6,"ACSM_TIME", value = None)
#%%
synop=[]
#We append into python lists each compound. 
for j in range(0, len(synpat)):
    for i in range(0,len(acsmbc)-1):
        if acsm_time.DAY.iloc[i]==synpat.DIA.iloc[j] and acsm_time.MONTH.iloc[i]==synpat.MES.iloc[j] and acsm_time.YEAR.iloc[i]==synpat.ANY.iloc[j]: 
            acsmbc.SynPatt.iloc[i]=synpat.SP.iloc[j] 
            acsmbc.ACSM_TIME.iloc[i]=str(acsm_time.DAY.iloc[i])+"/"+str(acsm_time.MONTH.iloc[i])+"/"+str(acsm_time.YEAR.iloc[i])+" "+str(acsm_time.HOUR.iloc[i])+":"+str(acsm_time.MIN.iloc[i])
            synop.append(acsmbc.SynPatt.iloc[i])
#%%
l=acsmbc.SynPatt.value_counts()
x=range(len(l))
#l.plot.bar()
suma=sum(l)
l2=100.0*l/suma
oc=pd.DataFrame({'Absolute': l,'%': l2})
oc.plot.bar(subplots=True)
axes[1].legend(loc=2)
plt.savefig('Episodes_Period_B')
#%%
file_=acsmbc.to_csv('Episodes_so18_externals.txt',sep="\t",na_rep='NaN')
#%% POINTLESS
SP=pd.DataFrame(columns=["REG","ANW","ANT","NAF","ANW EN ALTURA","AW","MED","EU","AN","ANW-REG","AN EN ALTURA","REG-EU","REG-ANW","ANT-MED?","REG-ANW EN ALTURA","AN-REG?","AN EN ALTURA","AW EN ALTURA-REG","AN-ANW","ANT-AN EN SUPERFÍCIE","ANW HEIGHTS INTENSE","AW-ANT","AN-ANT?","MED-AW?","EU-AN","ANW HEIGHTS-ANT","AN-REG","AW-REG?","EU-MED","REG-AN","ANW-MED EN SUPERFICIE?","REG-ANW","REG-EU?","MED-REG","ANW EN ALTURA-REG","REG-MED","MED-NAF"])
print SP

#%%
REG=[]
REG_EU=[]
REG_ANW_en_altura=[]
REG_ANW=[]
AW=[]
AW_ANT=[]
AW_en_altura=[]
ANW=[]
ANW_en_altura=[]
ANW_intenso=[]
ANW_en_altura_intenso=[]
ANT=[]
ANT_MED=[]
ANT_AN_en_superficie=[]
ANT_ANW_en_altura=[]
NAF=[]
AW=[]
AW_REG=[]
MED=[]
MED_AW=[]
MED_REG=[]
MED_NAF=[]
EU=[]
EU_MED=[]
AN=[]
AN_EU=[]
AN_REG=[]
AN_en_altura=[]
AN_ANW=[]
AN_ANT=[]

#he anat canviant ACSM per LOrg_fp, BBOA_fp etc.
for i in range(0,len(acsmbc)):
        s=str(acsmbc.SynPatt.iloc[i])
        if s == "REG":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                REG.append(float(acsmbc.COA.iloc[i]))
        elif (s == "REG-ANW" or s =="ANW-REG"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                REG_ANW.append(float(acsmbc.COA.iloc[i]))
        elif s == "REG-EU" :
            if str(acsmbc.COA.iloc[i]) != "nan": 
                REG_EU.append(float(acsmbc.COA.iloc[i])) 
        elif s == "AW":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AW.append(float(acsmbc.COA.iloc[i]))
        elif s == "AW-ANT":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AW_ANT.append(float(acsmbc.COA.iloc[i]))
        elif s == "AW en altura":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AW_en_altura.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANW":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANW.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANW intenso":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANW_intenso.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANW en altura":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANW_en_altura.append(float(acsmbc.COA.iloc[i])) 
        elif s == "ANW en altura intenso":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANW_en_altura_intenso.append(float(acsmbc.COA.iloc[i]))  
        elif s == "ANT":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANT.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANT-MED":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANT_MED.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANT-AN en superficie":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANT_AN_en_superficie.append(float(acsmbc.COA.iloc[i]))
        elif s == "ANT-ANW en altura":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                ANT_ANW_en_altura.append(float(acsmbc.COA.iloc[i]))
        elif s == "NAF":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                NAF.append(float(acsmbc.COA.iloc[i]))
        elif s == "AW":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AW.append(float(acsmbc.COA.iloc[i]))
        elif s == "AW-REG":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AW_REG.append(float(acsmbc.COA.iloc[i]))
        elif s == "MED":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                MED.append(float(acsmbc.COA.iloc[i]))  
        elif s == "MED-AW":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                MED_AW.append(float(acsmbc.COA.iloc[i]))  
        elif (s == "MED-REG" or s=="REG-MED"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                MED_REG.append(float(acsmbc.COA.iloc[i]))
        elif s == "MED-NAF":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                MED_NAF.append(float(acsmbc.COA.iloc[i]))
        elif s == "EU":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                EU.append(float(acsmbc.COA.iloc[i]))  
        elif (s == "EU-AN" or s =="AN-EU"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN_EU.append(float(acsmbc.COA.iloc[i]))  
        elif (s == "EU-MED" or s=="MED-EU"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                EU_MED.append(float(acsmbc.COA.iloc[i])) 
        elif s == "AN":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN.append(float(acsmbc.COA.iloc[i]))   
        elif s == "REG-ANW en altura":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                REG_ANW_en_altura.append(float(acsmbc.COA.iloc[i])) 
        elif s == "AN en altura":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN_en_altura.append(float(acsmbc.COA.iloc[i]))
        elif s == "AN-ANW":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN_ANW.append(float(acsmbc.COA.iloc[i]))
        elif s == "AN":
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN.append(float(acsmbc.COA.iloc[i]))
        elif (s == "AN-REG" or s=="REG-AN"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN_REG.append(float(acsmbc.COA.iloc[i]))
        elif (s == "AN-ANT" or s=="ANT-AN"):
            if str(acsmbc.COA.iloc[i]) != "nan": 
                AN_ANT.append(float(acsmbc.COA.iloc[i]))
#%%
listi=[REG,REG_EU,REG_ANW,REG_ANW_en_altura,REG_ANW,ANW,ANW_en_altura,ANW_intenso,ANW_en_altura_intenso,AW_ANT,ANT,ANT_MED,
       ANT_AN_en_superficie,ANT_ANW_en_altura,NAF,AW,AW_REG,AW_en_altura,MED,MED_REG,MED_AW,MED_NAF,EU,EU_MED,AN,AN_EU,
       AN_REG,AN_en_altura,AN_ANW,AN_ANT]         
#np.array([REG,ANW,ANT, NAF, AW, MED, EU, AN])
df=pd.DataFrame(listi, index=['REG','REG_EU','REG_ANW','REG_ANW_en_altura','REG_ANW','ANW','ANW_en_altura','ANW_intenso',
                              'ANW_en_altura_intenso','AW_ANT','ANT','ANT_MED','ANT_AN_en_superficie','ANT_ANW_en_altura',
                              'NAF','AW','AW_REG','AW_en_altura','MED','MED_REG','MED_AW','MED_NAF','EU','EU_MED','AN',
                              'AN_EU','AN_REG','AN_en_altura','AN_ANW','AN_ANT'])
df=df.transpose()
#df_out=df.to_csv('SO4_nwm_out.txt',sep="\t",na_rep='NaN')
df.boxplot(showfliers=False,rot=90, grid=True)
#%%
is_REG =  acsmbc['SynPatt']=="REG-ANW"
REG_f = acsmbc[is_REG]

#%%


 