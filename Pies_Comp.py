# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:51:24 2021

@author: Marta Via
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:32:10 2021

@author: Marta Via
"""

from scipy.odr import Model, Data, ODR
from scipy.stats import norm, kstest
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
#import statsmodels.api as sm

# %% Importing BARCELONA
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Barcelona")
bcn = pd.read_csv("Barcelona.txt", sep="\t", low_memory=False)
dr_all = pd.date_range("2017/09/21 00:00", end="2018/10/30")  # BCN
bcn['Nox'] = bcn['NOx']
bcn['Sc Res_R']=bcn['Sc Res_R']/92.0
bcn['Sc Res_S']=bcn['Sc Res_S']/92.0
bcn['Time'] = pd.to_datetime(bcn['datetime'], dayfirst=True, errors='coerce')
bcn['OA_app_Rolling']=bcn['OA_app_R']
bcn['OA_app_seas']=bcn['OA_app_s']
bcn['OA_app_S']=bcn['OA_app_s']
city='BCN-PR'
cityname='Barcelona - Palau Reial'
bcn_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
bcn_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
# %% BUCHAREST
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Bucharest/")
bcr = pd.read_csv("Bucharest.txt", sep="\t", low_memory=False, index_col=False)
bcr['Time'] = pd.to_datetime(bcr['datetime'], dayfirst=True, errors='coerce')
dr_all = pd.date_range("2016/09/01 00:00", end="2017/08/31")  # BUC
bcr['OA_app_S']=bcr['OA_app_s']
bcr['OA_app_seas']=bcr['OA_app_s']
bcr['OA_app_Rolling']=bcr['OA_app_R']
bcr['OA_app_seas']=bcr['OA_app_s']
bcr['Nox'] = bcr['NOx']
bcr['Sc Res_R']=bcr['Sc Res_R']/92.0
bcr['Sc Res_S']=bcr['Sc Res_S']/92.0
city='BCR'
cityname='Magurele - INOE'
bcr_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
bcr_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
# %% CYPRUS
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Cyprus")
cao = pd.read_csv("Cyprus.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2015/03/01 00:00", end="2017/01/16")  # TAR
cao['Time'] = pd.to_datetime(cao['datetime'], dayfirst=True, errors='coerce')
cao['BBOA_seas']=cao['BBOA_seas'].replace(0,np.nan)
cao['OA_app_s'],cao['OA_app_seas'],cao['OA_app_Rolling']=cao['OA_app_S'],cao['OA_app_S'],cao['OA_app_R']
cao['NOx']=cao['NO']+cao['NO2']
cao['OOA_Rolling']=cao['LO-OOA_Rolling']+cao['MO-OOA_Rolling']
cao['Sc Res_R']=cao['Sc Res_R']#/72.0
cao['Sc Res_S']=cao['Sc Res_S']#/72.0
city='CAO-AMX'
cao_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
cityname='Cyprus Atm. Obs. - Agia Xyliatou'
cao_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
# %% Importing DUBLIN
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Dublin")
dub = pd.read_csv("Dublin.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2016/09/01 00:00", end="2017/08/31")  # DUB#%%%
dub['Time'] = pd.to_datetime(dub['datetime'], dayfirst=True, errors='coerce')
dub['Sc Res_R']=dub['Sc Res_R']*70.0
dub['Sc Res_S']=dub['Sc Res_S']*70.0
# dub['OA_app_Rolling']=dub['OA_app_R']
# dub['OA_app_seas']=dub['OA_app_s']
dub_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
dub_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
# Scaled residuals do have to be normalized for the seasonal but not for the rolling.
city='DUB'
cityname='Dublin'
# %% Importing LILLE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Lille")
atoll = pd.read_csv("Lille.txt", sep="\t", low_memory=False)
dr_all = pd.date_range("2016/10/11 00:00", end="2017/08/18")  # LIL
atoll['Time'] = pd.to_datetime(atoll['datetime'], dayfirst=True, errors='coerce')
atoll['Sc Res_R']=atoll['Sc Res_R']/73.0
atoll['Sc Res_S']=atoll['Sc Res_S']/73.0
atoll['OA_app_S']=atoll['OA_app_s']
atoll['OA_app_seas']=atoll['OA_app_s']
atoll['OA_app_Rolling']=atoll['OA_app_R']
atoll_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
atoll_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
city='ATOLL'
cityname='ATOLL'
# %% Importing MAGADINO
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Magadino")
mgd = pd.read_csv("Magadino.txt", sep="\t")
dr_all = pd.date_range("2013/08/28 00:00", end="2014/10/30")  # MAG
mgd['Sc Res_R'] = mgd['Sc Res_R']/70.0
mgd['Sc Res_S'] = mgd['Sc Res_S']/70.0
mgd['OA_app_Rolling'] = mgd['HOA_Rolling']+mgd['BBOA_Rolling'] + mgd['LO-OOA_Rolling']+mgd['MO-OOA_Rolling']+mgd['LOA_Rolling']  # MAG
mgd['OA_app_R']=mgd['OA_app_Rolling']
f7 = pd.DataFrame(mgd, columns=['HOA_seas', 'BBOA_seas', 'LO-OOA_seas', 'MO-OOA_seas', 'LOA_seas'])
mgd['OA_app_seas'] = f7.sum(axis=1, skipna=True)  # MAG
mgd['OA_app_s'] = mgd['OA_app_seas']
mgd['Time'] = pd.to_datetime(mgd['datetime'], dayfirst=True, errors='coerce')
mgd['OA_app_S']=mgd['OA_app_s']
#mgd['OOA_Rolling']=mgd['MO-OOA_Rolling']+mgd['LO-OOA_Rolling']
mgd_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
mgd_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
city='MGD'
cityname='Magadino'
# %% MARSEILLE
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Marseille")
mrs = pd.read_csv("Marseille.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2017/01/31 00:00", end="2018/04/13")  # MAR
#mrs['OA_app_R']=mrs['COA_Rolling']+mrs['HOA_Rolling']+mrs['BBOA_Rolling'] + mrs['SHINDOA_Rolling']+mrs['LO-OOA_Rolling']+mrs['MO-OOA_Rolling']
mrs['OA_app_Rolling']=mrs['OA_app_R']
mrs['OA_app_S']=mrs['OA_app_s']
mrs['OA_app_seas']=mrs['OA_app_s']
mrs['Time'] = pd.to_datetime(mrs['datetime'], dayfirst=True, errors='coerce')
mask = ~np.isnan(mrs['Org']) & ~ np.isnan(mrs['OA_app_R']) & ~np.isnan(mrs['OA_app_S']) &~(mrs['OA_app_R']==0)& ~(mrs['OA_app_S']==0)
mrs=mrs[mask]
mrs_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
mrs_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
city='MRS-LCP'
cityname='Marseille - Longchamp'
# %% SIRTA
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/SIRTA")
sir = pd.read_csv("SIRTA.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2016/01/01 00:00", end="2017/06/01")  # TAR
sir['Time'] = pd.to_datetime(sir['datetime'], dayfirst=True, errors='coerce')
sir['OA_app_R']=sir['OA_app_Rolling']
sir['OA_app_seas']=sir['OA_app_S']
city = 'SIR'
sir_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
sir_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
# %% TARTU
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Tartu")
trt = pd.read_csv("TARTU.txt", sep="\t", low_memory=False, index_col=False)
dr_all = pd.date_range("2016/09/01 00:00", end="2017/07/24")  # TAR
trt['OA_app_R'] = trt['HOA_Rolling']+trt['BBOA_Rolling'] + trt['LO-OOA_Rolling']+trt['MO-OOA_Rolling']
trt['OA_app_Rolling']=trt['OA_app_R']
# trt['OA_app_seas']= trt['OA_app_s']
trt['Time'] = pd.to_datetime(trt['datetime'], dayfirst=True, errors='coerce')
trt_a2_tr=pd.read_csv('A2_tr_4443.txt', sep="\t", low_memory=False)
trt['Sc Res_R']=trt['Sc Res_R']*73.0
trt['Sc Res_S']=trt['Sc Res_S']*73.0
city_='TRT'
trt_a2=pd.read_csv('A2_4443.txt', sep="\t", low_memory=False)
cityname='Tartu'
#%% PIES OF APPORTIONMENT!
# PIE OF APPORTIONMENT

fig, axs = plt.subplots(3,6, figsize=(15,9))
# fig.suptitle('Barcelona',fontsize=25, y=1)
#BARcelona
fracs_R = [(bcn['COA_Rolling']/bcn['OA_app_Rolling']).mean(),(bcn['HOA_Rolling']/bcn['OA_app_Rolling']).mean(),
           (bcn['BBOA_Rolling']/bcn['OA_app_Rolling']).mean(),
           (bcn['LO-OOA_Rolling']/ bcn['OA_app_Rolling']).mean(), (bcn['MO-OOA_Rolling']/bcn['OA_app_Rolling']).mean()]
fracs_S = [(bcn['COA_seas']/bcn['OA_app_seas']).mean(),(bcn['HOA_seas']/bcn['OA_app_seas']).mean(),(bcn['BBOA_seas']/bcn['OA_app_seas']).mean(),
           (bcn['LO-OOA_seas']/ bcn['OA_app_seas']).mean(), (bcn['MO-OOA_seas']/bcn['OA_app_seas']).mean()]
my_labels='COA', 'HOA', 'BBOA','LO-OOA','MO-OOA'
axs[0,0].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['mediumorchid','grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[0,0].set_title('Barcelona - Palau Reial',fontsize=18, x=1.15, y=1)
axs[0,1].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['mediumorchid','grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
# axs[1,0].set_title('Seasonal')
axs[0,0].set_xlabel('ROLLING', fontsize=16)
axs[0,1].set_xlabel('SEASONAL', fontsize=16)

#Cyprus
fracs_R = [(cao['HOA_Rolling']/cao['OA_app_Rolling']).mean(),(cao['BBOA_Rolling']/cao['OA_app_Rolling']).mean(),
           (cao['LO-OOA_Rolling']/ cao['OA_app_Rolling']).mean(), (cao['MO-OOA_Rolling']/cao['OA_app_Rolling']).mean()]
fracs_S = [(cao['HOA_seas']/cao['OA_app_seas']).mean(),(cao['BBOA_seas']/cao['OA_app_seas']).mean(),
           (cao['LO-OOA_seas']/ cao['OA_app_seas']).mean(), (cao['MO-OOA_seas']/cao['OA_app_seas']).mean()]
my_labels='HOA', 'BBOA','LO-OOA','MO-OOA'
axs[0,2].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True,
             startangle=90, counterclock=False, colors=['grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[0,2].set_title('Cyprus Atm. Obs. - Agia Xyliatou',fontsize=18,  x=1.1, y=1)
axs[0,3].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
# axs[1,2].set_title('Seasonal')
axs[0,2].set_xlabel('ROLLING', fontsize=16)
axs[0,3].set_xlabel('SEASONAL', fontsize=16)
#Dublin
fracs_R = [(dub['HOA_Rolling']/dub['OA_app_Rolling']).mean(),(dub['Wood_Rolling']/dub['OA_app_Rolling']).mean(),
           (dub['Coal_Rolling']/dub['OA_app_Rolling']).mean(),(dub['Peat_Rolling']/dub['OA_app_Rolling']).mean(),
           (dub['LO-OOA_Rolling']/ dub['OA_app_Rolling']).mean(), (dub['MO-OOA_Rolling']/dub['OA_app_Rolling']).mean()]
fracs_S = [(dub['HOA_seas']/dub['OA_app_seas']).mean(),(dub['Wood_Rolling']/dub['OA_app_Rolling']).mean(),
           (dub['Coal_seas']/dub['OA_app_Rolling']).mean(),(dub['Peat_Rolling']/dub['OA_app_Rolling']).mean(),
           (dub['LO-OOA_seas']/ dub['OA_app_seas']).mean(), (dub['MO-OOA_seas']/dub['OA_app_seas']).mean()]
my_labels='HOA','WCOA', 'CCOA', 'PCOA','LO-OOA','\n\nMO-OOA'
axs[0,4].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','darkkhaki','sienna', 'tan','limegreen','darkgreen'])#'olive','mediumorchid','lightskyblue', 
axs[0,4].set_title('Dublin',fontsize=18, x=1.15, y=1)
axs[0,5].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey', 'darkkhaki','sienna', 'tan','limegreen','darkgreen' ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[0,4].set_xlabel('ROLLING', fontsize=16)
axs[0,5].set_xlabel('SEASONAL', fontsize=16)
#ATOLL
fracs_R = [(atoll['HOA_Rolling']/atoll['OA_app_Rolling']).mean(),(atoll['BBOA_Rolling']/atoll['OA_app_Rolling']).mean(),
           (atoll['LO-OOA_Rolling']/ atoll['OA_app_Rolling']).mean(), (atoll['MO-OOA_Rolling']/atoll['OA_app_Rolling']).mean()]
fracs_S = [(atoll['HOA_seas']/atoll['OA_app_seas']).mean(),(atoll['BBOA_seas']/atoll['OA_app_seas']).mean(),
           (atoll['LO-OOA_seas']/ atoll['OA_app_seas']).mean(), (atoll['MO-OOA_seas']/atoll['OA_app_seas']).mean()]
my_labels='HOA', 'BBOA','LO-OOA','\n\nMO-OOA'
axs[1,0].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[1,0].set_title('Lille Atm. Obs. (ATOLL)', fontsize=18, x=1.15, y=1)
axs[1,1].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1,0].set_xlabel('ROLLING', fontsize=16)
axs[1,1].set_xlabel('SEASONAL', fontsize=16)
#MAGADINO
fracs_R = [(mgd['LOA_Rolling']/mgd['OA_app_Rolling']).mean(),(mgd['HOA_Rolling']/mgd['OA_app_Rolling']).mean(),
           (mgd['BBOA_Rolling']/mgd['OA_app_Rolling']).mean(),
           (mgd['LO-OOA_Rolling']/ mgd['OA_app_Rolling']).mean(), (mgd['MO-OOA_Rolling']/mgd['OA_app_Rolling']).mean()]
fracs_S = [(mgd['LOA_seas']/mgd['OA_app_seas']).mean(),(mgd['HOA_seas']/mgd['OA_app_seas']).mean(),(mgd['BBOA_seas']/mgd['OA_app_seas']).mean(),
           (mgd['LO-OOA_seas']/ mgd['OA_app_seas']).mean(), (mgd['MO-OOA_seas']/mgd['OA_app_seas']).mean()]
my_labels='58-OA', 'HOA', '\nBBOA','LO-OOA','\n\n\nMO-OOA'
axs[1,2].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['olive','grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[1,2].set_title('Magadino',fontsize=18, x=1.15, y=1)
axs[1,3].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['olive','grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1,2].set_xlabel('ROLLING', fontsize=16)
axs[1,3].set_xlabel('SEASONAL', fontsize=16)


#Bucharest
fracs_R = [(bcr['HOA_Rolling']/bcr['OA_app_Rolling']).mean(),(bcr['BBOA_Rolling']/bcr['OA_app_Rolling']).mean(),
           (bcr['LO-OOA_Rolling']/ bcr['OA_app_Rolling']).mean(), (bcr['MO-OOA_Rolling']/bcr['OA_app_Rolling']).mean()]
fracs_S = [(bcr['HOA_seas']/bcr['OA_app_seas']).mean(),(bcr['BBOA_seas']/bcr['OA_app_seas']).mean(),
           (bcr['LO-OOA_seas']/ bcr['OA_app_seas']).mean(), (bcr['MO-OOA_seas']/bcr['OA_app_seas']).mean()]
my_labels='HOA', 'BBOA','LO-OOA','MO-OOA'
axs[1,4].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[1,4].set_title('Magurele - INOE',fontsize=18, x=1.15, y=1)
axs[1,5].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True,startangle=90, counterclock=False,
             colors=['grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1,4].set_xlabel('ROLLING', fontsize=16)
axs[1,5].set_xlabel('SEASONAL', fontsize=16)
#Marseille
fracs_R = [(mrs['COA_Rolling']/mrs['OA_app_Rolling']).mean(),(mrs['HOA_Rolling']/mrs['OA_app_Rolling']).mean(),
           (mrs['SHINDOA_Rolling']/mrs['OA_app_Rolling']).mean(),(mrs['BBOA_Rolling']/mrs['OA_app_Rolling']).mean(),
           (mrs['LO-OOA_Rolling']/ mrs['OA_app_Rolling']).mean(), (mrs['MO-OOA_Rolling']/mrs['OA_app_Rolling']).mean()]
fracs_S = [(mrs['COA_seas']/mrs['OA_app_seas']).mean(),(mrs['HOA_seas']/mrs['OA_app_seas']).mean(),
           (mrs['SHINDOA_seas']/mrs['OA_app_seas']).mean(),(mrs['BBOA_seas']/mrs['OA_app_seas']).mean(),
           (mrs['LO-OOA_seas']/ mrs['OA_app_seas']).mean(), (mrs['MO-OOA_seas']/mrs['OA_app_seas']).mean()]
my_labels='COA', 'HOA','\nSHINDOA','BBOA','LO-OOA','MO-OOA'
axs[2,0].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['mediumorchid','grey','lightskyblue', 'brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[2,0].set_title('Marseille - Longchamp',fontsize=18, x=1.15, y=1)
axs[2,1].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['mediumorchid','grey','lightskyblue','brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[2,0].set_xlabel('ROLLING', fontsize=16)
axs[2,1].set_xlabel('SEASONAL', fontsize=16)
#SIRTA
fracs_R = [(sir['HOA_Rolling']/sir['OA_app_Rolling']).mean(),
           (sir['BBOA_Rolling']/sir['OA_app_Rolling']).mean(),
           (sir['LO-OOA_Rolling']/ sir['OA_app_Rolling']).mean(), (sir['MO-OOA_Rolling']/sir['OA_app_Rolling']).mean()]
fracs_S = [(sir['HOA_seas']/sir['OA_app_seas']).mean(),
           (sir['BBOA_seas']/sir['OA_app_seas']).mean(),
           (sir['LO-OOA_seas']/ sir['OA_app_seas']).mean(), (sir['MO-OOA_seas']/sir['OA_app_seas']).mean()]
my_labels='HOA','\nBBOA','LO-OOA','MO-OOA'
axs[2,2].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[2,2].set_title('SIRTA',fontsize=18, x=1.15, y=1)
axs[2,3].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[2,2].set_xlabel('ROLLING', fontsize=16)
axs[2,3].set_xlabel('SEASONAL', fontsize=16)
#TARTU
fracs_R = [(trt['HOA_Rolling']/trt['OA_app_Rolling']).mean(),(trt['BBOA_Rolling']/trt['OA_app_Rolling']).mean(),
           (trt['LO-OOA_Rolling']/ trt['OA_app_Rolling']).mean(), (trt['MO-OOA_Rolling']/trt['OA_app_Rolling']).mean()]
fracs_S = [(trt['HOA_seas']/trt['OA_app_seas']).mean(),(trt['BBOA_seas']/trt['OA_app_seas']).mean(),
           (trt['LO-OOA_seas']/ trt['OA_app_seas']).mean(), (trt['MO-OOA_seas']/trt['OA_app_seas']).mean()]
my_labels='HOA', 'BBOA','LO-OOA','MO-OOA'
axs[2,4].pie(fracs_R,labels=my_labels,autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey','brown','limegreen','darkgreen',])#'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan' 
axs[2,4].set_title('Tartu',fontsize=18, x=1.15, y=1)
axs[2,5].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%', shadow=True, startangle=90, counterclock=False,
             colors=['grey', 'brown','limegreen','darkgreen', ])#'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[2,4].set_xlabel('ROLLING', fontsize=16)
axs[2,5].set_xlabel('SEASONAL', fontsize=16)
# axs[1,7].set_title('')



#%% SCALED RESIDUALS
# SCALED RESIDUALS


fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(12,12))
fig.suptitle('Scaled Residuals',fontsize=25, y=1)
#Barcelona
bcn['Sc Res_R'].plot.hist(ax=axs[0,0], density=True,bins=300, alpha=0.5,  color='tomato')
bcn['Sc Res_S'].plot.hist(ax=axs[0,0], density=True, bins=300, alpha=0.5,color='blue')
axs[0,0].set_title('Barcelona - Palau Reial',fontsize=15)
axs[0,0].grid()
axs[0,0].set_xlim(-3,3)
# cao['Sc Res_R']=cao['Sc Res_R']*72.0
# cao['Sc Res_S']=cao['Sc Res_S']*72.0
#Cyprus
cao['Sc Res_R'].plot.hist(ax=axs[0,1], density=True, bins=300, alpha=0.5,  color='tomato')
cao['Sc Res_S'].plot.hist(ax=axs[0,1], density=True,bins=300, alpha=0.5,color='blue')
axs[0,1].set_title('Cyprus Atm. Obs.- '+'\n'+'Agia Xyliatou',fontsize=14)
axs[0,1].grid()
axs[0,1].set_xlim(-2,2)
#Dublin
dub['Sc Res_R'].plot.hist(ax=axs[0,2], density=True,bins=300, alpha=0.5,color='tomato')
dub['Sc Res_S'].plot.hist(ax=axs[0,2],density=True, bins=300, alpha=0.5,color='blue')
axs[0,2].set_title('Dublin',fontsize=15)
axs[0,2].grid()
axs[0,2].set_xlim(-3,3)
#ATOLL
atoll['Sc Res_R'].plot.hist(ax=axs[1,0], density=True,bins=300, alpha=0.5,  color='tomato')
atoll['Sc Res_S'].plot.hist(ax=axs[1,0], density=True,bins=300, alpha=0.5,color='blue')
axs[1,0].set_title('ATOLL',fontsize=15)
axs[1,0].grid()
# axs[0,3].set_xlim(-200,200)
#Magadino
mgd['Sc Res_R'].plot.hist(ax=axs[1,1],density=True, bins=300, alpha=0.5,  color='tomato')
mgd['Sc Res_S'].plot.hist(ax=axs[1,1], density=True,bins=300, alpha=0.5,color='blue')
axs[1,1].set_title('Magadino',fontsize=15)
axs[1,1].grid()
axs[1,1].set_xlim([-3,3])
#Magurele -INOE
bcr['Sc Res_R'].plot.hist(ax=axs[1,2],density=True, bins=300, alpha=0.5,  color='tomato')
bcr['Sc Res_S'].plot.hist(ax=axs[1,2],density=True, bins=300, alpha=0.5,color='blue')
axs[1,2].set_title('Magurele - INOE',fontsize=15)
axs[1,2].grid()
axs[1,2].set_xlim(-1,1)
#Marseille
mrs['Sc Res_R'].plot.hist(ax=axs[2,0], density=True,bins=300, alpha=0.5,  color='tomato')
mrs['Sc Res_S'].plot.hist(ax=axs[2,0],density=True, bins=300, alpha=0.5,color='blue')
axs[2,0].set_title('Marseille - Longchamp',fontsize=15)
axs[2,0].grid()
axs[2,0].set_xlim(-7,7)
#SIRTA
sir['Sc Res_R'].plot.hist(ax=axs[2,1], density=True,bins=300, alpha=0.5,  color='tomato')
sir['Sc Res_S'].plot.hist(ax=axs[2,1],density=True, bins=300, alpha=0.5,color='blue')
axs[2,1].set_title('SIRTA',fontsize=15)
axs[2,1].grid()
axs[2,1].set_xlim(-1,1)
axs[2,1].set_ylim(0,20)
#Tartu
trt['Sc Res_R'].plot.hist(ax=axs[2,2], density=True,bins=300, alpha=0.5,  color='tomato')
trt['Sc Res_S'].plot.hist(ax=axs[2,2], density=True,bins=300, alpha=0.5,color='blue')
axs[2,2].set_title('Tartu',fontsize=15)
axs[2,2].grid()
axs[2,2].set_xlim([-1,1])

axs[0,1].set_ylabel('')
axs[0,2].set_ylabel('')
axs[1,1].set_ylabel('')
axs[1,2].set_ylabel('')
axs[2,1].set_ylabel('')
axs[2,2].set_ylabel('')


axs[0,0].set_ylabel('Frequency', fontsize=18)
axs[1,0].set_ylabel('Frequency', fontsize=18)

fig.legend(['Rolling', 'Seasonal'], loc=(0.45,0.91))
#%% ADAPTABILITY 2 on TRANSITION PERIODS

fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(10,12))
fig.suptitle('Adaptability plot 44/43'+'\n'+'TRANSITION PERIODS',fontsize=24, y=1.0)
#Barcelona
bcn_a2_tr['Substr_R'].plot.kde(ax=axs[0,0],  color='tomato')
bcn_a2_tr['Substr_S'].plot.kde(ax=axs[0,0],  color='blue')
axs[0,0].set_title('Barcelona - Palau Reial',fontsize=15)
axs[0,0].grid()
#Cyprus
cao_a2_tr['Substr_R'].plot.kde(ax=axs[0,1], color='tomato')
cao_a2_tr['Substr_S'].plot.kde(ax=axs[0,1] ,color='blue')
axs[0,1].set_title('Cyprus Atm. Obs.- '+'\n'+'Agia Xyliatou',fontsize=15)
axs[0,1].grid()
axs[0,1].set_xlim(-10,10)
#Dublin
dub_a2_tr['Substr_R'].plot.kde(ax=axs[0,2]  ,color='tomato')
dub_a2_tr['Substr_S'].plot.kde(ax=axs[0,2]  ,color='blue')
axs[0,2].set_title('Dublin',fontsize=15)
axs[0,2].grid()
axs[0,2].set_xlim(-100,100)
#ATOLL
atoll_a2_tr['Substr_R'].plot.kde(ax=axs[1,0],  color='tomato')
atoll_a2_tr['Substr_S'].plot.kde(ax=axs[1,0],color='blue')
axs[1,0].set_title('Atm. Obs. Lille (ATOLL)',fontsize=15)
axs[1,0].grid()
axs[1,0].set_xlim(-10,10)
#Magadino
mgd_a2_tr['Substr_R'].plot.kde(ax=axs[1,1],   color='tomato')
mgd_a2_tr['Substr_S'].plot.kde(ax=axs[1,1],color='blue')
axs[1,1].set_title('Magadino',fontsize=15)
axs[1,1].grid()
axs[1,1].set_xlim([-3,3])
#Magurele -INOE
bcr_a2_tr['Substr_R'].plot.kde(ax=axs[1,2],  color='tomato')
bcr_a2_tr['Substr_S'].plot.kde(ax=axs[1,2],color='blue')
axs[1,2].set_title('Magurele - INOE',fontsize=15)
axs[1,2].grid()
axs[1,2].set_xlim(-10,10)
#Marseille
mrs_a2_tr['Substr_R'].plot.kde(ax=axs[2,0],  color='tomato')
mrs_a2_tr['Substr_S'].plot.kde(ax=axs[2,0],color='blue')
axs[2,0].set_title('Marseille - Longchamp',fontsize=15)
axs[2,0].grid()
axs[2,0].set_xlim(-7,7)
#SIRTA
sir_a2_tr['Substr_R'].plot.kde(ax=axs[2,1],  color='tomato')
sir_a2_tr['Substr_S'].plot.kde(ax=axs[2,1],color='blue')
axs[2,1].set_title('SIRTA',fontsize=15)
axs[2,1].grid()
axs[2,1].set_xlim(-15,15)
#Tartu
trt_a2_tr['Substr_R'].plot.kde(ax=axs[2,2] ,  color='tomato')
trt_a2_tr['Substr_S'].plot.kde(ax=axs[2,2], color='blue')
axs[2,2].set_title('Tartu',fontsize=15)
axs[2,2].grid()
axs[2,2].set_xlim([-3,3])

axs[0,1].set_ylabel('')
axs[0,2].set_ylabel('')
axs[1,1].set_ylabel('')
axs[1,2].set_ylabel('')
axs[2,1].set_ylabel('')
axs[2,2].set_ylabel('')


axs[0,0].set_ylabel('Frequency', fontsize=18)
axs[1,0].set_ylabel('Frequency', fontsize=18)
axs[2,0].set_ylabel('Frequency', fontsize=18)
axs[2,1].set_xlabel('$m/z_{44} / m/z_{43} - (m/z_{44} / m/z_{43})_{OOA} $', fontsize=18)

fig.legend(['Rolling', 'Seasonal'], loc=(0.75,0.9), fontsize=15)


