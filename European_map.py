
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:51:21 2023

@author: Marta Via
"""
#%%
import plotly 
import pandas as pd
import os as os
import matplotlib.pyplot as plt
# import Treatment
#%%
path="C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/RI-Urbans/"
os.chdir(path)
df=pd.read_csv('Urban_NRPM1.txt', sep='\t')
print(df.head())
#%%
trt = Treatment()
trt.Hello()
print(trt.x)
#%%
import plotly.express as px
fig = px.scatter_geo(df, lat="Lat", lon="Lon", color="Study",
                     hover_name="Country", size="NRPM1 (ugm3)", size_max=40,
                     projection="natural earth")
fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="Grey")
fig.show()
fig.write_html("file.html")
# fig.write_image("file.png") # save as test.png
#%%
oa=pd.read_csv('OA_SA.txt', sep='\t', dayfirst=True)
oa['dt']=pd.to_datetime(oa['Start_period'], dayfirst=True)
oa['Site_Period']=(oa['Site'] +' (' + oa['dt'].dt.year.astype(str)+')')#[-4:]
oa['Period_Site']=(oa['dt'].dt.year.astype(str)+oa['Site'])#[-4:]
oa['Year']=oa['dt'].dt.year
oa=oa.sort_values(by= 'Period_Site')
oa_sa2=oa[['HOA', 'COA', 'BBOA', 'LO-OOA', 'MO-OOA', 'CSOA', 'ShINDOA', 'Coffee-Roaster OA', 'CCOA']]
oa_sa2['OOA'] = oa_sa2['LO-OOA']+oa_sa2['MO-OOA']
oa_sa2 = oa_sa2[['HOA', 'COA', 'BBOA', 'LO-OOA', 'MO-OOA','OOA', 'CSOA', 'ShINDOA', 'Coffee-Roaster OA', 'CCOA']]
oa_sa=oa_sa2[oa_sa2.columns[::-1]]
del oa_sa2
# oa_sa = oa_sa.T/oa_sa.sum()
#%%
fig, axs=plt.subplots(figsize=(6,4))
bp = dict(linestyle='-', linewidth=1, color='k')
mp = dict(linestyle='-', linewidth=1.5, color='darkgrey')
mmp = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
wp = dict(linestyle='-', linewidth=1, color='k')
oa_sa.boxplot(showmeans=True, vert=False, boxprops=bp, medianprops=mp, meanprops=mmp, whiskerprops=wp, ax=axs)
nb_sites=oa_sa.count()
for i in range(0,len(nb_sites)):
    axs.text(-3, i+0.9, '('+str(nb_sites[i])+')')
axs.set_xlabel('Percentage to NR-PM$_1$ (%)', fontsize=12)
axs.set_ylabel('Factors', fontsize=12)
#%%
oa_sa2=oa_sa[oa_sa.columns[::-1]]
oa_sa2.index=oa['Site_Period']
cc=['grey', 'olive', 'sienna', 'lightgreen', 'darkgreen','mediumpurple', 'steelblue', 'gold','midnightblue' ]
fig, axs=plt.subplots(figsize=(7,5))
# oa_sa2.drop(labels=['North Kensington (2013)'], inplace=True)
oa_sa2.drop('OOA', axis=1, inplace=True)
oa_sa2.plot.bar(stacked=True, color=cc,ax=axs)
axs.set_xlabel('Site (Period start)', fontsize=14)
axs.set_ylabel('Percentage of OA (%)', fontsize=14)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", )
# axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%%
oa['LOMO']=oa['LO-OOA']/oa['MO-OOA']
#%%
# oa.drop(labels=[1], inplace=True)
fig, axs=plt.subplots(figsize=(4,4))
axs.scatter(x=oa['Lat'], y=oa['BBOA'], color='sienna')
for i in range(len(oa)):
  axs.text(oa['Lat'].iloc[i], oa['BBOA'].iloc[i], oa['Acronym'].iloc[i], size=12)
axs.set_xlim(30,65)
axs.set_ylim(0,40)
axs.grid('x')
axs.set_xlabel('Latitude', fontsize=14)
axs.set_ylabel('BBOA (%)', fontsize=14)
axs.text(x=31, y=37, s='R$^2$ = 0.31', fontsize=14)
#%%
oa_ug=pd.read_csv('OA_SA_ug.txt', sep='\t', dayfirst=True)
fig, axs=plt.subplots(figsize=(6,4))
oa_ug['OOA']=oa_ug['LO-OOA']+oa_ug['MO-OOA']
oa_ug2=oa_ug[['CCOA','Coffee-Roaster OA', 'ShINDOA', 'CSOA','OOA','MO-OOA', 'LO-OOA','BBOA','COA', 'HOA']]
oa_ug2.boxplot(showmeans=True, vert=False, boxprops=bp, medianprops=mp, meanprops=mmp, whiskerprops=wp, ax=axs,showfliers=True)
axs.set_xlabel('Mass concentration ($\mu g$·$m^{-3}$)', fontsize=12)
axs.set_ylabel('Factors', fontsize=12)
#%%
oa_ug['LO_MO']=oa_ug['LO-OOA']/oa_ug['MO-OOA']
fig, axs=plt.subplots(figsize=(4,4))
axs.scatter(x=oa_ug['Lat'], y=oa_ug['LO_MO'], color='green')
for i in range(len(oa_ug)):
  axs.text(oa_ug['Lat'].iloc[i], oa_ug['LO_MO'].iloc[i], oa_ug['Acronym'].iloc[i], size=12)
axs.set_xlim(30,65)
axs.set_ylim(0,3)
axs.grid('x')
axs.set_xlabel('Latitude', fontsize=14)
axs.set_ylabel('LO-OOA / MO-OOA (adim.)', fontsize=14)
axs.text(x=31, y=2.75, s='R$^2$ = 0.06', fontsize=14)


#%%
bp = dict(linestyle='-', linewidth=1, color='k')
mdp = dict(linestyle='-', linewidth=1.5, color='darkgrey')
mp = dict(marker='o',linewidth=1, markeredgecolor='black', markerfacecolor='k')
wp = dict(linestyle='-', linewidth=1, color='k')





