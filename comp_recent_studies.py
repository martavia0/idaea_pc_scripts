# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:13:04 2023

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import matplotlib.pyplot as plt
#%% In't Veld_2022
intveld_2022=pd.DataFrame({'compos':[30,21,18,13,10,5,3,1,1]})
intveld_2022=intveld_2022/intveld_2022.sum()
intveld_2022_labels = ['Secondary \nsulphate', 'Combustion', 'OC rich', 'Secondary \nnitrate', 'Road dust', 'Heavy oil', 'Sea Spray', 'Mineral', 'Industry']
intveld_2022_color= ['red', 'grey', 'green', 'blue', 'dimgrey', 'darkred', 'deepskyblue', 'tan', 'magenta']
#
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
intveld_2022.T.plot.bar(stacked=True, legend=False,color=intveld_2022_color, ax=axs)
ac=0
for i in range(0, len(intveld_2022_labels)):
    axs.text(x=-0.15, y=ac+intveld_2022.iloc[i]-0.9*intveld_2022.iloc[i], s=intveld_2022_labels[i])
    ac=ac+intveld_2022.iloc[i]
#%% In't veld_2021
intveld_2021=pd.DataFrame({'compos':[25,14,11,9,5,5,4,2]})
intveld_2021=intveld_2021/intveld_2021.sum()
intveld_2021_labels = ['SOA', 'Secondary \nsulphate', 'Non-exhaust \nroad traffic', 'Sea spray', 'Mineral', 'Secondary \nnitrate', 'Heavy oil', 'Industry']
intveld_2021_color= ['green', 'red', 'darkgrey', 'deepskyblue','tan', 'blue', 'darkred', 'magenta']
#
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
intveld_2021.T.plot.bar(stacked=True, legend=False,color=intveld_2021_color, ax=axs)
ac=0
for i in range(0, len(intveld_2021_labels)):
    axs.text(x=-0.2, y=ac+intveld_2021.iloc[i]-0.9*intveld_2021.iloc[i], s=intveld_2021_labels[i])
    ac=ac+intveld_2021.iloc[i]
#%% Brines 2019
brines_2019=pd.DataFrame({'compos':[19,19,15,14,14,11,8]})
brines_2019=brines_2019/brines_2019.sum()
brines_2019_labels = ['Fresh traffic', 'Ship + SOA', 'Urban mix', 'Industrial NE +\nsea salt', 'Bio SOA', 'Biomass \nburning', 'Industrial W\n+ pinene']
brines_2019_color= ['grey', 'darkred', 'mediumpurple', 'deepskyblue', 'green', 'peru', 'magenta']
#
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
brines_2019.T.plot.bar(stacked=True, legend=False,color=brines_2019_color, ax=axs)
ac=0
for i in range(0, len(brines_2019_labels)):
    axs.text(x=-0.2, y=ac+brines_2019.iloc[i]-0.9*brines_2019.iloc[i], s=brines_2019_labels[i])
    ac=ac+brines_2019.iloc[i]
#%% Alier 2013
alier_2013=pd.DataFrame({'compos':[34,18,14,13,12,9]})
alier_2013=alier_2013/alier_2013.sum()
alier_2013_labels = ['POA Urban', 'OOA Urban', 'SOA Aged', 'SOA \nbio pin', 'SOA Iso', 'BBOA Regional']
alier_2013_color= ['grey', 'olive', 'darkgreen', 'limegreen', 'yellowgreen', 'peru']
#
fig, axs=plt.subplots(figsize=(3, 6))
# axs.set_ylim([0,1.1])
alier_2013.T.plot.bar(stacked=True, legend=False,color=alier_2013_color, ax=axs)
ac=0
for i in range(0, len(alier_2013_labels)):
    axs.text(x=-0.2, y=ac+alier_2013.iloc[i]-0.9*alier_2013.iloc[i], s=alier_2013_labels[i])
    ac=ac+alier_2013.iloc[i]
#%% Amato 2016
amato_2016=pd.DataFrame({'compos':[38, 19, 13, 11, 7, 5, 3,1 ]})
amato_2016=amato_2016/amato_2016.sum()
amato_2016_labels = ['Secondary \nsulphate\n+ organics', 'Vehicle \nexhaust', 'Secondary \nnitrate' , 'Industrial', 
                     'Mineral', 'Heavy oil', 'Aged sea salt', 'Vehicle \nnon-exhaust']
amato_2016_color= ['red', 'grey', 'blue', 'magenta', 'tan', 'darkred', 'deepskyblue', 'darkgrey', 'blue']
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
amato_2016.T.plot.bar(stacked=True, legend=False,color=amato_2016_color, ax=axs)
ac=0
for i in range(0, len(amato_2016_labels)):
    axs.text(x=-0.2, y=ac+amato_2016.iloc[i]-0.9*amato_2016.iloc[i], s=amato_2016_labels[i])
    ac=ac+amato_2016.iloc[i]
#%% Minguillon 2016
minguillon_2016=pd.DataFrame({'compos':[39,29,20,12]})
minguillon_2016=minguillon_2016/minguillon_2016.sum()
minguillon_2016_labels = ['LO-OOA', 'MO-OOA', 'COA', 'HOA', ]
minguillon_2016_color= ['lightgreen', 'darkgreen', 'mediumpurple', 'grey']
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
minguillon_2016.T.plot.bar(stacked=True, legend=False,color=minguillon_2016_color, ax=axs)
ac=0
for i in range(0, len(minguillon_2016_labels)):
    axs.text(x=-0.1, y=ac+minguillon_2016.iloc[i]-0.9*minguillon_2016.iloc[i], s=minguillon_2016_labels[i])
    ac=ac+minguillon_2016.iloc[i]
#%% Mohr 2012
mohr_2012=pd.DataFrame({'compos':[28,26,19,16,11]})
mohr_2012=mohr_2012/mohr_2012.sum()
mohr_2012_labels = ['MO-OOA', 'LO-OOA', 'COA', 'HOA', 'BBOA']
mohr_2012_color= ['darkgreen', 'lightgreen', 'mediumpurple', 'grey','peru' ]
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
mohr_2012.T.plot.bar(stacked=True, legend=False,color=mohr_2012_color, ax=axs)
ac=0
for i in range(0, len(mohr_2012_labels)):
    axs.text(x=-0.1, y=ac+mohr_2012.iloc[i]-0.9*mohr_2012.iloc[i], s=mohr_2012_labels[i])
    ac=ac+mohr_2012.iloc[i]
#%% Via 2023
via_2023=pd.DataFrame({'compos':[26,17,16,14,9,8,5,4]})
via_2023=via_2023/via_2023.sum()
via_2023_labels = ['AS + \naged SOA\n+heavy oil', 'AN + ACl', 'Aged SOA', 'Road traffic', 'Biomass \nburning', 'Fresh SOA', 'COA', 'Industry']
via_2023_color= ['darkred', 'blue', 'darkgreen', 'grey', 'peru', 'lightgreen', 'mediumpurple', 'magenta' ]
fig, axs=plt.subplots(figsize=(3, 6))
axs.set_ylim([0,1.1])
via_2023.T.plot.bar(stacked=True, legend=False,color=via_2023_color, ax=axs)
ac=0
for i in range(0, len(via_2023_labels)):
    axs.text(x=-0.1, y=ac+via_2023.iloc[i]-0.9*via_2023.iloc[i], s=via_2023_labels[i])
    ac=ac+via_2023.iloc[i]






