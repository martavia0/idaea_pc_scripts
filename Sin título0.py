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
#%%
os.chdir("C:/Users/Marta Via/Documents/1. PhD/A. BCN_Series/ACSM_PalauReial_2017_09/MVIA/ACSM_new/Org_BC_NR_Chem/Data_treatment")
#print("Tell the byte at which the file cursor is:",my_file.tell())
#%%
l=[]
l_ts=[]
l_pr=[]
with open('EXPL_var_1_10.txt') as f    :  
    lines=f.read().splitlines()
for i in range(0,len(lines)):
    if '_mean' in lines[i]:    
        l.append(i)
#%%
d=[]
len_l=len(l)
for i in range(0,len_l-1):
    d.append(lines[l[i]:l[i+1]])
d.append(lines[l[len_l-1]:])
        