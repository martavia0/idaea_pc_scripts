# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:21:53 2022

@author: Marta Via
"""

all_files = glob.glob(path+'TS_run_*') #we import all files which start by Res in the basecase folder (10 runs)
for filename in all_files:
    a=filename[-7]
    b=filename[-6:-4]
    print(a,b)
    if a=='_':
        c=pd.read_csv(filename, sep='\t')
        c.to_csv('TS_run_0'+b+'.txt', sep='\t')