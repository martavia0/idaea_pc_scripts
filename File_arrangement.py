# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:59:38 2022

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

class File_arrangements:
    
    def __init__(self,x):
        self.x = 0
        pass
    
    def Hello(self):
        print("Hello")
            
    def Runs_check(self, output, path, nb_runs):
    #This function checks all the runs files and if some file is not labelled with the zero before the number (run_1 instead of run_01) 
    #it saves the proper version including the zero. It does so for runs up to 100. This will put all files in the correct order. 
    #Output: Can be 'Profile_run', 'Res_run', 'REL_prof_run', etc. 
    #Path: path where these changes are to be made
    #Nb_runs: number of runs in the folder
        os.chdir(path)
        f_EVP= glob.glob(path+output+'*')
        f_EVP.sort()
        run_nb=[int(i[-7:-4]) for i in f_EVP]
        li=list(range(0, nb_runs))
        for j in li:
            if j in run_nb:
                print('y')
            else:
                print('no_'+str(j))
                b=pd.DataFrame(np.nan, index=[0, 1, 2, 3], columns=[output])
                if j<10:
                    b.to_csv(output+'_00'+str(j)+'.txt', sep='\t')
                if j>=10 and j<100:
                    b.to_csv(output+'_0'+str(j)+'.txt', sep='\t') 
                else:
                    b.to_csv(output+'_'+str(j)+'.txt', sep='\t')
      #path="C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/MTR-PMF/Traditional/Runs/"
      #Runs_check('Profile_run')

    def Date_Selecter(self, df1, df2,  namecol_df)
        #df1 is the file in which the times shown in df2 want to be kept (and those which are not there, removed)
        df1['dt']=pd.to_datetime(df[name_dt_column], dayfirst=True).dt.floor('Min')
        df2['dt']=pd.to_datetime(t['t_base'], dayfirst=True).dt.floor('Min')#round(freq='min') #
        df1_out=pd.DataFrame()
        df1_out['dt']= df2['dt']
        df1_out=df1[df1['dt'].isin(df2.dt)]
        df1_out.drop('dt', axis=1, inplace=True)
        return df1_out
#%%



