# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:37:10 2021

@author: Marta Via
"""
import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
#import seaborn as sns
import matplotlib as plt
#%%  IMPORT ALL
path = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Profiles/Output"
abs_p=Treat_Profiles('ABS')
rel_p=Treat_Profiles('REL') 
abs_p.drop('class', inplace=True, axis=1, errors='ignore')
rel_p.drop('class', inplace=True, axis=1, errors='ignore')
#%% 
abs_prof_names=list(abs_p.columns)
#%%
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/orig")    
mz=pd.read_csv("mz.txt")
mzlab=pd.read_csv("mz_lab.txt", skip_blank_lines=(False))
class_pr=pd.read_csv("Class_PR.txt")
class_pr2=pd.read_csv("Class_PR2.txt", sep="\t") 
ts=pd.read_csv("utc_end_time.txt", skip_blank_lines=False, sep="\t", infer_datetime_format=True)
ts['Time']=pd.to_datetime(ts['utc_start_time'], dayfirst=True, errors='coerce')
 #%%
 """ 5 - Treat_Profiles(Combination,Prof_or_Rel):
         * Generates a df with 6 lists for each combination with the dfs or PROF or REL.
      --> Return: d, DataFRame with the lists 4-9 factors (rows) per each combination (columns)
"""
def Treat_Profiles(choose) :
    MT_1h_C10, MT_1h_C1, MT_1h_C01, MT_1h_C001, MT_1h_C0001, MT_1h_C00001 = [],[],[],[],[],[]
    MT_2h_C10, MT_2h_C1, MT_2h_C01, MT_2h_C001, MT_2h_C0001, MT_2h_C00001 = [],[],[],[],[],[]
    MT_3h_C10, MT_3h_C1, MT_3h_C01, MT_3h_C001, MT_3h_C0001, MT_3h_C00001 = [],[],[],[],[],[]
    MT_6h_C10, MT_6h_C1, MT_6h_C01, MT_6h_C001, MT_6h_C0001, MT_6h_C00001 = [],[],[],[],[],[]
    MT_12h_C10, MT_12h_C1, MT_12h_C01, MT_12h_C001, MT_12h_C0001, MT_12h_C00001 = [],[],[],[],[],[]
    MT_24h_C10, MT_24h_C1, MT_24h_C01, MT_24h_C001, MT_24h_C0001, MT_24h_C00001 = [],[],[],[],[],[]
    MT_30m_C10, MT_30m_C1, MT_30m_C01, MT_30m_C001, MT_30m_C0001, MT_30m_C00001 = [],[],[],[],[],[]
    if choose =='ABS':
        n=0
    if choose =='REL':  
        n=2
    for i in range(4,10,1):
        MT_1h_C10.append(Profile_Clustering(i,'1h_C10')[n]) #delete the [n] for also obtaining the std
        MT_1h_C1.append(Profile_Clustering(i,'1h_C1')[n]) #delete the [n] for also obtaining the std
        MT_1h_C01.append(Profile_Clustering(i,'1h_C01')[n]) #delete the [n] for also obtaining the std
        MT_1h_C001.append(Profile_Clustering(i,'1h_C001')[n]) #delete the [n] for also obtaining the std
        MT_1h_C0001.append(Profile_Clustering(i,'1h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_1h_C00001.append(Profile_Clustering(i,'1h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_2h_C10.append(Profile_Clustering(i,'2h_C10')[n]) #delete the [n] for also obtaining the std
        MT_2h_C1.append(Profile_Clustering(i,'2h_C1')[n]) #delete the [n] for also obtaining the std
        MT_2h_C01.append(Profile_Clustering(i,'2h_C01')[n]) #delete the [n] for also obtaining the std
        MT_2h_C001.append(Profile_Clustering(i,'2h_C001')[n]) #delete the [n] for also obtaining the std
        MT_2h_C0001.append(Profile_Clustering(i,'2h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_2h_C00001.append(Profile_Clustering(i,'2h_C00001')[n]) #delete the [n] for also obtaining the st
        MT_3h_C10.append(Profile_Clustering(i,'3h_C10')[n]) #delete the [n] for also obtaining the std
        MT_3h_C1.append(Profile_Clustering(i,'3h_C1')[n]) #delete the [n] for also obtaining the std
        MT_3h_C01.append(Profile_Clustering(i,'3h_C01')[n]) #delete the [n] for also obtaining the st
        MT_3h_C001.append(Profile_Clustering(i,'3h_C001')[n]) #delete the [n] for also obtaining the std
        MT_3h_C0001.append(Profile_Clustering(i,'3h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_3h_C00001.append(Profile_Clustering(i,'3h_C00001')[n]) #delete the [n] for also obtaining the st
        MT_6h_C10.append(Profile_Clustering(i,'6h_C10')[n]) #delete the [n] for also obtaining the std
        MT_6h_C1.append(Profile_Clustering(i,'6h_C1')[n]) #delete the [n] for also obtaining the std
        MT_6h_C01.append(Profile_Clustering(i,'6h_C01')[n]) #delete the [n] for also obtaining the st
        MT_6h_C001.append(Profile_Clustering(i,'6h_C001')[n]) #delete the [n] for also obtaining the std
        MT_6h_C0001.append(Profile_Clustering(i,'6h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_6h_C00001.append(Profile_Clustering(i,'6h_C00001')[n]) #delete the [n] for also obtaining the st
        MT_12h_C10.append(Profile_Clustering(i,'12h_C10')[n]) #delete the [n] for also obtaining the std
        MT_12h_C1.append(Profile_Clustering(i,'12h_C1')[n]) #delete the [n] for also obtaining the std
        MT_12h_C01.append(Profile_Clustering(i,'12h_C01')[n]) #delete the [n] for also obtaining the st
        MT_12h_C001.append(Profile_Clustering(i,'12h_C001')[n]) #delete the [n] for also obtaining the std
        MT_12h_C0001.append(Profile_Clustering(i,'12h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_12h_C00001.append(Profile_Clustering(i,'12h_C00001')[n]) #delete the [n] for also obtaining the st
        MT_24h_C10.append(Profile_Clustering(i,'24h_C10')[n]) #delete the [n] for also obtaining the std
        MT_24h_C1.append(Profile_Clustering(i,'24h_C1')[n]) #delete the [n] for also obtaining the std
        MT_24h_C01.append(Profile_Clustering(i,'24h_C01')[n]) #delete the [n] for also obtaining the st
        MT_24h_C001.append(Profile_Clustering(i,'24h_C001')[n]) #delete the [n] for also obtaining the std
        MT_24h_C0001.append(Profile_Clustering(i,'24h_C0001')[n]) #delete the [n] for also obtaining the std
        MT_24h_C00001.append(Profile_Clustering(i,'24h_C00001')[n]) #delete the [n] for also obtaining the st
        MT_30m_C10.append(Profile_Clustering(i,'30m_C10')[n]) #delete the [n] for also obtaining the std
        MT_30m_C1.append(Profile_Clustering(i,'30m_C1')[n]) #delete the [n] for also obtaining the std
        MT_30m_C01.append(Profile_Clustering(i,'30m_C01')[n]) #delete the [n] for also obtaining the st
        MT_30m_C001.append(Profile_Clustering(i,'30m_C001')[n]) #delete the [n] for also obtaining the std
        MT_30m_C0001.append(Profile_Clustering(i,'30m_C0001')[n]) #delete the [n] for also obtaining the std
        MT_30m_C00001.append(Profile_Clustering(i,'30m_C00001')[n]) #delete the [n] for also obtaining the st
    print('MT_1h_C01: ', len(MT_1h_C01),'\n','MT_3h_C00001:', len(MT_3h_C00001), '\n','MT_12h_C001',len(MT_12h_C001), '\n','MT_30m_C01', len(MT_30m_C01))
    d=pd.DataFrame({'1h_C10':MT_1h_C10,'1h_C1':MT_1h_C1,'1h_C01':MT_1h_C01,'1h_C001':MT_1h_C001,'1h_C0001':MT_1h_C0001,'1h_C00001':MT_1h_C00001,
                    '2h_C10':MT_2h_C10,'2h_C1':MT_2h_C1,'2h_C01':MT_2h_C01,'2h_C001':MT_2h_C001,'2h_C0001':MT_2h_C0001,'2h_C00001':MT_2h_C00001,
                    '3h_C10':MT_3h_C10,'3h_C1':MT_3h_C1,'3h_C01':MT_3h_C01,'3h_C001':MT_3h_C001,'3h_C0001':MT_3h_C0001,'3h_C00001':MT_3h_C00001,
                    '6h_C10':MT_6h_C10,'6h_C1':MT_6h_C1,'6h_C01':MT_6h_C01,'6h_C001':MT_6h_C001,'6h_C0001':MT_6h_C0001,'6h_C00001':MT_6h_C00001,
                    '12h_C10':MT_12h_C10,'12h_C1':MT_12h_C1,'12h_C01':MT_12h_C01,'12h_C001':MT_12h_C001,'12h_C0001':MT_12h_C0001,'12h_C00001':MT_12h_C00001,
                    '24h_C10':MT_24h_C10,'24h_C1':MT_24h_C1,'24h_C01':MT_24h_C01,'24h_C001':MT_24h_C001,'24h_C0001':MT_24h_C0001,'24h_C00001':MT_24h_C00001,
                    '30m_C10':MT_30m_C10,
                    '30m_C1':MT_30m_C1,
                    '30m_C01':MT_30m_C01,
                    '30m_C001':MT_30m_C001,
                    '30m_C0001':MT_30m_C0001,
                    '30m_C00001':MT_30m_C00001})
    d.to_csv(choose+'.txt', sep='\t')
    d['numf']=['F4','F5', 'F6', 'F7', 'F8', 'F9']
    d['class']=class_pr2
    d.set_index('numf', inplace=True)
    return d
   #%%
   MT_try=[]
for i in range(4,19,1): 
    MT_try.append(Profile_Clustering(i,'30m_C10')[0]) #delete the [0] for also obtaining the st
    #%%
""" 1 - Import_each_combination_prof(Combination,Prof_or_Rel):
         * Importa m/z, m/z_lab, class_pr, class_pr2 (guardades en una carpeta anterior)
         * Importa tots els prof en el seu estat més raw en format abs o rel segons indicació.   
         * Fa llegibles els fitxers eliminant nans.
       --> Return: macro_l, list of all dataframes with all combinations 
"""
def Import_each_combination_prof(Combination,Prof_or_Rel):
    os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/orig")    
    mz=pd.read_csv("mz.txt")
    mzlab=pd.read_csv("mz_lab.txt", skip_blank_lines=(False),keep_default_na=(True))
    class_pr=pd.read_csv("Class_PR.txt")
    class_pr2=pd.read_csv("Class_PR2.txt", sep="\t") 
    path = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Profiles/"+Combination
    os.chdir(path)
    comb=Combination
    print(comb)
    namef=os.listdir(path)
    all_files = glob.glob(path+'/'+Prof_or_Rel+'*')#Rel* or Prof*
    f=pd.DataFrame()
    c1=0
    macro_l=[]
    for filename in all_files:
        df=pd.read_csv(filename, skiprows=3,#prefix='F',
                       skipfooter=2,engine='python',sep='\t',keep_default_na=True,na_values='np.nan',skipinitialspace=True)
        df=df.fillna(-1)
        df=df.astype(float)
        list_ok=[str(i) for i in range(0,len(df.iloc[1]))]
        row0=list(df.columns[:len(df.columns)])
        row0.remove('Unnamed: 0')
        row0 = [i for i in row0]
        df2=pd.DataFrame(df.set_axis(list_ok,axis=1))
        row0=pd.DataFrame(row0).T
        row0.columns = df2.columns[1:]
        del(df2['0'])
        df2=pd.concat([row0,df2],ignore_index=True,axis=0)
        df2=df2.replace(['NAN','NAN.1','NAN.2','NAN.3','NAN.4','NAN.5','NAN.6','NAN.7','NAN.8'],np.nan)
        df2['mz']=mz
        df2['mz_lab']=mzlab
        df2['class']=class_pr2
        macro_l.append(df2)
        c1=c1+1   
    return macro_l
#%%
check_list= Import_each_combination_prof('30m_C10', 'Prof')
#%%
""" 2 - Import_each_combination_prof(Combination,Prof_or_Rel):
         * Slices the macro_l list into a list
      --> Return: li_out, list separeating 4-9 factors data of each combination.
"""
def Slicing_Num_Factors(li_in):
    li_out=[li_in[0:10],li_in[10:20], li_in[20:30], li_in[30:40], li_in[40:50], li_in[50:60]]
    return li_out
#%%
aa=Slicing_Num_Factors(check_list)
#%%
""" 3 - Profiles_by_num_facts(nf,comb)
         * Calls 1 and 2 to do the previous steps
         * Concatenates all profiles of all factors of all runs per a given combination
      --> Return: F, F2, dataframes with all factors for all runs for the inserted combination.
"""
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
def Profiles_by_num_facts(nf,comb):
    Prof=Import_each_combination_prof(comb,'Prof')
    Prof_sep=Slicing_Num_Factors(Prof)
    Rel_prof=Import_each_combination_prof(comb,'Rel') 
    Rel_prof_sep=Slicing_Num_Factors(Rel_prof)
    conc=Prof_sep[nf-4]
    conc2=Rel_prof_sep[nf-4]
    F=pd.concat(conc,axis=1)
    F2=pd.concat(conc2,axis=1)
    F.drop(['mz','mz_lab','class'], axis=1,inplace=True)
    F2.drop(['mz','mz_lab','class'], axis=1,inplace=True)
    namecols=[]
    for i in range(0,10):
        for j in range(1,1+nf):
            k='F'+str(j)+'_Run'+str(i)
            namecols.append(k)
    print(len(namecols), len(F2.columns))
    F.columns=namecols
    F2.columns=namecols
    return F,F2
#%%
prof_test, prof_rel_test=Profiles_by_num_facts(7,'30m_C10')

#%%
""" 4 - Profile_Clustering(num_f, comb)
         * Calls 3 and exports df, relf.
         * Generates dendrogram over df and averages the profiles regarding the num_f last iterations. 
         * It is applied to relf also (but we are not clustering relf, just averaging on df clusters).
         * Plot the dendrogram of the last num_f steps.
      --> Return: F, F2, dataframes with all factors for all runs for the inserted combination.
"""
def Profile_Clustering(num_f, comb):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib import pyplot as plt
    df,relf = Profiles_by_num_facts(num_f,comb)
    df=df.replace(np.nan, 0)
    relf=relf.replace(np.nan, 0)
    df=df.astype(float)
    relf=relf.astype(float)
    linked=linkage(df.T, 'single')
    Z=pd.DataFrame()
    ZR=pd.DataFrame()
    Z_std=pd.DataFrame()
    ZR_std=pd.DataFrame()
    link_copy=pd.DataFrame(linked)  
    for i in range(0,len(link_copy)):
        a=int(link_copy[0].iloc[i])-1
        b=int(link_copy[1].iloc[i])-1
        if(a>=len(link_copy) and b<len(link_copy)):
            a_prima=a-len(link_copy)
#            print(i, 'a:',a, 'a_p:',a_prima)
            df_new=pd.DataFrame({'a': Z.iloc[:,a_prima], 'b':df.iloc[:,b]})
            relf_new=pd.DataFrame({'a': ZR.iloc[:,a_prima], 'b':relf.iloc[:,b]})
        if(a<len(link_copy) and b>=len(link_copy)):
            b_prima=b-len(link_copy)
#            print(i, 'b:',b, 'b_p:',b_prima)
            df_new=pd.DataFrame({'a': df.iloc[:,a], 'b':Z.iloc[:,b_prima]})
            relf_new=pd.DataFrame({'a': relf.iloc[:,a], 'b':ZR.iloc[:,b_prima]})
        if(a>=len(link_copy) and b>=len(link_copy)):
            a_prima=a-len(link_copy)
            b_prima=b-len(link_copy)
#            print(i,'a:', a,'a_p:',a_prima,'b:',b,'b_p:',b_prima)
            df_new=pd.DataFrame({'a': Z.iloc[:,a_prima], 'b':Z.iloc[:,b_prima]})
            relf_new=pd.DataFrame({'a': ZR.iloc[:,a_prima], 'b':ZR.iloc[:,b_prima]})
        if(a<len(link_copy) and b<len(link_copy)): #CAS EASY. 
#            print(i, 'a:',a, 'b:',b)
            df_new=pd.DataFrame({'a': df.iloc[:,a], 'b':df.iloc[:,b]})
            relf_new=pd.DataFrame({'a': relf.iloc[:,a], 'b':relf.iloc[:,b]})
        Z[str(i)]=df_new.mean(axis=1)
        ZR[str(i)]=relf_new.mean(axis=1)
        Z_std[str(i)]=df_new.std(axis=1)
        ZR_std[str(i)]=relf_new.std(axis=1)
    p_clustered=pd.DataFrame(Z.iloc[:,(len(link_copy)-num_f):])
    p_clustered_std=pd.DataFrame(Z_std.iloc[:,len(link_copy)-num_f:])
    r_clustered=pd.DataFrame(ZR.iloc[:,len(link_copy)-num_f:])
    r_clustered_std=pd.DataFrame(ZR_std.iloc[:,len(link_copy)-num_f:])
    #GRAPH
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.size':22})
    dendr=dendrogram(linked, p=7, truncate_mode='lastp', distance_sort='descending',show_leaf_counts=True , show_contracted=True)
    ax=plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=15)
    #Return
    return p_clustered, p_clustered_std, r_clustered, r_clustered_std
#%%
prof_mean, prof_std, rel_mean, rel_std=Profile_Clustering(4, '24h_C00001')
    #%% 
#%%
#******************************      DATA TREATMENT       ************************++**+
#%%
""" 6 - Export_df(d, num_f)
         * Calls 3 and exports df, relf.
         * Generates dendrogram over df and averages the profiles regarding the num_f last iterations. 
         * It is applied to relf also (but we are not clustering relf, just averaging on df clusters).
         * Plot the dendrogram of the last num_f steps.
      --> Return: F, F2, dataframes with all factors for all runs for the inserted combination.
"""
def Export_df(d, num_fact):
    os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Profiles/Output/")    
    d_nf=d.iloc[num_fact-4]
    df=pd.DataFrame()
    names=[]
    for i in range(0,len(d_nf)):
        for j in d_nf[i]:
            print(j)
            names.append(d_nf.index[i] + '_F')#+str(int(j)+1))
            df=pd.concat([df,d_nf[i][j]], axis=1)       
            df=df.astype(float)
            df.columns=names
            df.to_csv(str(num_fact)+'F.txt', sep='\t')
    return df
#%%
#%%
for i in range(4,10,1):
    abs_profiles=Export_df(abs_p,i)
    rel_profiles=Export_df(rel_p,i)
#%%
for i in range(0,42):
    print(abs_prof[i].columns)
#    abs_prof[i].drop(labels='mzOA', inplace=True, axis=1)
    abs_prof[i].drop(labels='class', inplace=True, axis=1)

#%%
prof_names=abs_prof_names
#%%
import matplotlib.pyplot as plt
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Profiles/Plots/8F")    
numf=4
abs_prof=abs_p.iloc[numf].to_list()
rel_prof=rel_p.iloc[numf].to_list()
li_names=[['0','1','2','3'],['0','1','2','3','4'],['0','1','2','3','4','5'],['0','1','2','3','4','5','6'], 
          ['0','1','2','3','4','5','6','7'],['0','1','2','3', '4','5','6','7','8']]
for i in range(0,len(abs_prof)):
    abs_prof[i].columns=li_names[numf]
    rel_prof[i].columns=li_names[numf]
    #%%
for i in range(0,len(abs_profiles)):
    print(i,abs_prof_names[i])
    #Profiles_OARP(abs_prof[i][0:116],rel_prof[i][0:116],abs_prof_names[i])
    Profiles_NRBCFRP(abs_prof[i][0:116],rel_prof[i][0:116],abs_prof_names[i])
#%%
Profiles_OARP(abs_prof[2][0:116],rel_prof[2][0:116],abs_prof_names[i])

#%%
def Profiles_OARP(df_O,df_RP,name):
    df_O['mzOA']=mzlab['mz_lab']
    df_O['class']=class_pr2['Class_PR']
    df_O=df_O.rename(columns={'0':'F1','1':'F2','2':'F3','3':'F4','4':'F5','5':'F6','6':'F7','7':'F8'})#,'8':'F9'})
    df_RP=df_RP.rename(columns={'0':'RF1','1':'RF2','2':'RF3','3':'RF4','4':'RF5','5':'RF6','6':'RF7','7':'RF8'})#,'8':'RF9'})
    mask =(class_pr2['Class_PR']=="ACSM")
    df_O_m=df_O[mask]
    df_RP_m=df_RP[mask]
    df2=pd.DataFrame()
    df2=pd.concat([df_O_m, df_RP_m], axis=1)
    df2=df2.dropna(axis=1, how='all')
    num=((len(df2.columns)-3)//2)+1
    fig,axes=plt.subplots(nrows=num,ncols=1, figsize=(30,26))
    fig.canvas.set_window_title('MT_OA_NR_BCs_F')
    count=1
    for c in range(num):
        name1="F"+str(count)
        name2="RF"+str(count)
        axes[c].bar(df2['mzOA'], df2[name1])
        axes[c].set_ylim([0,0.05])
        axes[c].tick_params(labelrotation=90)
        ax2=axes[c].twinx()
        ax2.plot(df2['mzOA'], df2[name2], marker='o', linewidth=False,color='black')
        axes[c].grid(axis='x')
        axes[c].set_axisbelow(True)
        axes[c].set_title(name1)
        count=count+1
    plotname_PRO =name+"_PR_OA_REL.png"
    plt.savefig(plotname_PRO)
    #%%
#··········· PROFILES NRBCF only ·············    
def Profiles_NRBCFRP(df_O,df_RP,name):
    df_O['mzOA']=mzlab['mz_lab']
    df_O['class']=class_pr2['Class_PR']
    df_RP['class']=class_pr2['Class_PR']
    mask =(class_pr2['Class_PR'] !="ACSM")
    df_O=df_O.rename(columns={'0':'F1','1':'F2','2':'F3','3':'F4','4':'F5','5':'F6','6':'F7','7':'F8'})#,'8':'F9'})
    df_RP=df_RP.rename(columns={'0':'RF1','1':'RF2','2':'RF3','3':'RF4','4':'RF5','5':'RF6','6':'RF7','7':'RF8'})#,'8':'RF9'})
    df_O=df_O.dropna(axis=1, how='all')
    df_RP=df_RP.dropna(axis=1, how='all')
    df_O_m=df_O[mask]
    df_RP_m=df_RP[mask]
    df3=pd.DataFrame()
    df3=pd.concat([df_O_m, df_RP_m], axis=1)
    num2=((len(df3.columns)-3)//2)
    fig,axes=plt.subplots(nrows=num2,ncols=1, figsize=(28,26))
    fig.canvas.set_window_title('MT_NR_BCs_F')
    count=1
    for c in range(num2):
        name1="F"+str(count)
        name2="RF"+str(count)
        axes[c].bar(df3["mzOA"], df3[name1])
        axes[c].set_yscale("log")
        ax2=axes[c].twinx()
        ax2.plot(df3['mzOA'], df3[name2], marker='o', linewidth=False,color='black')
        axes[c].grid(axis='x')
        axes[c].set_axisbelow(True)
        axes[c].set_title(name1)
        count=count+1
    plotname_PRO = name+ "_PR_NRBCF_REL.png"
    plt.savefig(plotname_PRO)
        #%%
#%%
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Profiles/Plots/8F")    
factors=pd.read_csv("Factors.txt", sep="\t") #Something is wrong as the imported file does not resemble the original one,. 
names=factors['run']
for i in range(0,len(abs_prof)):
    abs_prof[i].columns = [''] * 42
    d=[factors.iloc[i,1],factors.iloc[i,2],factors.iloc[i,3],factors.iloc[i,4]]
    print(i, d)
    abs_prof[i].columns=d
#%%
#df_abs_prof=abs_prof
#cat_names=['HOA', 'BBOA', 'LO-OOA', 'MO-OOA']#,'LO-OOA + BBOA', 'MO-OOA + BBOA']
#num_cat = 4
def Assembly_factors(df_abs_prof,num_cat, cat_names):
    print(cat_names)
    df_factors=[]
    all_factors=[]
    for j in range(0,num_cat):
        factor=pd.DataFrame()
        for i in range(0,len(df_abs_prof)):
            for k in range(0,num_cat):
                if df_abs_prof[i].columns[k]==cat_names[j]:
                    factor[abs_prof_names[i]]=(df_abs_prof[i][cat_names[j]])
        all_factors.append(pd.DataFrame(factor))
        all_factors[j].to_csv(cat_names[j]+'.txt', sep='\t')
    return all_factors, cat_names;
#%%
df_factors, cat_names=Assembly_factors(abs_prof, 4, ['HOA', 'BBOA', 'LO-OOA', 'MO-OOA'])
#%%
def Metrics_calculations(df_factors, num_cat, cat_names):
    R_squared, R_squared_log, Ind_agree, ID_sum, Cos_simil = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(0,num_cat):
        factor=pd.DataFrame(df_factors[i])
        factor_mean=pd.Series(factor.mean(axis=1))
        li_r2, li_r2log, li_iof, li_ID, li_cos= [],[],[],[],[]
        for j in range(0,len(factor.columns)):
            li_r2.append([factor.columns[j], R2(factor_mean, factor.iloc[:,j])])
            li_r2log.append([factor.columns[j], R2_log(factor_mean, factor.iloc[:,j])])
            li_iof.append([factor.columns[j], Index_of_agreement(factor_mean, factor.iloc[:,j])])
            li_ID.append([factor.columns[j], ID(factor_mean, factor.iloc[:,j])])
            li_cos.append([factor.columns[j], ID(factor_mean, factor.iloc[:,j])])
        print(li_r2)
        li=pd.DataFrame(li_r2)
        print('ind_'+cat_names[i])
        R_squared['ind_'+cat_names[i]]=li[0]
        R_squared[cat_names[i]]=(li[1])
        R_squared.to_csv('R2.txt', sep='\t')
        li=pd.DataFrame(li_r2log)
        R_squared_log['ind_'+cat_names[i]]=li[0]
        R_squared_log[cat_names[i]]=(li[1])
        R_squared_log.to_csv('R2_log.txt', sep='\t')
        li=pd.DataFrame(li_iof)
        Ind_agree['ind_'+cat_names[i]]=li[0]
        Ind_agree[cat_names[i]]=(li[1])
        Ind_agree.to_csv('Ind_agree.txt', sep='\t')
        li=pd.DataFrame(li_ID)
        ID_sum['ind_'+cat_names[i]]=li[0]
        ID_sum[cat_names[i]]=(li[1])
        ID_sum.to_csv('ID.txt', sep='\t')
        li=pd.DataFrame(li_cos)
        Cos_simil['ind_'+cat_names[i]]=li[0]
        Cos_simil[cat_names[i]]=(li[1])
        Cos_simil.to_csv('Cos_Sim.txt', sep='\t')
    return
    #%%   
Metrics_calculations(df_factors,4,['HOA', 'BBOA', 'LO-OOA', 'MO-OOA'])   

#%%
def ID(a,b):
    c=pd.DataFrame({"a":a, "b":b})
    ID=sum(abs(a.iloc[i]-b.iloc[i])/(2**0.5) for i in range(0,len(a)))
#    MAD_pairs=sum(abs(a.iloc[i]-b.iloc[i])/2**0.5 for i in range(0,len(a)))
#    MADsou= MAD_pairs # contribution of the source profile uncertainty. 95th oercentile of the MAD of all the oairs of soyrce profiles in every source cattegory, 
#    MADana =0.010768479# Average analytical uncertainty of the input dataset
#    MADsec=0.94 #Additional uncertainty attributed to the secondary sources to account for the assumptions made
#    in the calculation of their SCE and the difficulty to estimate their uncertainty from the analytical data
#    MADrel=(MADsou**2+MADana**2+MADsec**2)**0.5
    #SID = ID/(len(a))#*MADrel)
    return ID
#%%
def Index_of_agreement(a,b):
    r=np.sqrt(R2(a,b))
    c=pd.DataFrame({'a':a, 'b':b})
    sigma_a = c['a'].std()
    sigma_b = c['b'].std()
    mean_a = c['a'].mean()
    mean_b =c['b'].mean()
    l_iof=2.0*r/(sigma_a/sigma_b + sigma_b/sigma_a +((mean_a - mean_b)**2/(sigma_a*sigma_b)))
    return l_iof
#%%
def Z_score(df):    
    mean=df.mean()
    std=df.std()
    Z=(df-mean)/std
    return Z
#%%

#%%
print(Index_of_agreement(LOOOA['1h_C1'], LOOOA['1h_C10']))  
    
#%%
def R2_log(a,b):
    c=pd.DataFrame({"a":-1/np.log(a), "b":-1/np.log(b)})
    cm=c.corr(method='pearson')
    r=cm.iloc[0,1]
    return r**2
#%%
def R2(a,b):
    c=pd.DataFrame({"a":a, "b":b})
    cm=c.corr(method='pearson')
    r=cm.iloc[0,1]
    return r**2
#%%
print(R2(HOA_mean, HOA['1h_C1']))
#%%
def cos_sim(a,b):
    c=pd.DataFrame({"a":a, "b":b})
    ac_top=0.0
    ac_a=0.0
    ac_b=0.0
    for i in range(0,len(a)):
        if (c.a.iloc[i]!='NAN') and (c.b.iloc[i]!='NAN'):
            ac_top=ac_top+(float(c.a.iloc[i])*float(c.b.iloc[i]))
            ac_a=ac_a+float(c.a.iloc[i])**2
            ac_b=ac_b+float(c.b.iloc[i])**2
    cosine=ac_top/(ac_a*ac_b)  
    return cosine

#%%
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
print(df.sum(axis=1))
df=df.replace(np.nan,0)
plt.figure(figsize=(15, 7))
#print(df.T[:30])
linked_combinations= linkage(df.T[:30], 'single')     
plt.figure(figsize=(10, 7))
a=dendrogram(linked_combinations, p=4, orientation='top',distance_sort='descending',show_leaf_counts=True)   

#%%
HOA=pd.DataFrame()
BBOA=pd.DataFrame()
MOOOA=pd.DataFrame()
LOOOA=pd.DataFrame()
for i in range(0,len(abs_prof)):
    for j in range(0,4):
        if abs_prof[i].columns[j]=='HOA':
            HOA[abs_prof_names[i]]=(abs_prof[i]['HOA'])
        if abs_prof[i].columns[j] =='BBOA':
            BBOA[names[i]]=(abs_prof[i]['BBOA'])     
        if abs_prof[i].columns[j] =='LO-OOA':
            LOOOA[names[i]]=(abs_prof[i]['LO-OOA']) 
        if abs_prof[i].columns[j] =='MO-OOA':
            MOOOA[names[i]]=(abs_prof[i]['MO-OOA'])     
        if abs_prof[i].columns[j] =='LO-OOA + BBOA':
            LOOOA[names[i]]=(abs_prof[i]['LO-OOA + BBOA']) 
            BBOA[names[i]]=(abs_prof[i]['LO-OOA + BBOA']) 
        if abs_prof[i].columns[j] =='MO-OOA + BBOA':
            MOOOA[names[i]]=(abs_prof[i]['MO-OOA + BBOA'])  
            BBOA[names[i]]=(abs_prof[i]['MO-OOA + BBOA'])                                  
HOA.to_csv('HOA.txt', sep='\t')
BBOA.to_csv('BBOA.txt', sep='\t')
LOOOA.to_csv('LO-OOA.txt', sep='\t')
MOOOA.to_csv('MO-OOA.txt', sep='\t')
   #%%li=[str(i) for i in range(0,10*num_f)]
    df0.columns=(li)
    relf0.columns=(li)
    F_med=pd.DataFrame()
    F_med_rel=pd.DataFrame()
    F_std=pd.DataFrame()
    F_std_rel=pd.DataFrame()
    columnes=[str(j) for j in range(0,num_f)]
    for i in range(0,len(columnes)):
       # F_med[str(columnes[i])]=df0[a['ivl'][i*10:(i+1)*10]].mean(axis=1) 
       # F_med_rel[str(columnes[i])]=(relf0[a['ivl'][i*10:(i+1)*10]].mean(axis=1)) 
     #   F_std[str(columnes[i])]=df0[a['ivl'][i*10:(i+1)*10]].std(axis=1)
      #  F_std_rel[str(columnes[i])]=relf0[a['ivl'][i*10:(i+1)*10]].std(axis=1)
    return F_med, F_std, F_med_rel, F_std_rel, linked, dendr, codebook, whitened,df
#%%.
med_test,std_test,med_rel_test, std_rel_test, link,dendr,codebook, whitened,df= Profile_Clustering(7,'24h_C00001')
#%%

