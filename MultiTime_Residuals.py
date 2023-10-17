# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:03:14 2021

@author: Marta Via
"""
import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
#import seaborn as sns
import matplotlib.pyplot as plt
#
#%% Functions definition
#%% IMPORT RESIDUALS
# Function to import all the combinations.
macro_list=[]
def Import_each_combination_res(Combination):
    path = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Residuals/"+Combination+'/' 
    os.chdir(path) #changing the path for each combination
    namef=os.listdir(path)
    all_files = glob.glob(path+'Res*') #we import all files which start by Res in the combination folder (60 runs --> 60 files)
    print(Combination)
    df2=pd.DataFrame()
    sc_res_combination=pd.DataFrame()
    for filename in all_files: #We import all files for the given combination
        df=pd.read_csv(filename, skiprows=1,engine='python',sep='\t',keep_default_na=True,na_values='np.nan',skipinitialspace=True)
        df=df.astype(float)
        df.columns=['Res', 'Abs', 'Sc_Res', 'Abs_Sc_Res', 'Q']
        sc_res_combination[filename[-10:-4]]=df['Sc_Res']
        df2=pd.concat([df2, sc_res_combination],axis=1)
    macro_list.append(sc_res_combination)    
    return macro_list #we return a list of lists for each combination
#%%  We import m/z and classes
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_20210202/orig")    
mz=pd.read_csv("mz.txt")
mzlab=pd.read_csv("mz_lab.txt", skip_blank_lines=(False),keep_default_na=(True))
class_pr=pd.read_csv("Class_PR.txt")
class_pr2=pd.read_csv("Class_PR2.txt", sep="\t") 
#We set combination names
combinations_names=['1h_C_1_01','1h_C_1_1','1h_C_1_2','1h_C_1_10','1h_C_01_1','1h_C_2_1','1h_C_10_1',
              '2h_C_1_01','2h_C_1_1','2h_C_1_2','2h_C_1_10','2h_C_01_1','2h_C_2_1','2h_C_10_1',
              '3h_C_1_01','3h_C_1_1','3h_C_1_2','3h_C_1_10','3h_C_01_1','3h_C_2_1','3h_C_10_1',
              '6h_C_1_01','6h_C_1_1','6h_C_1_2','6h_C_1_10','6h_C_01_1','6h_C_2_1','6h_C_10_1',
              '12h_C_1_01','12h_C_1_1','12h_C_1_2','12h_C_1_10','12h_C_01_1','12h_C_2_1','12h_C_10_1',
              '24h_C_1_01','24h_C_1_1','24h_C_1_2','24h_C_1_10','24h_C_01_1','24h_C_2_1','24h_C_10_1',
              '24hF_C_1_01','24hF_C_1_1','24hF_C_1_2','24hF_C_1_10','24hF_C_01_1','24hF_C_2_1','24hF_C_10_1',
              '30m_C_1_01','30m_C_1_1','30m_C_1_2','30m_C_1_10','30m_C_01_1','30m_C_2_1','30m_C_10_1'] 
#%%%We import all combinations given by the names defined in previous cell.
sc_res=[]
# os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022")    
for i in combinations_names: #We iterate for each combination
    sc_res=Import_each_combination_res(i) #scres is a list of 42 combination lists

#%% IMPORT TIMES
#WE import each time for every combination
path_t = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Residuals/Time/"
os.chdir(path_t)
all_files_t = glob.glob(path_t+ '*.txt')
macro_t=[]
resolutions=['1h','2h','3h','6h','12h','24h', '24hF','30m']
time_R=pd.DataFrame()
for file_t in all_files_t:
    cols=str(file_t[-7:-4])
    df2=pd.read_csv(file_t, infer_datetime_format=True,header=0)
    macro_t.append(df2)
#%% We reorder the list of times as originally it starts at 1h instead of 30'
nb_LR_points=int(82)
order_r=[7,0,1,2,3,4,5,6]
order_c=[6,0,1,2,3,4,5]
order_comb= [x for x in range(49,57)]+[x for x in range(0,48)]
resolutions=[resolutions[i] for i in order_r]
macro_t=[macro_t[i] for i in order_c]
sc_res_df=pd.DataFrame(sc_res)
sc_res_df=sc_res_df.T
li_col=[]
sc_res_df.columns=[j for j in order_comb]
# sc_res=[sc_res[i] for i in order_comb]
#%% To make some plots 
HR=pd.DataFrame(HR_mean)
HR= HR.T
HR.iloc[:,35:42].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(10,5))
#%% LR analysis
LR=pd.DataFrame(LR_mean)
LR=LR.T
LR.iloc[:,49:56].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(10,5))
#%%
#     for i in range(0, 48,7):
fig, ax=plt.subplots(1,2, figsize=(10,5), sharey=True)
HR.iloc[:,49:56].boxplot(showfliers=False, showmeans=True, rot=90, ax=ax[0])
ax[0].set_title('HR')
LR.iloc[:,48:56].boxplot(showfliers=False, showmeans=True, rot=90,ax=ax[1])
ax[1].set_title('LR')
#%%
HRLR=pd.DataFrame()
for i in HR.columns:
    HRLR=pd.concat([HRLR, HR[i], LR[i]], axis=1)
    #%%
fig, ax=plt.subplots(2,4, figsize=(18,12))    
HRLR.iloc[:,0:14].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[0,0])
HRLR.iloc[:,14:28].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[0,1])    
HRLR.iloc[:,28:42].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[0,2])    
HRLR.iloc[:,42:56].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[0,3])
HRLR.iloc[:,56:70].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[1,0])    
HRLR.iloc[:,70:84].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[1,1])    
HRLR.iloc[:,84:98].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[1,2])  
HRLR.iloc[:,98:112].boxplot(showfliers=False, showmeans=True, rot=90, figsize=(15,5),ax=ax[1,3]) 
#%%
all_plot=pd.DataFrame()
all_plot['HR']=HR.mean(axis=1)
all_plot['LR']=LR.mean(axis=1)
all_plot.boxplot(showmeans=True, showfliers=False)
#%% Re
#%% We take the mean of combinations sc res. 
df_mean=pd.DataFrame()
for df in sc_res:
    df_mean=pd.concat([df_mean, df.mean(axis=1)], axis=1)
df_mean.columns=combinations_names
#%%
HR_mean, LR_mean=[],[]
for i in df_mean.columns:
   HR_mean.append(df_mean[i].dropna()[:-56])
   LR_mean.append(df_mean[i].dropna()[-56:])
#%%
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_2022/Residuals/Plots/")
for i in range(21,len(HR_mean)):
    fig, axs=plt.subplots(figsize=(5,5))
    HR_mean[i].plot.kde(color='green')
    LR_mean[i].plot.kde(color='orange')
    # HR_mean[i].plot.hist(bins=200, alpha=0.5, color='green')
    # LR_mean[i].plot.hist(bins=200, alpha=0.5, color='orange')
    # plt.xlim(-10,10)
    fig.suptitle(df_mean.columns[i])
    plotname_PRO =df_mean.columns[i]+"_Res_nozoom.png"
    plt.savefig(plotname_PRO)
#%%
HR,LR=[],[]
for i in range(0, len(sc_res), 8): #for each resolution
    a=(sc_res[i:i+8]) #This is the whole compendium of C lists for a given resolution
    for j in range(0,len(a)): #for each C value
        HR.append(a[j].iloc[:-nb_LR_points])
        LR.append(a[j].iloc[-nb_LR_points:])
HR=HR[:56]
LR=LR[:56]
#%%
#This function separates regarding number of factors the introduced list. 
def Slicing_Num_Factors(li_in): 
    li_out=[li_in.iloc[:,0:10],li_in.iloc[:,10:20], li_in.iloc[:,20:30], 
            li_in.iloc[:,30:40], li_in.iloc[:,40:50], li_in.iloc[:,50:60]]
    return li_out
#%%
HR_dataset, LR_dataset=[],[]
HR_dataset=[Slicing_Num_Factors(i) for i in HR]  
LR_dataset=[Slicing_Num_Factors(i) for i in LR] 
#Datasets are 3D datasets with 43 combinations x different nb of factors x 10 runs each.
#%%
import numpy as np
import matplotlib.pyplot as plt
intersect_k=[]
for k in range(0,56): #nb of combinations
    intersect_j=[]
    print(k)
    for j in range(0,6): #nb of factors
        intersect_i=[]
        for i in range(0,10):#nb of runs per factor
            a=HR_dataset[k][j].iloc[:,i]
            b=LR_dataset[k][j].iloc[:,i]
            # print(a,b)
            values_a, bins_a, patches_a = plt.hist(a[np.isfinite(a)], bins=100, density=True)
            values_b, bins_b, patches_b = plt.hist(b[np.isfinite(b)], bins=100, density=True)
            intersect_i.append(histogram_intersection(values_a,values_b)) # we append the histogram intersect
#           print(histogram_intersection(values_a,values_b),np.sum(np.minimum(a,b)) )
        intersect_j.append(intersect_i)
    intersect_k.append(intersect_j)    
intersection=pd.DataFrame(intersect_k, index=combinations_names, columns=['F4','F5','F6','F7','F8','F9'])
#Here is the intersection between all 
#%%
a.hist(bins=100)
b.hist(bins=100, legend=True)
factors=['F4', 'F5', 'F6', 'F7', 'F8', 'F9']   #labels of nb of factors
#%% Nb. of factors selection.
list_HR, list_LR =[],[]
for k in range(0,42):
#    for j in range(0,6):
    list_HR.append(HR_dataset[k][2].mean(axis=1)) #here we are only selecting the nb. fact=6
    list_LR.append(LR_dataset[k][2].mean(axis=1))
 #%% F9or a given R, boxplot oF9 all Cs
R1=pd.DataFrame()
R1['C2=10e-4']=pd.concat([pd.Series(intersection['F9'][0]),pd.Series(intersection['F9'][7]),pd.Series(intersection['F9'][12]),
                        pd.Series(intersection['F9'][18]),pd.Series(intersection['F9'][24]),pd.Series(intersection['F9'][36])])
nR=1
R1['C2=10e-3']=pd.concat([pd.Series(intersection['F9'][nR]),pd.Series(intersection['F9'][nR*2+6]),pd.Series(intersection['F9'][nR*3+6]),
                        pd.Series(intersection['F9'][nR*4+6]),pd.Series(intersection['F9'][nR*5+6]),pd.Series(intersection['F9'][nR*6+6])])
nR=2
R1['C2=C10e-2']=pd.concat([pd.Series(intersection['F9'][nR]),pd.Series(intersection['F9'][nR*2+6]),pd.Series(intersection['F9'][nR*3+6]),
                        pd.Series(intersection['F9'][nR*4+6]),pd.Series(intersection['F9'][nR*5+6]),pd.Series(intersection['F9'][nR*6+6])])
nR=3
R1['C2=10e-1']=pd.concat([pd.Series(intersection['F9'][nR]),pd.Series(intersection['F9'][nR*2+6]),pd.Series(intersection['F9'][nR*3+6]),
                        pd.Series(intersection['F9'][nR*4+6]),pd.Series(intersection['F9'][nR*5+6]),pd.Series(intersection['F9'][nR*6+6])])
nR=4
R1['C2=10e0']=pd.concat([pd.Series(intersection['F9'][nR]),pd.Series(intersection['F9'][nR*2+6]),pd.Series(intersection['F9'][nR*3+6]),
                        pd.Series(intersection['F9'][nR*4+6]),pd.Series(intersection['F9'][nR*5+6]),pd.Series(intersection['F9'][nR*6+6])])
nR=5
R1['C2=10e1']=pd.concat([pd.Series(intersection['F9'][nR]),pd.Series(intersection['F9'][nR*2+6]),pd.Series(intersection['F9'][nR*3+6]),
                        pd.Series(intersection['F9'][nR*4+6]),pd.Series(intersection['F9'][nR*5+6]),pd.Series(intersection['F9'][nR*6+6])])
#%%
import matplotlib.patches as mpatches
bp=dict( linewidth=0.6, color='black')
mp=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
mdp=dict( linewidth=0.6, color='black')
wp=dict( linewidth=0.6, color='black')
fig,ax=plt.subplots(figsize=(6,6))
R1.boxplot(showfliers=False, showmeans=True, ax=ax,meanprops=mp, medianprops=mdp, whiskerprops=wp,boxprops=bp)
ax.set_xlabel('$C_2$ values',fontsize=14)
ax.set_title('Scaled Residuals'+'\n'+'nF=9',fontsize=14)
ax.set_ylabel('Intersection between HR-LR histograms', fontsize=12)
ax.set_xticklabels(['$10^-4$','$10^-3$','$10^-2$','$10^-1$','$10^0$','$10^1$' ],fontsize=12)
#%%
C2=pd.DataFrame()
C2['30 min']=pd.concat([pd.Series(intersection['F4'][0]),pd.Series(intersection['F4'][1]),pd.Series(intersection['F4'][2]),
                        pd.Series(intersection['F4'][3]),pd.Series(intersection['F4'][4]),pd.Series(intersection['F4'][5])])
nR=6
C2['1h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                        pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
nR=12
C2['2h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                        pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
nR=18
C2['3h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                        pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
nR=24
C2['6h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                    pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
nR=30
C2['12h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                        pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
nR=36
C2['24h']=pd.concat([pd.Series(intersection['F4'][nR]),pd.Series(intersection['F4'][nR+1]),pd.Series(intersection['F4'][nR+2]),
                        pd.Series(intersection['F4'][nR+3]),pd.Series(intersection['F4'][nR+4]),pd.Series(intersection['F4'][nR+5])])
#%%
fig,ax=plt.subplots()
C2.boxplot(showfliers=False, showmeans=True, ax=ax,meanprops=mp, medianprops=mdp, whiskerprops=wp,boxprops=bp)
ax.set_xlabel('$R_1$ values',fontsize=14)
ax.set_title('Scaled Residuals'+'\n'+'nF=4',fontsize=14)
ax.set_ylabel('Intersection between HR-LR histograms', fontsize=12)
# ax.set_xticklabels(['$10^-4$','$10^-3$','$10^-2$','$10^-1$','$10^0$','$10^1$' ],fontsize=12)




#%% Plot the dual histogram plot for a given resolution
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/MT_C/Residuals/plots/24h")
#import seaborn as sns
#print(type(list_HR[5]))
for k in range(36,42):
    fig, ax=plt.subplots()
    ax_hr=pd.Series(list_HR[k]).plot.kde(title='Scaled Residuals '+combinations_names[k])
    ax_lr=pd.Series(list_LR[k]).plot.kde(grid=True)
 #   ax.set_xlim(-800,100)
    ax.legend(['HR', 'LR'])
    ax.figure.savefig('Scaled Residuals '+combinations_names[k])
#%% By C
fig, ax=plt.subplots()
nC=5 #0:e-4, 1:e-3 2:e-2, 3:e-1, 4:e0, 5:e1
list_HR2=[list_HR[nC],list_HR[nC+6],list_HR[nC+2*6],list_HR[nC+3*6],list_HR[nC+4*6],list_HR[nC+5*6],list_HR[nC+6*6]]
list_LR2=[list_LR[nC],list_LR[nC+6],list_LR[nC+2*6],list_LR[nC+3*6],list_LR[nC+4*6],list_LR[nC+5*6],list_LR[nC+6*6]]
C_hr=pd.DataFrame(list_HR2).transpose().mean(axis=1)
C_lr=pd.DataFrame(list_LR2).transpose().mean(axis=1)
C_hr.plot.kde(title='Scaled Residuals $C_2=10^{10}$')
C_lr.plot.kde(grid=True)
#ax.set_xlim(-10,10)
ax.legend(['HR', 'LR'])
#%% By R1
fig, ax=plt.subplots()
nR=0
Ri_hr=pd.DataFrame(list_HR[nR*6:6*nR+6]).transpose().mean(axis=1)
Ri_lr=pd.DataFrame(list_LR[nR*6:6*nR+6]).transpose().mean(axis=1)
list_HR3=pd.DataFrame([list_HR[nR],list_HR[nC+6],list_HR[nC+2*6],list_HR[nC+3*6],list_HR[nC+4*6],list_HR[nC+5*6],list_HR[nC+6*6]]).transpose()
Ri_hr.plot.kde(title='Scaled Residuals R$_1$=2h')
Ri_lr.plot.kde(grid=True)
ax.set_xlim(-10,10)
ax.legend(['HR', 'LR'])
values_a, bins_a, patches_a = plt.hist(Ri_hr, bins=50, density=True)
values_b, bins_b, patches_b = plt.hist(Ri_lr, bins=50, density=True)
print(histogram_intersection(values_a,values_b))

#%%
values_a, bins_a, patches_a = plt.hist(C_hr, bins=50, density=True)
values_b, bins_b, patches_b = plt.hist(C_lr, bins=50, density=True)
#%%
print(histogram_intersection(values_a,values_b))

#%%

#intersection=intersection.transpose()
def Hist_Intersect(df, fact_list, num_fact):
    df2=pd.DataFrame(df[fact_list[num_fact-4]].values)
    df2=df2.transpose()
    df2.columns=combinations_names#[5:]
    df3=pd.DataFrame()
    for i in range(0,len(df2.columns)):
        df3[df2.columns[i]]=pd.Series(np.log10(df2[df2.columns[i]][0]))#np.log10(pd.Series(df2[df2.columns[i]][0]))
    df2=df3
    fig, ax=plt.subplots(figsize=(8,5))
    bp=df2.boxplot(showfliers=False, rot=90, grid=True, fontsize=12)
#    bp.set_ylim(0,0.005)
    bp.set_ylabel('log$_{10}$(Histogram intersection)')
    bp.set_title(str(num_fact)+ ' FACTORS')
    return df2

#%%
factors=['F4','F5','F6','F7','F8','F9']
intersect_factors=[]
for i in range(0,len(factors)):
    intersect_factors.append(Hist_Intersect(intersection, factors, i+4))
#%%
avg_factor=(intersect_factors[0]+intersect_factors[1]+intersect_factors[2]+intersect_factors[3]+intersect_factors[4]+intersect_factors[5])/6
bp1=avg_factor.boxplot(rot=90, showfliers=False, fontsize=15, showmeans=True, color='black', figsize=(10,6))
bp1.set_ylabel('log$_{10}$(Histogram intersection)', fontsize=18)
#%%
HR_dataset[k][j].plot.kde()
LR_dataset[k][j].plot.kde()
#%%
def histogram_intersection(h1, h2):
    sm = 0
    sM=0
    for i in range(50):
        sm += min(h1[i], h2[i])
        sM+=max(h1[i], h2[i])
    return sm*100/sM




