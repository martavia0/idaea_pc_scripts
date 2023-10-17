# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:34:23 2022

@author: Marta Via
"""
import os as os
import pandas as pd
import matplotlib.pyplot as plt
#%% Import TS
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic/Truth")
df = pd.read_csv("TS.txt", sep="\t", low_memory=False, index_col=False)
df['Time'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
dr_all = pd.date_range("2011/02/01 00:00", end="2011/12/31")  # dates
city='SYN'
cityname='Synthetic dataset'
#%% Import PR
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic/Profiles/')#
HOA_R = pd.read_csv("TDP_HOA_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
HOA_S = pd.read_csv("TDP_HOA_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
LO_R = pd.read_csv("TDP_LO_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
LO_S = pd.read_csv("TDP_LO_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
MO_R = pd.read_csv("TDP_MO_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
MO_S = pd.read_csv("TDP_MO_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
OOA_R = pd.read_csv("TDP_OOA_Rolling.txt", sep="\t", infer_datetime_format=True, low_memory=False)
OOA_S = pd.read_csv("TDP_OOA_Seasonal.txt", sep="\t", infer_datetime_format=True, low_memory=False)
BBOA_R = pd.read_csv("TDP_BBOA_Rolling.txt", sep="\t", infer_datetime_format=True)
BBOA_S = pd.read_csv("TDP_BBOA_Seasonal.txt", sep="\t", infer_datetime_format=True)  
os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic/Truth/')
BBOA_T=pd.read_csv("TDP_BBOA.txt", sep="\t", infer_datetime_format=True)
HOA_T=pd.read_csv("TDP_HOA.txt", sep="\t", infer_datetime_format=True)
SOA_T=pd.read_csv("TDP_SOA.txt", sep="\t", infer_datetime_format=True)
SOA_TR=pd.read_csv("TDP_SOA_TR.txt", sep="\t", infer_datetime_format=True)
SOA_BB=pd.read_csv("TDP_SOA_BB.txt", sep="\t", infer_datetime_format=True)
SOA_BIO=pd.read_csv("TDP_SOA_BIO.txt", sep="\t", infer_datetime_format=True)
#%% Import synthetic
os.chdir("C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Synthetic/")
df1=pd.read_csv('Synthetic.txt', sep='\t')
df1['Time'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')

#%% Pie of Apportionment

my_labels = 'HOA', 'BBOA', 'LO-OOA', 'MO-OOA'
fracs_R = [(df['HOA_Rolling']/df['OA_app_Rolling']).mean(),
           (df['BBOA_Rolling']/df['OA_app_Rolling']).mean(),
           (df['LO-OOA_Rolling'] / df['OA_app_Rolling']).mean(), 
           (df['MO-OOA_Rolling']/df['OA_app_Rolling']).mean()]
fracs_S = [(df['HOA_seas']/df['OA_app_seas']).mean(),
           (df['BBOA_seas']/df['OA_app_seas']).mean(),
           (df['LO-OOA_seas'] / df['OA_app_seas']).mean(), 
           (df['MO-OOA_seas']/df['OA_app_seas']).mean()]
fracs_T = [(df['HOA']/df['OA']).mean(),
           (df['BB']/df['OA']).mean(),
           (df['SOA_tr'] / df['OA_app_seas']).mean(),
           (df['SOA_bb'] / df['OA_app_seas']).mean(),
           (df['SOA_bio']/df['OA_app_seas']).mean()]
fig, axs = plt.subplots(1, 3, figsize=(14, 10))
fig.suptitle(cityname, fontsize=25, y=0.75)
axs[0].pie(fracs_R, labels=my_labels, autopct='%1.0f%%',textprops={'fontsize': 18}, shadow=True, colors=['grey', #'firebrick', #darkkhaki', 'sienna','mediumorchid',
           'sienna', 'limegreen', 'darkgreen', ])  # 'olive','mediumorchid','lightskyblue','darkkhaki','sienna', 'tan'
axs[0].set_title('Rolling', fontsize=18)
axs[1].pie(fracs_S, labels=my_labels,  autopct='%1.0f%%',textprops={'fontsize': 18}, shadow=True, colors=['grey',#'firebrick', #'darkkhaki', 'sienna','mediumorchid'
           'sienna', 'limegreen', 'darkgreen', ])  # 'olive',mediumorchid','lightskyblue''darkkhaki','sienna', 'tan'
axs[1].set_title('Seasonal', fontsize=18)
axs[2].pie(fracs_T, labels=['HOA', 'BBOA', 'SOA$_{tr}$', 'SOA$_{bb}$', 'SOA$_{bio}$'], 
           autopct='%1.0f%%',textprops={'fontsize': 18}, shadow=True, colors=['grey','sienna', '#00DB00', '#53FF53','#A6FFA6' ])  
axs[2].set_title('Truth', fontsize=18)
#%% Time Series
df=df.set_index(df.Time)
df_copy=df
df_avg = df.resample('D').mean()
df_plot=df_avg
fig, ax = plt.subplots(3,1,  figsize=(15,15), sharex=True)#, gridspec_kw={
ax[0].plot(df_plot['HOA'], color='black')
ax[0].plot(df_plot['HOA_Rolling'], color='r')
ax[0].plot(df_plot['HOA_seas'], color='b')
ax[0].set_title('HOA', fontsize=18)
ax[0].grid()

ax[0].tick_params(axis='y', labelsize=14 )
ax[0].set_ylabel('24h-avg. conc. $\mu g · m^{-3}$', fontsize=16)
ax[1].plot(df_plot['BB'], color='black')
ax[1].plot(df_plot['BBOA_Rolling'], color='r' )
ax[1].plot(df_plot['BBOA_seas'], color='b')
ax[1].set_title('BBOA', fontsize=18)
ax[1].grid()
ax[1].tick_params(axis='y', labelsize=14 )
ax[1].set_ylabel('24h-avg. conc. $\mu g · m^{-3}$', fontsize=16)
ax[2].plot(df_plot['SOA'], color='black')
ax[2].plot(df_plot['OOA_Rolling'], color='r')
ax[2].plot(df_plot['OOA_seas'], color='b')
ax[2].set_title('OOA', fontsize=18)
ax[2].grid()
ax[2].tick_params(axis='y', labelsize=14 )
ax[2].tick_params(axis='x', labelsize=14 )
ax[2].set_ylabel('24h-avg. conc.  $( \mu g · m^{-3})$', fontsize=16)
ax[2].legend(['Truth', 'Rolling', 'Seasonal'], fontsize=14, loc='upper right')

#%% Ions treatment
ions=pd.DataFrame()
factor='HOA'
ions['4344_'+factor]=df1['mz43_'+factor]/df1['mz44_'+factor]
ions['4443_'+factor]=df1['mz44_'+factor]/df1['mz43_'+factor]
ions['4344_R_'+factor]=df1['mz43_SOA_Rolling']/df1['mz44_SOA_Rolling']
ions['4443_R_'+factor]=df1['mz44_SOA_Rolling']/df1['mz43_SOA_Rolling']
ions['4344_S_'+factor]=df1['mz43_SOA_seas']/df1['mz44_SOA_seas']
ions['4443_S_'+factor]=df1['mz44_SOA_seas']/df1['mz43_SOA_seas']
ions['5744_'+factor]=df1['mz57_'+factor]/df1['mz44_'+factor]
ions['5744_R_'+factor]=df1['mz57_SOA_Rolling']/df1['mz44_SOA_Rolling']
ions['5744_S_'+factor]=df1['mz57_SOA_seas']/df1['mz44_SOA_seas']
ions['6044_'+factor]=df1['mz60_'+factor]/df1['mz44_'+factor]
ions['6044_R_'+factor]=df1['mz60_Rolling']/df1['mz44_SOA_Rolling']
ions['6044_S_'+factor]=df1['mz60_SOA_seas']/df1['mz44_SOA_seas']
ions['6057_'+factor]=df1['mz60_'+factor]/df1['mz57_'+factor]
ions['6057_R_'+factor]=df1['mz60_SOA_Rolling']/df1['mz57_SOA_Rolling']
ions['6057_S_'+factor]=df1['mz60_SOA_seas']/df1['mz57_SOA_seas']
ions=ions.set_index(df1.Time)
#%% Adaptability plots

fig, ax=plt.subplots(4,1, figsize=(10,9))
ax[0].plot(ions['4344_'+factor], color='black')
ax[0].plot(ions['4344_R_'+factor], alpha=0.5, color='red')
ax[0].plot(ions['4344_S_'+factor], alpha=0.5, color='blue')
ax[0].grid() 
ax[0].set_ylabel('43/44 from '+factor)
ax[1].plot(ions['5744_'+factor], color='black')
ax[1].plot(ions['5744_R_'+factor], alpha=0.5, color='red')
ax[1].plot(ions['5744_S_'+factor], alpha=0.5, color='blue')
ax[1].set_ylabel('57/44 from '+factor)
ax[1].grid()
ax[2].plot(ions['6044_'+factor], color='black')
ax[2].plot(ions['6044_R_'+factor], alpha=0.5, color='red')
ax[2].plot(ions['6044_S_'+factor], alpha=0.5, color='blue')
ax[2].set_ylabel('60/44 from '+factor)
ax[2].grid()
ax[3].plot(ions['6057_'+factor], color='black')
ax[3].plot(ions['6057_R_'+factor], alpha=0.5, color='red')
ax[3].plot(ions['6057_S_'+factor], alpha=0.5, color='blue')
ax[3].set_ylabel('60/57 from '+factor)
ax[3].grid()
#%% Substraction KDE
Subs=pd.DataFrame()
fig, ax=plt.subplots(4,1, figsize=(5,9))
Subs['44_43_R']=ions['4344_R_'+factor]-ions['4344_'+factor]
# Subs['44_43_S']=ions['4344_S_'+factor]-ions['4344_'+factor]
Subs['44_43_S']=[0.012727/0.118893, 0.017158/0.109766, 0.012825/0.115706 ] #For HOA
Subs['44_43_R'].plot.hist(bins=100,ax=ax[0], color='r', alpha=0.5)
Subs['44_43_S'].plot.hist(bins=100,ax=ax[0], color='b' ,alpha=0.5)
ax[0].grid()
ax[0].set_ylabel('43/44 from '+factor)
Subs['57_44_R']=ions['5744_R_'+factor]-ions['5744_'+factor]
Subs_S['57_44_S']=[0.079147/0.012727, 0.082735/0.017158, 0.080402/0.012825]
# Subs['57_44_S']=ions['5744_S_'+factor]-ions['5744_'+factor]
Subs['57_44_R'].plot.hist(bins=100,ax=ax[1],color='r' ,alpha=0.5)
Subs['57_44_S'].plot.hist(bins=100,ax=ax[1],color='b',alpha=0.5)
ax[1].grid()
ax[1].set_ylabel('57/44 from '+factor)
Subs['60_44_R']=ions['6044_R_'+factor]-ions['6044_'+factor]
Subs['60_44_S']=ions['6044_S_'+factor]-ions['6044_'+factor]
Subs['60_44_R'].plot.hist(bins=100,ax=ax[2], color='r', alpha=0.5)
Subs['60_44_S'].plot.hist(bins=100,ax=ax[2],color='b',  alpha=0.5)
ax[2].grid()
ax[2].set_ylabel('60/44 from '+factor)
Subs['60_57_R']=ions['6057_R_'+factor]-ions['6057_'+factor]
Subs['60_57_S']=ions['6057_S_'+factor]-ions['6057_'+factor]
Subs['60_57_R'].plot.hist(bins=100,ax=ax[3],color='r', alpha=0.5)
Subs['60_57_S'].plot.hist(bins=100,ax=ax[3],color='b', alpha=0.5)
ax[3].grid()
ax[3].set_ylabel('60/57 from '+factor)
ax[3].set_xlabel('Rolling or Seasonal minus Truth')
# fig.legend(['Rolling', 'Seasonal'],loc='upper right')
#%% Scattered ions
# SOA_ALL=pd.DataFrame()
SOA_ALL = SOA_TR+SOA_BB+SOA_BIO
#SOA_TR*df['SOA_tr']/df['SOA'] #+ SOA_BB*df['SOA_bb']/df['SOA'] +SOA_BIO*df['SOA_bio']/df['SOA'] 
((SOA_ALL['60']/SOA_ALL['57'])/ (SOA_ALL['44']/SOA_ALL['43'])).plot()
ionrat1='4443'
ionrat2='6057'
# fig, axs=plt.subplots()
# plt.scatter(x=ions['4344_SOA'], y=ions['5744_SOA'])
plt.figure(figsize=(5,5))
plt.grid()

# plt.scatter(OOA_R['60']/OOA_R['57'],OOA_R['44']/OOA_R['43'], marker='s',  color='b', s=150 )
# plt.scatter(OOA_S['60']/OOA_S['57'],OOA_S['44']/OOA_S['43'],marker='s',color='r', s=150 )
# plt.scatter(y=SOA_BB['60']/SOA_BB['57'],x=SOA_BB['44']/SOA_BB['43'], color='black',marker='s', s=150)
# plt.scatter(y=SOA_BIO['60']/SOA_BIO['57'],x=SOA_BIO['44']/SOA_BIO['43'], color='black',marker='s', s=150)
# plt.scatter(y=SOA_TR['60']/SOA_TR['57'],x=SOA_TR['44']/SOA_TR['43'], color='black',marker='s', s=150)
plt.scatter(y=SOA_ALL['60']/SOA_ALL['57'],x=SOA_ALL['44']/SOA_ALL['43'], color='black',marker='.')
# plt.scatter(y=BBOA_T['60']/BBOA_T['57'],x=BBOA_T['44']/BBOA_T['43'], color='black',marker='s', s=150)
# plt.scatter(y=HOA_T['60']/HOA_T['57'],x=HOA_T['44']/HOA_T['43'], color='black',marker='s', s=150)

# plt.scatter(x=ions[ionrat1+'_'+factor], y=ions[ionrat2+'_'+factor], marker='.', color='black')
plt.scatter(x=ions[ionrat1+'_R_'+factor], y=ions[ionrat2+'_R_'+factor], marker='.', color='r')
plt.scatter(x=ions[ionrat1+'_S_'+factor], y=ions[ionrat2+'_S_'+factor], marker='.', color='b')
plt.xlabel(factor+' '+ ionrat1[0:2]+'/'+ionrat1[2:], fontsize='14')
plt.ylabel(factor+' '+ ionrat2[0:2]+'/'+ionrat2[2:], fontsize='14')

#%% Diel
columns_R = ['HOA','BB','SOA','HOA_Rolling', 'BBOA_Rolling', 'OOA_Rolling','HOA_seas', 'BBOA_seas','OOA_seas']
diel = pd.DataFrame(df[columns_R])
diel['Hour'] = df['Time'].dt.hour
diel_h = diel.groupby('Hour', axis=0).mean()
fig, ax = plt.subplots(3,1,  figsize=(5,15), sharex=True)#, gridspec_kw={
ax[0].tick_params(axis='y', labelsize=14 )
ax[0].plot(diel_h['HOA'], color='black', lw=3)
ax[0].plot(diel_h['HOA_Rolling'], color='r', lw=3)#, ls=':')
ax[0].plot(diel_h['HOA_seas'], color='b', lw=3)#, ls='--')
ax[0].set_title('HOA', fontsize=16)
ax[0].grid()
ax[1].tick_params(axis='y', labelsize=14 )
ax[1].tick_params(axis='x', labelsize=14 )
ax[1].plot(diel_h['BB'], color='black', lw=3)
ax[1].plot(diel_h['BBOA_Rolling'], color='r', lw=3)#, ls=':')
ax[1].plot(diel_h['BBOA_seas'], color='b', lw=3)#, ls='--')
ax[1].set_title('BBOA', fontsize=16)
ax[1].grid()
ax[2].plot(diel_h['SOA'], color='black', lw=3)
ax[2].plot(diel_h['OOA_Rolling'], color='r', lw=3)#, ls=':')
ax[2].plot(diel_h['OOA_seas'], color='b', lw=3)#, ls='--')
ax[2].set_title('OOA', fontsize=16)
ax[2].grid()
ax[2].legend(['Truth', 'Rolling', 'Seasonal'], fontsize=14)
ax[2].set_xlabel('Hour', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#%%
import matplotlib.patches as mpatches

re=pd.read_csv('Relative errors.txt', sep='\t')
boxprops = dict(linestyle='-', linewidth=0.6)  
meanprops=dict(marker='o', linewidth=0.6,markeredgecolor='black', markerfacecolor='black')
medianprops=dict(color='black')
whiskerprops=dict(color='black')
#%%
fig, axs=plt.subplots(1,3,figsize=(10,3),sharey=True)
re_HOA, re_BBOA, re_SOA=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
re_HOA['Rolling'],re_HOA['Seasonal']=re['HOA_R'], re['HOA_S']
re_BBOA['Rolling'],re_BBOA['Seasonal']=re['BBOA_R'], re['BBOA_S']
re_SOA['Rolling'],re_SOA['Seasonal']=re['SOA_R'], re['SOA_S']

re_HOA.boxplot(ax=axs[0], showfliers=False,showmeans=True, boxprops=boxprops, meanprops=meanprops ,
           medianprops=medianprops, whiskerprops=whiskerprops,  fontsize=11)
re_BBOA.boxplot(ax=axs[1], showfliers=False,showmeans=True, boxprops=boxprops, meanprops=meanprops ,
           medianprops=medianprops, whiskerprops=whiskerprops, fontsize=11)
re_SOA.boxplot(ax=axs[2], showfliers=False,showmeans=True, boxprops=boxprops, meanprops=meanprops ,
           medianprops=medianprops, whiskerprops=whiskerprops, fontsize=11)
axs[0].set_ylabel('Relative error (adim)', fontsize=13)
axs[0].set_title('HOA', fontsize=16)
axs[1].set_title('BBOA', fontsize=16)
axs[2].set_title('SOA', fontsize=16)

# %%  R2 function


def R2(a, b):
    c = pd.DataFrame({"a": a, "b": b})
    cm = c.corr(method='pearson')
    r = cm.iloc[0, 1]
    return r**2


def R_log(a, b):
    c = pd.DataFrame({"a": -1/np.log(a.astype('float')),
                     "b": -1/np.log(b.astype('float'))})
    cm = c.corr(method='pearson')
    r = cm.iloc[0, 1]
    return r**2


def slope(b, a):
    c = pd.DataFrame({"a": a, "b": b})
    mask = ~np.isnan(a) & ~np.isnan(b)
    a1 = a[mask]
    b1 = b[mask]
    if (a1.empty) or (b1.empty):
        s = np.nan
    else:
        s, intercept, r_value, p_value, std_err = linregress(a1, b1)
    return s


def f(B, x):
    return B[0]*x + B[1]


def orthoregress(x, y):
    linreg = linregress(x, y)
    linear = Model(f)
    dat = Data(x, y)
    od = ODR(dat, model=linear, beta0=linreg[0:2])
    out = od.run()
    return list(out.beta)