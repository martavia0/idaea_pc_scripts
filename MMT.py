# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:54:31 2021

@author: Marta Via
"""

import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import seaborn as sns
import matplotlib as plt
#%%  IMPORT ALL
path = "C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/MTR/MT_C/QQexp"
os.chdir(path)
namef=os.listdir(path)
all_files = glob.glob(path+"/*.txt")
f=pd.DataFrame()
c1=0
for filename in all_files:
    df=pd.read_csv(filename, header=0, names=[namef[c1][0:len(namef[c1])-4]],keep_default_na=False)
    f=pd.concat([f,df],axis=1)
    c1=c1+1
# Preliminar analysis plot
f.plot(kind="box",showfliers=False,logy=True, rot=90, title="Q/Qexp for all MT runs",grid=True)
#%%We drop rare runs
f=f.drop('6h_C5', axis=1)
f=f.drop('6h_C10000', axis=1)
f=f.drop('6h_C1000', axis=1)
f=f.drop('12h_C1000', axis=1)
f=f.drop('12h_C10000', axis=1)
#%%
f=f.set_index(f.index)
f=f.drop([59],axis=0)
f = f.reindex(['30m_C0.0001', '30m_C0.001','30m_C0.01','30m_C0.1','30m_C1','30m_C10','30m_C100',
               '1h_C0.0001', '1h_C0.001','1h_C0.01','1h_C0.1','1h_C1','1h_C10','1h_C100',
               '2h_C0.0001', '2h_C0.001','2h_C0.01','2h_C0.1', '2h_C1','2h_C10','2h_C100',
               '3h_C0.0001', '3h_C0.001','3h_C0.01','3h_C0.1', '3h_C1','3h_C10','3h_C100',
               '6h_C0.0001', '6h_C0.001','6h_C0.01','6h_C0.1', '6h_C1','6h_C10','6h_C100',
               '12h_C0.0001', '12h_C0.001','12h_C0.01','12h_C0.1','12h_C1', '12h_C10','12h_C100',
               '24h_C0.0001', '24h_C0.001','24h_C0.01','24h_C0.1','24h_C1', '24h_C10','24h_C100'], axis=1)
#%%
f4=f[0:9]
f5=f[10:19]
f6=f[20:29]
f7=f[30:39]
f8=f[40:49]
f9=f[50:59]
#%%
ax=f4.plot(kind="box",showfliers=False,logy=True, rot=90, title="4 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
ax=f5.plot(kind="box",showfliers=False,logy=True, rot=90, title="5 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
ax=f6.plot(kind="box",showfliers=False,logy=True, rot=90, title="6 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
ax=f7.plot(kind="box",showfliers=False,logy=True, rot=90, title="7 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
ax=f8.plot(kind="box",showfliers=False,logy=True, rot=90, title="8 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
ax=f9.plot(kind="box",showfliers=False,logy=True, rot=90, title="9 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax.set_xlabel("Runs")
ax.set_ylabel("Q/Qexp")
#%%
R=[]
C=[]
for label in f.columns:
    if label[0:3]=='12h' or label[0:3]=='24h' or label[0:3]=='30m':
        res=label[0:3]
        c1=label[5:]
    else:
        res=label[0:2]
        c1=label[4:]
    R.append(res)
    C.append(c1)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
def Heatmap(df):
    li=[]
    df_avg=np.log10(df.median(axis=0,skipna=True))
    print(len(df_avg))
    for i in range(0,49,7):
        print([df_avg.iloc[i],df_avg.iloc[i+1],df_avg.iloc[i+2],df_avg.iloc[i+3],df_avg.iloc[i+4],df_avg.iloc[i+5],df_avg.iloc[i+6]])
        li.append([df_avg.iloc[i],df_avg.iloc[i+1],df_avg.iloc[i+2],df_avg.iloc[i+3],df_avg.iloc[i+4],df_avg.iloc[i+5],df_avg.iloc[i+6]])
    df2=pd.DataFrame(li, columns=['C=0.0001','C=0.001', 'C=0.01','C=0.1','C=1','C=10','C=100'],index=['30m','1h', '2h', '3h','6h','12h','24h'])
    plt.figure(figsize=(30,20))
    sns.set(font_scale=6)
    ax=plt.axes()
    sns.heatmap(df2, cmap="Blues",annot=True,ax=ax)
    plt.xlabel('C values')
    plt.ylabel('R$_1$')#Resolution of the dataset '+'\n'+'containing OA, NR-PM$_{1}$, BC'+'\n')
    ax.set_title('Log Q/Q$_{exp}$'+'\n'+'')
    plt.show()
    return df2
#%%tu 
aa=Heatmap(f1)
#%%

path = "C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/MTR/MT_C/Separated/Q"
os.chdir(path)
namef=os.listdir(path)
all_files = glob.glob(path+"/*h.txt")
t=pd.DataFrame()
c1=0
for filename in all_files:
    df=pd.read_csv(filename, header=0, names=[namef[c1][0:len(namef[c1])-4]], skip_blank_lines=False)
    t=pd.concat([t,df],axis=1)
    c1=c1+1
#%%
all_files_2 = glob.glob(path+"/QQexp_C*")
t2=pd.DataFrame()
c2=7
for filename2 in all_files_2:
    df2=pd.read_csv(filename2, header=0, names=[namef[c2][0:len(namef[c2])-4]], skip_blank_lines=False)
    t2=pd.concat([t2,df2],axis=1)
    c2=c2+1    
    #%%
del(t2['QQexp_C1000'])
del(t2['QQexp_C10000'])
    #%%
t = t.reindex(['QQexp_0.5h', 'QQexp_1h','QQexp_2h','QQexp_3h','QQexp_6h','QQexp_12h','QQexp_24h'],axis=1)
t2 = t2.reindex(['QQexp_C0.0001', 'QQexp_C0.001','QQexp_C0.01','QQexp_C0.1','QQexp_C1','QQexp_C10','QQexp_C100'],axis=1)
#%%
t2.plot(kind='box',showfliers=False, figsize=(20,15),rot=90, grid=True)
plt.yscale('log')
    #%%
tp=pd.DataFrame()
Res_val=pd.Series({'0.5h':0.5, '1h':1, '2h':2,'3h':3,'6h':6,'12h':12,'24h':24})
C_val=pd.Series({'C0.0001':0.0001, 'C0.001':0.001, 'C0.01':0.01,'C0.1':0.1,'C1':1,'C10':10,'C100':100})#,'C1000':1000,'C10000':10000})
tp['p4']=t[0:9].median()
tp['p5']=t[10:19].median()
tp['p6']=t[20:29].median()
tp['p7']=t[30:39].median()
tp['p8']=t[40:49].median()
tp['p9']=t[50:59].median()
tfi=pd.DataFrame()
tfi['p4']=t2[0:9].median()
tfi['p5']=t2[10:19].median()
tfi['p6']=t2[20:29].median()
tfi['p7']=t2[30:39].median()
tfi['p8']=t2[40:49].median()
tfi['p9']=t2[50:59].median()
#%% Verification QACSM I: does Q/Qexp(ACSM) depend on p? 
bp=tp.boxplot(showfliers=False,grid=True,fontsize=12,figsize=(6,6))
bp.set_title("Q/Qexp ACSM", fontsize=15)
bp.set_xlabel('Number of factors',fontsize=13)
bp.set_ylabel('Q/Q$_{exp}$',fontsize=13)
#Yes it does.
#%% Verification QACSM II: does Q/Qexp(ACSM) depend on R1? 
tp2=np.log10(tp.transpose())
tp2=tp2.rename({'QQexp_0.5h':'0.5h','QQexp_1h':'1h','QQexp_2h':'2h','QQexp_3h':'3h','QQexp_6h':'6h','QQexp_12h':'12h','QQexp_24h':'24h'})
bp=tp2.boxplot(showfliers=False,grid=True,fontsize=12,rot=90, figsize=(6,6))
bp.set_title("Log$_{10}$Q/Q$_{exp    ACSM}$", fontsize=15)
bp.set_xlabel('R$_1$',fontsize=13)
bp.set_xticklabels(['0.5h','1h','2h', '3h', '6h','12h', '24h'], fontsize=12, minor=False)
bp.set_ylabel('Log$_{10}$ Q/Q$_{exp}$',fontsize=13)
#Yes it does.
#%% Verificaction QACSM III: does QACSM depend on p?
lACSM=[]
print(p)
c=0
for i in (Res_val.index.values):
    lACSM.append(float(m)*(nd*24/Res_val[i])-p*(float(m)+nd*24/Res_val[i]))
QexpACSM=pd.Series(lACSM)
Q_only_ACSM=pd.DataFrame(data=tp.mul(QexpACSM.values,axis=0))
bp2=Q_only_ACSM.boxplot(showfliers=False,grid=True,fontsize=10, figsize=(6,6))
bp2.set_xlabel('Number of factors',fontsize=13)
bp2.set_ylabel('Q$_{ACSM}$ (10$^6$)',fontsize=13)
bp2.set_title('Q $_{ACSM}$ = Q/Q$_{exp}$)$_{ACSM}$ · Q$_{exp   ACSM}$', fontsize=14)#\cdot$ Q$_{exp ACSM}', fontsize=15)
bp2.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],fontsize=12, minor=False)
#Yes it does
#%% Verification QACSM IV: does Q/Qexp(ACSM) depend on R1? 
Q_only_ACSM2=np.log10(Q_only_ACSM.transpose())
bp3=Q_only_ACSM2.boxplot(showfliers=False,grid=True,fontsize=12,rot=90, figsize=(6,6))
bp3.set_xlabel('R$_1$',fontsize=13)
bp3.set_ylabel('Q$_{ACSM}$',fontsize=13)
bp3.set_xticklabels(['0.5h','1h','2h', '3h', '6h','12h', '24h'], fontsize=12, minor=False)
bp3.set_title('Log$_{10}$ of Q$_{ACSM}$ = Q/Q$_{exp}$)$_{ACSM}$ · Q$_{exp   ACSM}$', fontsize=14)#\cdot$ Q$_{exp ACSM}', fontsize=15)
#%% Verification QACSM V: How did the dependency of QQexp(ACSM) depend on p and R1 along? 
plt.figure(figsize=(30,20))
sns.set(font_scale=6)
ax=plt.axes()
plt.xlabel('C values')
sns.heatmap(Q_only_ACSM2, cmap='Blues', annot=True, ax=ax)
ax.set_xlabel('R$_1$', fontsize=14)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
ax.set_xticklabels(['0.5h','1h','2h', '3h', '6h','12h', '24h'], minor=False, fontsize=40)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6])
ax.set_yticklabels(['p4','p5','p6', 'p7', 'p8','p9'], minor=False, fontsize=40)
ax.set_ylabel('Number of factors',fontsize=40)
ax.set_title('Log$_{10}$ of QQ$_{exp ACSM}$ $', fontsize=50)
plt.show()
#%% Verification QACSM VI: How does the dependency of Q(ACSM) depend on p and R1 along? 
fig,ax=plt.subplots(figsize=(8,6))
hm3=sns.heatmap(Q_only_ACSM2, cmap='Blues',annot=True)
plt.xlabel('Resolution')
ax.set_xlabel('R$_1$', fontsize=14)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
ax.set_xticklabels(['0.5h','1h','2h', '3h', '6h','12h', '24h'], minor=False, fontsize=12)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6])
ax.set_ylabel('Number of factors',fontsize=14)
ax.set_yticklabels(['p4','p5','p6','p7','p8','p9'],fontsize=12, minor=False)
ax.set_title('Log$_{10}$ of Q$_{ACSM}$ = Q/Q$_{exp}$)$_{ACSM}$· Q$_{exp   ACSM}$' , fontsize=16)
#%% Verification Qfilt I: Does Q/Qexp(F) depend on p? 
bp=tfi.boxplot(showfliers=False,grid=True,fontsize=12,figsize=(6,6))
bp.set_title("Q/Qexp Filters", fontsize=15)
bp.set_xlabel('Number of factors',fontsize=13)
bp.set_ylabel('Q/Q$_{exp}$',fontsize=13)
# No it doesn't.
#%% Verification Qfilt II: does Q/Qexp(F) depend on C? 
tfi2=np.log10(tfi.transpose())
bp=tfi2.boxplot(showfliers=False,grid=True,fontsize=12,rot=90, figsize=(6,6))
bp.set_title("Log$_{10}$Q/Q$_{exp   Filters}$", fontsize=15)
bp.set_xlabel('C value',fontsize=13)
bp.set_xticklabels(['C=0.0001','C=0.001','C=0.01', 'C=0.1', 'C=1','C=10', 'C=100'], fontsize=12, minor=False)
bp.set_ylabel('Log$_{10}$ Q/Q$_{exp}$',fontsize=13) 
#Yes it does.
#%% Verificaction Qfilt III: does QF depend on p?
Q_only_filters=pd.DataFrame(data=tfi*(int(m)*int(nf)-int(p)*(int(m+nf))))
bp2=Q_only_filters.boxplot(showfliers=False,grid=True,fontsize=12, figsize=(6,6))
bp2.set_xlabel('Number of factors',fontsize=13)
bp2.set_ylabel('Q$_{Filters}$ (10$^5$)',fontsize=13)
bp2.set_title('Q $_{Filters}$ = Q/Q$_{exp}$)$_{Filters}$ · Q$_{exp   Filters}$', fontsize=14)#\cdot$ Q$_{exp ACSM}', fontsize=15)
bp2.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],fontsize=12, minor=False)
#No it does not.
#%% Verification Qfilt IV: does Q(Filt) depend on R1? 
Q_only_filters2=np.log10(Q_only_filters.transpose())
bp3=Q_only_filters2.boxplot(showfliers=False,grid=True,fontsize=12,rot=90, figsize=(6,6))
bp3.set_xlabel('R$_1$',fontsize=13)
bp3.set_ylabel('Q$_{ACSM}$',fontsize=13)
bp3.set_xticklabels(['C=0.0001','C=0.001','C=0.01', 'C=0.1', 'C=1','C=10', 'C=100'], fontsize=12, minor=False)
bp3.set_title('Log$_{10}$ of Q$_{ACSM}$ = Q/Q$_{exp}$)$_{ACSM}$ · Q$_{exp   ACSM}$', fontsize=14)#\cdot$ Q$_{exp ACSM}', fontsize=15)
#%% Verification QF V: How did the dependency of QQexp(Filters) depend on p and R1 along? 
fig,ax=plt.subplots(figsize=(8,6))
hm4=sns.heatmap(tfi2, cmap='Blues',annot=True)
ax.set_xlabel('C value', fontsize=14)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
ax.set_xticklabels(['C=0.0001','C=0.001','C=0.01', 'C=0.1', 'C=1','C=10', 'C=100'], minor=False, fontsize=12)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6])
ax.set_yticklabels(['p4','p5','p6', 'p7', 'p8','p9'], minor=False, fontsize=12)
ax.set_ylabel('Number of factors',fontsize=14)
ax.set_title('Log$_{10}$ of QQ$_{exp Filt}$ ', fontsize=16)
#%% Verification QF VI: How does the dependency of Q(Filters) depend on p and R1 along? 
fig,ax=plt.subplots(figsize=(8,6))
hm3=sns.heatmap(Q_only_filters2, cmap='Blues',annot=True)
plt.xlabel('Resolution')
ax.set_xlabel('C values', fontsize=14)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
ax.set_xticklabels(['C=0.0001','C=0.001','C=0.01', 'C=0.1', 'C=1','C=10', 'C=100'], minor=False, fontsize=12)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6])
ax.set_ylabel('Number of factors',fontsize=14)
ax.set_yticklabels(['p4','p5','p6','p7','p8','p9'],fontsize=12, minor=False)
ax.set_title('Log$_{10}$ of Q$_{Filt}$ = Q/Q$_{exp}$)$_{Filt}$· Q$_{exp   Filt}$' , fontsize=16)
#%% Verification Denominator I: dependency on R_1
denomp=[]
for i in range(4,10):
    denomp.append(1/((nf+24*nd/0.5)*m - i*(m+nf+24*nd/0.5)))
    print(i)
denominadorp=pd.Series(denomp, index=['4F','5F','6F', '7F','8F','9F'])
fig,ax=plt.subplots(figsize=(6,6))
dn=denominadorp.plot(linewidth=0, marker='o')
ax.set_xlabel('Num factors',fontsize=14)
ax.set_ylabel(r'$ \frac{1}{(24\cdot \frac{n_D}{R_1}+nf)\cdot m - p\cdot(m+24\cdot \frac{n_D}{R_1}+nf)}$', fontsize=17)
plt.yticks([6.4e-7,6.5e-7,6.6e-7,6.7e-7,6.8e-7, 6.9e-7, 6.94e-7])
ax.set_yticklabels(['6.4','6.5','6.6','6.7','6.8','6.9', '(·10$^{-7}$)'],fontsize=12, minor=False)
ax.set_title("Denominator (p)\n Res = 30 min", fontsize=17)
#%% Verification Denominator II: dependency on R_1
denom=[]
for i in Res_val:
    denom.append(1/((24*nd/i+nf)*m - p*(m+nf+24*nd/i)))
denominador=pd.Series(denom, index=['0.5h','1h','2h', '3h', '6h','12h', '24h'])
fig,ax=plt.subplots(figsize=(6,6))
dn=denominador.plot(linewidth=0, marker='o')
ax.set_xlabel('R$_1$',fontsize=16)
ax.set_ylabel(r'$ \frac{1}{(24\cdot \frac{n_D}{R_1}+nf)\cdot m - p\cdot(m+24\cdot \frac{n_D}{R_1}+nf)}$', fontsize=17)
plt.yticks([0.0,1e-5 ,2e-5,3e-5,3.5e-5])
ax.set_yticklabels(['0.0','1e-5','2e-5','3e-5','(· 10$^{-5})$'],fontsize=12, minor=False)
ax.set_title("Denominator(R$_1$)\n p=7", fontsize=17)
#%% Verification Denominator III: dependency along p,R_1
denomi=[]
deno=[]
ij=[]
for i in Res_val:
    denomi=[]
    for j in range(4,10,1):
        print(i,j,denomi)
        denomi.append(1/((24*nd/i+nf)*m - j*(m+nf+24*nd/i)))
        ij.append([i,j])
    deno.append(denomi)
den=pd.DataFrame(deno, columns=['4F','5F','6F','7F','8F','9F'], index=['0.5h','1h','2h','3h','6h','12h','24h'])
#%% Verification Denominator III: dependency along p,R_1
fig,ax=plt.subplots(figsize=(30,20))
den=den.T
denn=sns.heatmap(np.log10(den), annot=True, cmap='Blues', cbar=False,square=True)
ax.set_xticklabels(den.columns, fontsize=40)
ax.set_ylabel('Num. factors', fontsize=40)
ax.set_xlabel('R$_1$', fontsize=40)
ax.set_yticklabels(['p=4', 'p=5','p=6','p=7', 'p=8', 'p=9'], minor=False, fontsize=40)
ax.set_xticklabels(['0.5h','1h','2h','3h','6h','12h','24h'], minor=False, fontsize=40)
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
ax.set_title('Log$_{10}$(' +r' $\frac{1}{(24\cdot \frac{n_D}{R_1}+nf)\cdot m - p\cdot(m+24\cdot \frac{n_D}{R_1}+nf)})$'+'\n', fontsize=50)
#%% QQ theory calculation!!
nd=365.0
nf=90.0
m=92.0
def QQ(p,res,c): 
    print(p,res,c)
    p_txt='p'+str(p)
    res_txt='QQexp_'+res
    C_txt = 'QQexp_'+c
    Qexp_ACSM=m*(nd*24/Res_val[res])-p*(m+nd*24/Res_val[res])
    Qexp_F=m*nf-p*(m+nf)
    QQ_theory=(tp[p_txt][res_txt]*Qexp_ACSM+tfi[p_txt][C_txt]*Qexp_F/C_val[c]**2)/((nf+24*nd/Res_val[res])*m-p*(m+nf+24*nd/Res_val[res]))
    return QQ_theory
#%%
for C in C_val:
    p=7
    p_txt='p7'
    bb=pd.DataFrame(tfi[p_txt][C_txt]*Qexp_F/C_val[c]**2)
#%%
p=7
Q_Qexp=pd.DataFrame(columns=list(Res_val.index.values), index=list(C_val.index.values))
lQ=[]
for i in (Res_val.index.values):
    lQ=[]
    for j in list(C_val.index.values):
        print(i,j, "end")
        lQ.append(QQ(p,i,j))
    Q_Qexp[i]=np.log10(lQ)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
aaa=Q_Qexp.transpose()
fig,ax=plt.subplots(figsize=(22,20))
df2=pd.DataFrame(aaa, columns=['C=0.0001','C=0.001', 'C=0.01','C=0.1','C=1','C=10','C=100'],index=['30m','1h', '2h', '3h','6h','12h','24h'])
holi=sns.heatmap(aaa, cmap="Blues",annot=True,cbar=False)
sns.set(font_scale=5)
plt.xlabel('C values')
plt.ylabel('R$_1$')#Resolution of the dataset '+'\n'+'containing OA, NR-PM$_{1}$, BC'+'\n')
ax.set_title('Log Q/Q$_{exp}$'+'\n')
plt.show()
#%%
sns.heatmap(Q_Qexp,cmap='Blues',annot=True)
sns=plt.axes()
#%%
plt.xlabel('C values')
plt.ylabel('R$_1$')#Resolution of the dataset '+'\n'+'containing OA, NR-PM$_{1}$, BC'+'\n')
ax.set_title('Log Q/Q$_{exp}$'+'\n'+'')
#%%
C_value=C_val.to_list()*7
Res_value=Res_val.repeat(7)
#%%
QQ_exper=aa.transpose()
QQ_theor=Q_Qexp
QQ_th=pd.DataFrame({'QQ_theoret': [*QQ_theor['0.5h'],*QQ_theor['1h'],*QQ_theor['2h'],*QQ_theor['3h'],*QQ_theor['6h'],*QQ_theor['12h'],*QQ_theor['24h']]})
QQ_ex=pd.DataFrame({'QQ_experim': [*QQ_exper['30m'],*QQ_exper['1h'],*QQ_exper['2h'],*QQ_exper['3h'],*QQ_exper['6h'],*QQ_exper['12h'],*QQ_exper['24h']]})
QQ_ex['QQ_experimental']=10**QQ_ex['QQ_experim'].values
QQ_comp=pd.DataFrame({'QQ_theoretical':QQ_th['QQ_theoret'], 'QQ_experimental':QQ_ex['QQ_experimental']})
QQ_comp['C_val']=C_value
QQ_comp['Res_val']=Res_value.to_list()
#%%
# *******************************************************************************
#               UNEXPlAINED VARIATION
# *********************************************+******+++++++********************
path = "C:/Users/maria/Documents/Marta Via/1. PhD\A. BCN_Series/MTR/MT_C/Variation_matrix"
os.chdir(path)
namef=os.listdir(path)
all_files = glob.glob(path+"/*.txt")
f=pd.DataFrame()
c1=0
#for filename in all_files:
for c in range(0,44):
    df=pd.read_csv(namef[c1],sep='\t',header=None,keep_default_na=False)
    df2=pd.DataFrame(data=df.values,columns=[str(namef[c1])+'_F1',str(namef[c1])+'_F2',str(namef[c1])+'_F3',str(namef[c1])+'_F4',
                                             str(namef[c1])+'_F5',str(namef[c1])+'_F6',str(namef[c1])+'_F7',str(namef[c1])+'_F8',
                                             str(namef[c1])+'_F9',str(namef[c1])+'_Un_noisy',str(namef[c1])+'_Un_real'])
    f=pd.concat([f,df2],axis=1)
    c1=c1+1   
#%%
f1=pd.DataFrame()
for col in f.columns:
    if col[-4:]=='real':
        colu=col[:-12]
        print(colu)
        f1[colu]=f[col]     
#%%
f1 = f1.reindex(['0.5h_C0.0001', '0.5h_C0.001','0.5h_C0.01','0.5h_C0.1','0.5h_C1','0.5h_C10','0.5h_C100',
               '1h_C0.0001', '1h_C0.001','1h_C0.01','1h_C0.1','1h_C1','1h_C10','1h_C100',
               '2h_C0.0001', '2h_C0.001','2h_C0.01','2h_C0.1', '2h_C1','2h_C10','2h_C100',
               '3h_C0.0001', '3h_C0.001','3h_C0.01','3h_C0.1', '3h_C1','3h_C10','3h_C100',
               '6h_C0.0001', '6h_C0.001','6h_C0.01','6h_C0.1', '6h_C1','6h_C10','6h_C100',
               '12h_C0.0001', '12h_C0.001','12h_C0.01','12h_C0.1','12h_C1', '12h_C10','12h_C100',
               '24h_C0.0001', '24h_C0.001','24h_C0.01','24h_C0.1','24h_C1', '24h_C10','24h_C100'], axis=1)
 #%%
f1.plot(kind="box",showfliers=False, rot=90, title="Unexplained variation Real",grid=True, figsize=(20,10))
plt.tick_params(axis='both', labelsize=16) 
plt.title('Unexplained variation real', fontsize=20)
#%%
f=f.set_index(f.index)
f=f.drop([59],axis=0)

#%%
f4=f[0:9]
f5=f[10:19]
f6=f[20:29]
f7=f[30:39]
f8=f[40:49]
f9=f[50:59]
#%%
ax4=f4.plot(kind="box",showfliers=False,logy=True, rot=90, title="4 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax4.set_xlabel("Runs")
ax4.set_ylabel("Q/Qexp")
ax5=f5.plot(kind="box",showfliers=False,logy=True, rot=90, title="5 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax5.set_xlabel("Runs")
ax5.set_ylabel("Q/Qexp")
ax6=f6.plot(kind="box",showfliers=False,logy=True, rot=90, title="6 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax6.set_xlabel("Runs")
ax6.set_ylabel("Q/Qexp")
ax7=f7.plot(kind="box",showfliers=False,logy=True, rot=90, title="7 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax7.set_xlabel("Runs")
ax7.set_ylabel("Q/Qexp")
ax8=f8.plot(kind="box",showfliers=False,logy=True, rot=90, title="8 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax8.set_xlabel("Runs")
ax8.set_ylabel("Q/Qexp")
ax9=f9.plot(kind="box",showfliers=False,logy=True, rot=90, title="9 FACTORS"+ "\n"+"Q/Qexp",grid=True)
ax9.set_xlabel("Runs")
ax9.set_ylabel("Q/Qexp")
#%%
R=[]
C=[]
for label in f.columns:
    if label[0:3]=='12h' or label[0:3]=='24h' or label[0:3]=='30m':
        res=label[0:3]
        c1=label[5:]
    else:
        res=label[0:2]
        c1=label[4:]
    R.append(res)
    C.append(c1)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
def Heatmap(df):
    li=[]
    df_avg=(df.mean(axis=0,skipna=True))
    print(len(df_avg))
    for i in range(0,49,7):
        li.append([df_avg.iloc[i],df_avg.iloc[i+1],df_avg.iloc[i+2],df_avg.iloc[i+3],df_avg.iloc[i+4],df_avg.iloc[i+5],df_avg.iloc[i+6]])
    df2=pd.DataFrame(li, columns=['C=0.0001','C=0.001', 'C=0.01','C=0.1','C=1','C=10','C=100'],index=['30m','1h', '2h', '3h','6h','12h','24h'])
    plt.figure(figsize=(30,20))
    sns.set(font_scale=6)
    ax=plt.axes()
    sns.heatmap(df2, cmap="Blues",annot=True,ax=ax)
    plt.xlabel('C values')
    plt.ylabel('R$_1$')#Resolution of the dataset '+'\n'+'containing OA, NR-PM$_{1}$, BC'+'\n')
    ax.set_title('Unexplained variation (μg·m$^{-3}$)'+'\n'+'')
    plt.show()
    return df2
#%%tu 
aa=Heatmap(f6)
#%%
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.interpolate import griddata
resolution1=[0.5,1,2,3,6,12,24]
Cvalue=[0.0001,0.001,0.01,0.1,1,10,100]
p=[4,5,6,7,8,9]
nd=365.0
nf=90.0
m=92.0
lR1=[]
lC=[]
lp=[]
lq=[]
for i in p:
    p_txt='p'+str(i)
    for j in Cvalue:
        for k in resolution1:
            lR1.append(k)
            lC.append(j)
            lp.append(i)
            Qexp_ACSM=m*(nd*24.0/float(k))-i*(m+nd*24.0/float(k))
            Qexp_F=m*nf-i*(m+nf)
            lq.append((tp['p'+str(i)]['QQexp_'+str(k)+'h']*Qexp_ACSM+tfi['p'+str(i)]['QQexp_C'+str(j)]*Qexp_F/j**2)/((nf+24*nd/k)*m-i*(m+nf+24*nd/k)))

arrR1=np.array(lR1)
arrC=np.array(lC)
arrp=np.array(lp)
arrq=np.array(lq)
            #%%
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 12})
fig=plt.figure(figsize=(8,8))
ax=plt.axes(projection='3d')
hey=ax.plot_trisurf((arrR1),np.log(arrC),np.log10(arrq))
ax.set_xlabel('($_1$',fontsize=16, fontweight ='bold')
ax.set_ylabel('log10(C value)',fontsize=16,fontweight ='bold')
ax.set_yticks([-10,-8, -6,-4,-2,0])
plt.ylim(-12,2)
ax.set_yticklabels(['-0.0001','0.001', '0.01','0.1', '1','10'], fontsize=14)
ax.set_xticks([0,0.5,1,2,3,6,12,24])
ax.set_xticklabels(['','0.5h','1h','2h','3h','6h','12h','24h'], fontsize=14, rotation=-60)#range(0,25,4), fontsize=14)
ax.zaxis.set_rotate_label(True) 
ax.view_init(45,65)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
ax.set_zlabel('log10(Q/Q$_{exp}$)$_{th})$', fontsize=16, fontweight ='bold')
ax.set_zticks([0,2,4,6,8])
ax.set_zticklabels([0,'10$^2$','10$^4$','10$^6$','10$^8$'],fontsize=10)
plt.grid()
plt.show()
#img=ax.plot_surface(arrR1,arrC,np.array([arrq,arrp]), rstride=1, cstride=1)# c=arrp)
#%%
whole=[]
dfs=[Heatmap(f4),Heatmap(f5),Heatmap(f6),Heatmap(f7),Heatmap(f8),Heatmap(f9)]
for k in range(0,6):
    for i in range(0,7):
        for j in range(0,7):
            whole.append(float(dfs[k].iloc[i,j]))
#%%
arr_q=np.array(whole)
fig=plt.figure(figsize=(8,8))
ax=plt.axes(projection='3d')
hey=ax.plot_trisurf((arrR1),np.log(arrC), arr_q)
ax.set_xlabel('(R$_1$)',fontsize=16, fontweight ='bold')
ax.set_ylabel('log10(C value)',fontsize=16,fontweight ='bold')
ax.set_yticks([-10,-8, -6,-4,-2,0])
plt.ylim(-12,2)
plt.xlim(-1,25)
ax.set_yticklabels(['-0.0001','0.001', '0.01','0.1', '1','10'], fontsize=14)
ax.set_xticks([0,0.5,1,2,3,6,12,24])
ax.set_xticklabels(['','0.5h','1h','2h','3h','6h','12h','24h'], fontsize=14, rotation=-60)
ax.zaxis.set_rotate_label(True) 
ax.view_init(45,60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
ax.set_zlabel('log10(Q/Q$_{exp}$)$_{exp})$', fontsize=16, fontweight ='bold')
ax.set_zticks([-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0])
ax.set_zticklabels(['10$^{-2}$','10$^{-1.75}$','10$^{-1.5}$','10$^{-1.25}$','10$^{-1}$','10$^{-0.75}$','10$^{-0.5}$', '10$^{-0.25}$', ],fontsize=10)
plt.grid()
plt.show()