os.chdir('C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/Comp_Rolling_Seasonal/Time_dependent_factors/MAG')
COA_R=pd.read_csv("TDP_OV_COA_R.txt", sep="\t", infer_datetime_format=True)
COA_S=pd.read_csv("TDP_OV_COA_S.txt", sep="\t", infer_datetime_format=True)
HOA_R=pd.read_csv("TDP_OV_HOA_R.txt", sep="\t", infer_datetime_format=True)
HOA_S=pd.read_csv("TDP_OV_HOA_S.txt", sep="\t", infer_datetime_format=True)
BBOA_R=pd.read_csv("TDP_OV_BBOA_R.txt", sep="\t", infer_datetime_format=True)
BBOA_S=pd.read_csv("TDP_OV_BBOA_S.txt", sep="\t", infer_datetime_format=True)
LO_R=pd.read_csv("TDP_OV_LO_R.txt", sep="\t", infer_datetime_format=True)
LO_S=pd.read_csv("TDP_OV_LO_S.txt", sep="\t", infer_datetime_format=True)
MO_R=pd.read_csv("TDP_OV_MO_R.txt", sep="\t", infer_datetime_format=True)
MO_S=pd.read_csv("TDP_OV_MO_S.txt", sep="\t", infer_datetime_format=True)
d=pd.DataFrame()
d['Time']=pd.to_datetime(COA_R['datetime'], dayfirst=True, errors='coerce')
MO_S['Time']=d['Time']
dr_all=pd.date_range("2017/09/21 00:00",end="2018/11/01") # Change Start date and End date
#%% Time dependent profiles comparison R, S 
l=[]
for i in range(0,len(dr_all)):
    st_d=dr_all[i]
    dr_14=pd.date_range(st_d, periods=14) #You can change the length of rolling R2 window here (periods=14)
    en_d=dr_14[-1]
    mask_i=(d['Time']>st_d) & (d['Time']<=en_d)
    fRC=COA_R.loc[mask_i]
    fSC=COA_S.loc[mask_i]
    fRH=HOA_R.loc[mask_i]
    fSH=HOA_S.loc[mask_i]    
    fRB=BBOA_R.loc[mask_i]
    fSB=BBOA_S.loc[mask_i]
    fRL=LO_R.loc[mask_i]
    fSL=LO_S.loc[mask_i]
    fRM=MO_R.loc[mask_i]
    fSM=MO_S.loc[mask_i]
    rsq=[R2(fRC.mean(axis=0), fSC.mean(axis=0)), R2(fRH.mean(axis=0), fSH.mean(axis=0)),R2(fRB.mean(axis=0), fSB.mean(axis=0)),
         R2(fRL.mean(axis=0), fSL.mean(axis=0)),R2(fRM.mean(axis=0), fSM.mean(axis=0))]
    l.append(rsq)
R=pd.DataFrame(l, columns=['COA', 'HOA', 'BBOA','LO-OOA','MO-OOA'])
R['datetime']=dr_all
R.to_csv('Rolling_R2_TDP_R_S.txt')
R=R.set_index('datetime')
print(R)

#%%
fig_232=sns.heatmap(LO_R)
#%%
fig_R, axes=plt.subplots(nrows=5,ncols=1,sharex=True,figsize=(25,20), constrained_layout=True)
fig_R.suptitle("BARCELONA (Absolute)\n Correlation time-dependent profiles", fontsize=28)
plt.rcParams.update({'font.size': 22})
count=0
for c in range(5):
    name1=R.columns[c]
    axes[c].plot(R.index, R[name1], marker='o', color='black')
    axes[c].set_ylabel('R²')
    axes[c].grid(axis='x')
    axes[c].grid(axis='y')
    axes[c].set_axisbelow(True)
    axes[c].set_title(name1)
    count=count+1
fig_R.savefig('Rolling_R2slope_Rolling_vs_Seas_14_f.png')