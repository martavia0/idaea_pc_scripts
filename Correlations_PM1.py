# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:27:12 2021

@author: Marta Via
"""

import pandas as pd
import pandas as pd
import numpy as np
import glob
import os as os
import datetime as dt
import seaborn as sns
import matplotlib as plt
#%%
path = "C:/Users/maria/Documents/Marta Via/1. PhD/A. BCN_Series/MTR/Correlations_species/"
os.chdir(path)
specs=pd.read_csv('specs.txt', sep='\t')
dt=pd.read_csv('start_time.txt', sep='\t', infer_datetime_format=True)
mz=pd.read_csv('amus.txt', sep='\t')
dt=(pd.to_datetime(stt['start_time'], format="%d/%m/%Y", errors='coerce'))
#%%
specs.columns=mz['mz']
specs.index=dt[:-1]
#%%
corr=specs.corr()
import seaborn as sns
sns.heatmap(df)
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt