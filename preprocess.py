import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
def cate(a):
    if  0<=a<=50:
        return 0
    elif 51<=a<=100:
        return 1
    elif 101<=a<=150:
        return 2
    elif 151<=a<=200:
        return 3
    elif 201<=a<=300:
        return 4
    elif 301<=a:
        return 5
col=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
f_inp=r'combined.csv'
f_tar=r'drop_col_combined.csv'
data=pd.read_csv(f_inp,engine='python')
data = data.drop(columns=0, errors='ignore')
data['label']=data['AQI'].apply(cate)
dr_col=['City','Date','AQI_Bucket','StationId','Datetime','Unnamed: 0','AQI']
data = data.drop(columns=dr_col, errors='ignore')
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
norm=RobustScaler()
data[col]=norm.fit_transform(data[col]) #Normalization
data.to_csv(f_tar,mode='w',header=True,index=False)

