import os
import pandas as pd
a=[r'archive']
f_t=r'\.csv'

i=0
for pa in a:
    for f in os.listdir(pa):
        print(f"{pa}----{f}")
        data=pd.read_csv(os.path.join(pa,f),engine='python')
        if i==0:
            data.to_csv(f_t,mode='w',header=True)
        else:
            data.to_csv(f_t,mode='a',header=False)


