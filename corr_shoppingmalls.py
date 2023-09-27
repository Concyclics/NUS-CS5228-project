#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_train = pd.read_csv('/kaggle/input/cs5228-2310-final-team31/train.csv')
df_test = pd.read_csv('/kaggle/input/cs5228-2310-final-team31/test.csv')


# In[3]:


df_malls= pd.read_csv('/kaggle/input/cs5228-2310-final-team31/sg-shopping-malls.csv')


# In[4]:


df_train.describe()
df_train['rent_approval_date'] = pd.to_datetime(df_train['rent_approval_date'])
df_train['rent_approval_year'] = df_train['rent_approval_date'].dt.year
df_train['rent_approval_month'] = df_train['rent_approval_date'].dt.month/12 + df_train['rent_approval_date'].dt.year


# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def from_lat_long_to_dis_vect(df, latB, lonB):
    ra = 6378140
    rb = 6356755
    flatten = (ra - rb) / ra
    
    rad_lat_A = np.radians(df['latitude'])
    rad_lng_A = np.radians(df['longitude'])
    rad_lat_B = np.radians(latB)
    rad_lng_B = np.radians(lonB)
    
    pA = np.arctan(rb / ra * np.tan(rad_lat_A))
    pB = np.arctan(rb / ra * np.tan(rad_lat_B))
    x = np.arccos(np.sin(pA) * np.sin(pB) + np.cos(pA) * np.cos(pB) * np.cos(rad_lng_A - rad_lng_B))
    
    eps = 1e-6
    c1 = (np.sin(x) - x) * (np.sin(pA) + np.sin(pB)) ** 2 / (np.cos(x / 2) ** 2 + eps)
    c2 = (np.sin(x) + x) * (np.sin(pA) - np.sin(pB)) ** 2 / (np.sin(x / 2) ** 2 + eps)
    
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance /= 1000
    return distance

# for idx, row in df_malls.iterrows():
#     df_train['dist_to_mall_' + str(idx)] = from_lat_long_to_dis_vect(df_train, row['latitude'], row['longitude'])
    
# dist_to_malls_cols = ['dist_to_mall_' + str(idx) for idx in range(len(df_malls))]
# cols_to_check = ['monthly_rent'] + dist_to_malls_cols
# correlation_matrix = df_train[cols_to_check].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='Blues')


# In[8]:


import multiprocessing as mp

def calculate_distance(row):
    dist_to_malls = {}
    for _, mall in df_malls.iterrows():
        dist_to_malls[mall['name']] = from_lat_long_to_dis_vect(row, mall['latitude'], mall['longitude'])
    return dist_to_malls

pool = mp.Pool(mp.cpu_count())
result_list = pool.map(calculate_distance, [row for _, row in df_train[['latitude', 'longitude']].iterrows()])
pool.close()

df_dist_to_malls = pd.DataFrame(result_list)
df_train = pd.concat([df_train, df_dist_to_malls], axis=1)


# In[9]:


dist_to_malls_cols = [name for name in df_malls['name']]
cols_to_check = ['monthly_rent'] + dist_to_malls_cols

corr = df_train[cols_to_check].corr()
plt.figure(figsize=(100,100),dpi=300)
sns.heatmap(corr, annot=True, cmap='Blues')
plt.savefig('/kaggle/working/heatmap4.pdf', format='pdf')

