#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from scipy.stats import entropy


# In[2]:


from catboost import CatBoostClassifier

# 載入模型路徑
model_path = 'dataset_2nd/model.cbm'

# Create an instance of the CatBoostClassifier
loaded_model = CatBoostClassifier()

# Load the model from the file
loaded_model.load_model(model_path, format='cbm')


# In[3]:


new_val_data = pd.read_parquet('val_data.parquet')
example = pd.read_csv('dataset_1st/31_範例繳交檔案.csv')
new_val_data = new_val_data.set_index('txkey')


# In[4]:


columns_to_drop = ['chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change']

X = new_val_data.drop(columns=columns_to_drop)

for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
test_pool = Pool(X, cat_features=cat_features)


# In[5]:


y_pred = loaded_model.predict(test_pool).astype(int)
new_val_data['pred']= y_pred
new_val_data =new_val_data.reset_index()

output_df = new_val_data[['txkey', 'pred']].set_index('txkey')
example = example.drop_duplicates(subset='txkey')

df2_sorted = example[['txkey']].merge(output_df, on='txkey', how='left')
df2_sorted = df2_sorted.set_index('txkey')


# In[6]:


output_filename = 'dataset_2nd/predictions_secondround.csv'
df2_sorted.to_csv(output_filename, index='True')


# In[7]:


df2_sorted.value_counts()


# In[11]:


df2_sorted.value_counts()


# In[ ]:




