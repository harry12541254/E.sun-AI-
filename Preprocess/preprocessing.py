#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# # 特徵處理

# In[ ]:


old_train = pd.read_csv('dataset_1st/training.csv')
new_train = pd.read_csv('dataset_2nd/public.csv')

old_val = pd.read_csv('dataset_1st/public_processed.csv')
new_val = pd.read_csv('dataset_2nd/private_1_processed.csv')

example = pd.read_csv('dataset_1st/31_範例繳交檔案.csv')

df = pd.concat([old_train, new_train, new_val],sort=False)

pd.set_option('display.max_columns',500)


# In[ ]:


# Step 1 計算交易活動，計算每張卡的靜態交易次數以及這些次數在該客戶所有交易中的比例
sbctn = df_copy.groupby(by=['chid', 'cano'])['txkey'].count().rename('same_chid_different_cano_trade_number').reset_index()
df_copy = df_copy.merge(sbctn, on=['chid', 'cano'], how='left')
total_transactions_per_chid = df_copy.groupby('chid')['txkey'].count().rename('total_transactions_per_chid').reset_index()

# 計算每個卡的交易次數佔比
df_copy = df_copy.merge(total_transactions_per_chid, on='chid', how='left')
df_copy['chid_cano_trade_number_ratio'] = df_copy['same_chid_different_cano_trade_number'] / df_copy['total_transactions_per_chid']


# In[ ]:


# Step 2 計算每個卡號每天的交易頻率並進行正規化處理
transactions_per_day = df_copy.groupby(['cano', 'locdt']).size().reset_index(name='daily_transactions')

# 計算最大最小值
min_max_transactions = transactions_per_day.groupby('cano')['daily_transactions'].agg(['min', 'max']).reset_index()
min_max_transactions.columns = ['cano', 'min_daily_trans', 'max_daily_trans']
df_copy = df_copy.merge(min_max_transactions, on='cano', how='left')

# 正規化
df_copy = df_copy.merge(transactions_per_day, on=['cano', 'locdt'], how='left')
df_copy['normalized_trans_freq'] = df_copy.apply(lambda x: (x['daily_transactions'] - x['min_daily_trans']) / (x['max_daily_trans'] - x['min_daily_trans']) if x['max_daily_trans'] != x['min_daily_trans'] else 0, axis=1)

# fill na
df_copy['normalized_trans_freq'] = df_copy['normalized_trans_freq'].fillna(0)
df_copy.head()


# In[ ]:


# Step 3 計算每個卡號每天的交易金額並進行正規化處理
daily_amount_sum = df_copy.groupby(['cano', 'locdt'])['conam'].sum().reset_index(name='daily_amount_sum')

# 計算最大最小值
min_max_daily_amount = daily_amount_sum.groupby('cano')['daily_amount_sum'].agg(['min', 'max']).reset_index()
min_max_daily_amount.columns = ['cano', 'min_daily_amount', 'max_daily_amount']
daily_amount_sum = daily_amount_sum.merge(min_max_daily_amount, on='cano', how='left')

# 正規化
daily_amount_sum['normalized_daily_amount'] = daily_amount_sum.apply(
    lambda x: (x['daily_amount_sum'] - x['min_daily_amount']) / (x['max_daily_amount'] - x['min_daily_amount']) 
    if x['max_daily_amount'] != x['min_daily_amount'] else 0, 
    axis=1)

df_copy = df_copy.merge(daily_amount_sum[['cano', 'locdt', 'normalized_daily_amount']], on=['cano', 'locdt'], how='left')


# In[ ]:


# Step 4 利用秒數計算卡號的每筆資料與上次交易時間差距
def impute_time_zero(x):
    x = str(int(x))
    if len(x) == 1:
        x = '00000' + x
    elif len(x) == 2:
        x = '0000' + x
    elif len(x) == 3:
        x = '000' + x
    elif len(x) == 4:
        x = '00' + x
    elif len(x) == 5:
        x = '0' + x
    x = datetime.datetime.strptime(x, "%H%M%S").time()
    return x


df_copy['loctm'] = df_copy['loctm'].apply(impute_time_zero)
sorted_df = df_copy.sort_values(by=['cano', 'locdt', 'loctm'])
sorted_df['shift_locdt'] = sorted_df['locdt'].shift(1).fillna(0)
sorted_df['shift_loctm'] = sorted_df['loctm'].shift(1).fillna(datetime.time(0, 0))
sorted_df['loctm_seconds'] = sorted_df['loctm'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
sorted_df['shift_loctm_seconds'] = sorted_df['shift_loctm'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# 計算日期差異
sorted_df['difference_locdt'] = sorted_df['locdt'] - sorted_df['shift_locdt']
sorted_df['difference_loctm'] = sorted_df['loctm_seconds'] - sorted_df['shift_loctm_seconds']

# 天數轉換秒數
sorted_df['total_difference_second'] = 86400 * sorted_df['difference_locdt'] + sorted_df['difference_loctm']

# 將第一筆交易設為 -1
sorted_df['is_first_transaction'] = sorted_df.groupby('cano').cumcount() == 0
sorted_df.loc[sorted_df['is_first_transaction'], 'total_difference_second'] = -1

df_copy = df_copy.merge(sorted_df[['txkey', 'total_difference_second']], on='txkey', how='left')
df_copy.head()


# In[ ]:


# Step 5 計算每個卡號計算不同日子間交易頻率的平均值和標準差 觀察交易模式

def cal_difference_day_transaction_frequency_and_std(x):
    sorted_locdt = sorted(x['locdt'])
    sorted_loctm = sorted(x['loctm_seconds'])
    minus_locdt = []
    minus_loctm = []

    for i in range(len(sorted_locdt) - 1):
        minus_locdt.append(sorted_locdt[i+1] - sorted_locdt[i])
        minus_loctm.append(sorted_loctm[i+1] - sorted_loctm[i])

    if not minus_locdt:
        return -1, -1

    minus_locdt = np.array(minus_locdt) * 86400
    total_seconds = minus_locdt + np.array(minus_loctm)
    mean_frequency = np.mean(total_seconds) if total_seconds.size > 0 else -1
    std_frequency = np.std(total_seconds) if total_seconds.size > 0 else -1
    return mean_frequency, std_frequency

ddtf_std = sorted_df.groupby('cano').apply(cal_difference_day_transaction_frequency_and_std)
ddtf_std = pd.DataFrame(ddtf_std.tolist(), index=ddtf_std.index).reset_index()

ddtf_std.columns = ['cano', 'difference_day_transaction_frequency', 'std_day_transaction_frequency']
df_copy = df_copy.merge(ddtf_std, on='cano', how='left')
df_copy.head()


# In[ ]:


# Step 6 每張卡號在不同商品類別（mcc）下的交易情況

sbmstn = df_copy.groupby(by=['cano', 'mcc'])['txkey'].count().rename('same_cano_mcc_separate_trade_number').reset_index()
df_copy = df_copy.merge(sbmstn, on=['cano', 'mcc'], how='left')

# 計算每個卡號每種商品類別的總交易金額
mcc_total_amount = df_copy.groupby(['cano', 'mcc'])['conam'].sum().rename('mcc_total_amount').reset_index()
df_copy = df_copy.merge(mcc_total_amount, on=['cano', 'mcc'], how='left')


# 交易商品的標準差
std_conam = df_copy.groupby(['cano', 'mcc'])['conam'].std().rename('std_conam').reset_index()
df_copy = df_copy.merge(std_conam, on=['cano', 'mcc'], how='left')

df_copy.head()


# In[ ]:


# Step 7 觀察交易註記變化

df_copy = df_copy.sort_values(by=['cano', 'locdt', 'loctm'])

# initialize 
df_copy['hcefg_change'] = 0  
df_copy['unusual_3dsmk'] = 0  

# 判斷上次交易
previous_hcefg = df_copy.groupby('cano')['hcefg'].shift()
previous_3dsmk = df_copy.groupby('cano')['flg_3dsmk'].shift()

# 判斷卡號是否相同
df_copy['same_cano'] = df_copy['cano'] == df_copy['cano'].shift()

# 若支付型態改變則標註為 1
df_copy.loc[df_copy['same_cano'], 'hcefg_change'] = (df_copy['hcefg'] != previous_hcefg).astype(int)

# 若3D安全驗證改變則標註為 1
df_copy.loc[df_copy['same_cano'], 'unusual_3dsmk'] = ((previous_3dsmk == 0) & (df_copy['flg_3dsmk'] == 1)).astype(int)

df_copy.drop(columns=['same_cano'], inplace=True)

df_copy[['cano', 'locdt', 'loctm', 'hcefg', 'hcefg_change', 'flg_3dsmk', 'unusual_3dsmk']].head()


# In[ ]:


# Step 8 觀察交易地點變化

df_copy = df_copy.sort_values(by=['cano', 'locdt'])

# initialize 
df_copy['city_change'] = 0
df_copy['country_change'] = 0  

previous_locations = df_copy.groupby('cano')[['scity', 'stocn']].shift()

# 判斷卡號是否相同
df_copy['same_cano'] = df_copy['cano'] == df_copy['cano'].shift()

# 若支付城市改變則標註為 1
df_copy['city_change'] = ((df_copy['scity'] != previous_locations['scity']) & df_copy['same_cano']).astype(int)

# 若支付國家改變則標註為 1
df_copy['country_change'] = ((df_copy['stocn'] != previous_locations['stocn']) & df_copy['same_cano']).astype(int)

df_copy.drop(columns=['same_cano'], inplace=True)


# In[ ]:


# Step 9 把loctom拆成鐘點 

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

def time_to_hour(t):
    return t.hour

def time_to_string(t):
    return t.strftime("%H%M%S")

df_copy['hour'] = df_copy['loctm'].apply(time_to_hour)
df_copy['loctm_seconds'] = df_copy['loctm'].apply(time_to_seconds)
df_copy['loctm'] = df_copy['loctm'].apply(time_to_string)


# #  合併要訓練的資料

# In[ ]:


train_data = pd.concat([old_train,new_train],sort=False)
new_train_data = df_copy.merge(train_data[['txkey']], on='txkey', how='inner')
new_val_data = df_copy.merge(example[['txkey']], on='txkey', how='inner')


# In[ ]:


new_train_data.to_parquet('processed_data.parquet')
new_val_data.to_parquet('val_data.parquet')


# In[ ]:




