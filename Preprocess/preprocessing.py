#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""

這份檔案要用來執行特徵工程部分，主要利用pandas來進行處理


input：四份資料夾內文件，training、public、public_processed、private_1_processed

output：特徵處理完的training dataset(processed_data.parquet)、validation dataset(val_data.parquet)


"""


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


# 載入資料 
old_train = pd.read_csv('dataset_1st/training.csv')
new_train = pd.read_csv('dataset_2nd/public.csv')

old_val = pd.read_csv('dataset_1st/public_processed.csv')
new_val = pd.read_csv('dataset_2nd/private_1_processed.csv')

example = pd.read_csv('dataset_1st/31_範例繳交檔案.csv')

df = pd.concat([old_train, new_train, new_val],sort=False)
df_copy = df.copy()


# Step 1 計算交易活動，計算特定卡在前30天消費所佔的次數比率與之後的比率，計算之間的變化率

# 計算每張卡在其持有者所有交易中的活動次數
card_activity_count = df_copy.groupby(['chid', 'cano'])['txkey'].count()
card_activity_count = card_activity_count.rename('card_transaction_count').reset_index()
df_copy = df_copy.merge(card_activity_count, on=['chid', 'cano'], how='left')

# 計算每個持卡人的總交易次數
customer_total_transactions = df_copy.groupby('chid')['txkey'].count()
customer_total_transactions = customer_total_transactions.rename('customer_total_transactions').reset_index()

# 將每個持卡人的總交易次數合併到主資料集
df_copy = df_copy.merge(customer_total_transactions, on='chid', how='left')

# 篩選前30天的交易數據
df_copy_before_30 = df_copy[df_copy['locdt'] <= 30]

# 計算前30天的交易比例
card_activity_before_30 = df_copy_before_30.groupby(['chid', 'cano'])['txkey'].count().reset_index()
total_transactions_before_30 = df_copy_before_30.groupby('chid')['txkey'].count().reset_index()
card_ratio_before_30 = card_activity_before_30.merge(total_transactions_before_30, on='chid')
card_ratio_before_30['card_transaction_ratio_before_30'] = card_ratio_before_30['txkey_x'] / card_ratio_before_30['txkey_y']

# 篩選後30天的交易數據
df_copy_after_30 = df_copy[df_copy['locdt'] > 30]

# 計算後30天的交易比例
card_activity_after_30 = df_copy_after_30.groupby(['chid', 'cano'])['txkey'].count().reset_index()
total_transactions_after_30 = df_copy_after_30.groupby('chid')['txkey'].count().reset_index()
card_ratio_after_30 = card_activity_after_30.merge(total_transactions_after_30, on='chid')
card_ratio_after_30['card_transaction_ratio_after_30'] = card_ratio_after_30['txkey_x'] / card_ratio_after_30['txkey_y']

# 計算前後30天的比例變化率
card_ratio_change = card_ratio_before_30.merge(card_ratio_after_30, on=['chid', 'cano'])
card_ratio_change['ratio_change'] = (card_ratio_change['card_transaction_ratio_after_30'] - card_ratio_change['card_transaction_ratio_before_30']) / card_ratio_change['card_transaction_ratio_before_30']

# 已經計算好的 card_ratio_before_30, card_ratio_after_30, 和 card_ratio_change

# 將重要的欄位合併到 df_copy
df_copy = df_copy.merge(card_ratio_before_30[['chid', 'cano', 'card_transaction_ratio_before_30']], on=['chid', 'cano'], how='left')
df_copy = df_copy.merge(card_ratio_after_30[['chid', 'cano', 'card_transaction_ratio_after_30']], on=['chid', 'cano'], how='left')
df_copy = df_copy.merge(card_ratio_change[['chid', 'cano', 'ratio_change']], on=['chid', 'cano'], how='left')


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
df_copy['normalized_trans_freq'] = df_copy['normalized_trans_freq'].fillna(-1)


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


# Step 4 計算上次刷卡時間(秒數)

def impute_time_zero(x):
    x = str(int(x)).zfill(6)
    return datetime.datetime.strptime(x, "%H%M%S").time()

# Apply the function to convert 'loctm' to time objects
df_copy['loctm'] = df_copy['loctm'].apply(impute_time_zero)

# Sort the DataFrame based on card number and transaction datetime
sorted_df = df_copy.sort_values(by=['cano', 'locdt', 'loctm'])

# Group by 'cano' and shift the 'locdt' and 'loctm' to get the previous transaction's date and time
sorted_df['prev_locdt'] = sorted_df.groupby('cano')['locdt'].shift(1)
sorted_df['prev_loctm'] = sorted_df.groupby('cano')['loctm'].shift(1)

# Calculate the seconds since midnight for 'loctm' and 'prev_loctm'
sorted_df['loctm_seconds'] = sorted_df['loctm'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
sorted_df['prev_loctm_seconds'] = sorted_df['prev_loctm'].apply(
    lambda x: x.hour * 3600 + x.minute * 60 + x.second if pd.notnull(x) else 0)

# Calculate the difference in days and convert to seconds, then add the difference in seconds
sorted_df['difference_seconds'] = (sorted_df['locdt'] - sorted_df['prev_locdt']) * 86400 + \
                                  (sorted_df['loctm_seconds'] - sorted_df['prev_loctm_seconds'])

# For the first transaction, we don't have a previous transaction time, so we set it to -1
sorted_df['difference_seconds'] = sorted_df['difference_seconds'].fillna(-1)

# Merge the result back into the original DataFrame
df_copy = df_copy.merge(sorted_df[['txkey', 'difference_seconds']], on='txkey', how='left')

# Display the head of the DataFrame to confirm the changes
df_copy.head()


# Step 5 計算每個卡號刷卡的每筆之間的時間間隔的平均和標準差

def calculate_transaction_intervals(group):
    """
    計算信用卡交易時間間隔的平均值和標準差。

    :param group: Grouped DataFrame by 'cano'.
    :return: Tuple with average and standard deviation of transaction intervals in seconds.
    """
    # 計算日期差異（轉換為秒）和時間差異
    date_diffs = group['locdt'].diff().fillna(0) * 86400
    time_diffs = group['loctm_seconds'].diff().fillna(0)

    # 總時間差異
    total_diffs = date_diffs + time_diffs

    # 排除第一筆交易（因為它沒有前一筆交易可以比較）
    total_diffs = total_diffs[1:]

    # 計算平均值和標準差
    avg_interval = total_diffs.mean() if not total_diffs.empty else -1
    std_interval = total_diffs.std() if not total_diffs.empty else -1

    return avg_interval, std_interval

# 應用函數並創建新的 DataFrame
intervals_df = sorted_df.groupby('cano').apply(calculate_transaction_intervals)
intervals_df = pd.DataFrame(intervals_df.tolist(), index=intervals_df.index).reset_index()
intervals_df.columns = ['cano', 'avg_interval', 'std_interval']

# 合併到原始 DataFrame
df_copy = df_copy.merge(intervals_df, on='cano', how='left')



# Step 6 每張卡號在不同商品類別（mcc）下的交易情況

# 計算每張卡號在不同商品類別下的交易數量
df_copy['transactions_per_mcc'] = df_copy.groupby(['cano', 'mcc'])['txkey'].transform('count')

# 計算每個卡號每種商品類別的總交易金額
df_copy['mcc_total_amount'] = df_copy.groupby(['cano', 'mcc'])['conam'].transform('sum')

# 計算每個卡號在不同商品類別下的交易金額的變異數
df_copy['variance_transaction_amount_per_mcc'] = df_copy.groupby(['cano', 'mcc'])['conam'].transform('var')

# 計算每個卡號在不同商品類別下的交易金額的MAD
def mad(x):
    return (x - x.median()).abs().median()

df_copy['mad_transaction_amount_per_mcc'] = df_copy.groupby(['cano', 'mcc'])['conam'].transform(mad)



# Step 7: 分析每張卡號在不同 mchno下的交易情況

# 計算每張卡號在不同商戶編碼下的交易數量
df_copy['transactions_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['txkey'].transform('count')

# 計算每個卡號每個商戶編碼的總交易金額
df_copy['mchno_total_amount'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform('sum')

# 計算每個卡號在不同商戶編碼下的交易金額的變異數
df_copy['variance_transaction_amount_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform('var')

# 使用之前定義的mad函數計算每個卡號在不同商戶編碼下的交易金額的MAD
df_copy['mad_transaction_amount_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform(mad)



# Step 8 觀察交易註記變化

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



# Step 9 觀察交易地點變化

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



# Step 10 把loctom拆成鐘點 

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


train_data = pd.concat([old_train,new_train],sort=False)
new_train_data = df_copy.merge(train_data[['txkey']], on='txkey', how='inner')
new_val_data = df_copy.merge(example[['txkey']], on='txkey', how='inner')

new_train_data.to_parquet('processed_data.parquet')
new_val_data.to_parquet('val_data.parquet')






