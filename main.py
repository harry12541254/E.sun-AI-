#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import datetime
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from scipy.stats import entropy


# In[2]:


old_train = pd.read_csv('dataset_1st/training.csv')
new_train = pd.read_csv('dataset_2nd/public.csv')

old_val = pd.read_csv('dataset_1st/public_processed.csv')
new_val = pd.read_csv('dataset_2nd/private_1_processed.csv')

example = pd.read_csv('dataset_1st/31_範例繳交檔案.csv')

df = pd.concat([old_train, new_train, new_val],sort=False)


# In[3]:


df_copy = df.copy()


# In[4]:


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


# In[5]:


# 已經計算好的 card_ratio_before_30, card_ratio_after_30, 和 card_ratio_change

# 將重要的欄位合併到 df_copy
df_copy = df_copy.merge(card_ratio_before_30[['chid', 'cano', 'card_transaction_ratio_before_30']], on=['chid', 'cano'], how='left')
df_copy = df_copy.merge(card_ratio_after_30[['chid', 'cano', 'card_transaction_ratio_after_30']], on=['chid', 'cano'], how='left')
df_copy = df_copy.merge(card_ratio_change[['chid', 'cano', 'ratio_change']], on=['chid', 'cano'], how='left')


# In[6]:


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



# In[7]:


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


# In[8]:


# Step 4 計算上次刷卡時間(秒數)

def impute_time_zero(x):
    x = str(int(x)).zfill(6)
    return datetime.datetime.strptime(x, "%H%M%S").time()

df_copy['loctm'] = df_copy['loctm'].apply(impute_time_zero)

sorted_df = df_copy.sort_values(by=['cano', 'locdt', 'loctm'])

sorted_df['prev_locdt'] = sorted_df.groupby('cano')['locdt'].shift(1)
sorted_df['prev_loctm'] = sorted_df.groupby('cano')['loctm'].shift(1)

sorted_df['loctm_seconds'] = sorted_df['loctm'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
sorted_df['prev_loctm_seconds'] = sorted_df['prev_loctm'].apply(
    lambda x: x.hour * 3600 + x.minute * 60 + x.second if pd.notnull(x) else 0)

sorted_df['difference_seconds'] = (sorted_df['locdt'] - sorted_df['prev_locdt']) * 86400 + \
                                  (sorted_df['loctm_seconds'] - sorted_df['prev_loctm_seconds'])

sorted_df['difference_seconds'] = sorted_df['difference_seconds'].fillna(-1)

df_copy = df_copy.merge(sorted_df[['txkey', 'difference_seconds']], on='txkey', how='left')

df_copy.head()


# In[9]:


# Step 5 計算每個卡號刷卡的每筆之間的時間間隔的平均和標準差

def calculate_transaction_intervals(group):
    """
    計算信用卡交易時間間隔的平均值和標準差。

    :param group: Grouped DataFrame by 'cano'.
    :return: Tuple with average and standard deviation of transaction intervals in seconds.
    """
    # 計算日期差異（秒）和時間差異
    date_diffs = group['locdt'].diff().fillna(0) * 86400
    time_diffs = group['loctm_seconds'].diff().fillna(0)

    # 總時間差異
    total_diffs = date_diffs + time_diffs

    # 排除第一筆交易
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



# In[10]:


# Step 6 每張卡號在不同商品類別（mcc）下的交易情況

# 一次計算所有統計數據
grouped = df_copy.groupby(['cano', 'mcc'])

# 創建一個新的數據框來存儲結果
stats = pd.DataFrame({
    'transactions_per_mcc': grouped['txkey'].count(),
    'mcc_total_amount': grouped['conam'].sum(),
    'variance_transaction_amount_per_mcc': grouped['conam'].var()
})

# 計算MAD
def mad(series):
    return (series - series.median()).abs().median()

stats['mad_transaction_amount_per_mcc'] = grouped['conam'].apply(mad)

# 重置索引以便後續合併
stats.reset_index(inplace=True)

# 合併計算結果回原始數據框
df_copy = df_copy.merge(stats, on=['cano', 'mcc'], how='left')


# In[11]:


# Step 7: 分析每張卡號在不同 mchno下的交易情況

# 一次計算所有統計數據
grouped = df_copy.groupby(['cano', 'mchno'])

# 創建一個新的數據框來存儲結果
stats = pd.DataFrame({
    'transactions_per_mchno': grouped['txkey'].count(),
    'mchno_total_amount': grouped['conam'].sum(),
    'variance_transaction_amount_per_mchno': grouped['conam'].var()
})

# 計算MAD
stats['mad_transaction_amount_per_mchno'] = grouped['conam'].apply(mad)

# 重置索引以便後續合併
stats.reset_index(inplace=True)

# 合併計算結果回原始數據框
df_copy = df_copy.merge(stats, on=['cano', 'mchno'], how='left')


# In[12]:


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


# In[13]:


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


# In[14]:


train_data = pd.concat([old_train,new_train],sort=False)
new_train_data = df_copy.merge(train_data[['txkey']], on='txkey', how='inner')


# In[15]:


columns_to_drop = ['label', 'txkey', 'chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change']

# 從數據框中刪除指定的列
X = new_train_data.drop(columns=columns_to_drop)
y = new_train_data['label']


# In[16]:


del old_train
del train_data


# In[17]:


X = new_train_data.drop(columns=columns_to_drop)
for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.35, 
    random_state=40, 
    stratify=y
)

train_pool = Pool(X_train, y_train,cat_features=cat_features)
test_pool = Pool(X_test, y_test,cat_features=cat_features)


# In[18]:


feature_list = X_train.columns.tolist()
print(feature_list)


# In[ ]:


"""
註記：若電腦為MAC系列，則須無法使用 task_type='GPU'，可將其註解並且開啟subsample，接受另外兩個參數再進行訓練，超參數皆無須調整

"""
catboost_model = CatBoostClassifier(
    iterations=5000,  
    learning_rate=0.0325,
    depth=7,
    loss_function='Logloss',
    eval_metric='F1',  
    early_stopping_rounds=800,
    random_seed=42,
    verbose=100,
    l2_leaf_reg=3,
    leaf_estimation_iterations=10,
#     colsample_bylevel=0.8,
#     subsample=0.85,
    max_ctr_complexity=10,
    task_type='GPU',
    scale_pos_weight = 9.484, # 9.45
    random_strength=3,
    grow_policy='Lossguide'
)

catboost_model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)


# In[ ]:


y_pred = catboost_model.predict(test_pool)

precision = precision_score(y_test, y_pred, average='binary', pos_label=1)

print(f"Precision: {precision}")


# In[ ]:


recall = recall_score(y_test, y_pred, average='binary', pos_label=1)

print(f"Recall: {recall}")



# In[ ]:


# Get the feature names from the preprocessed dataset

feature_names =['locdt', 'loctm', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 
                'conam', 'ecfg', 'insfg', 'flam1', 'stocn', 'scity', 'stscd', 
                'hcefg', 'csmcu', 'csmam', 'flg_3dsmk', 'card_transaction_count', 
                'customer_total_transactions', 'card_transaction_ratio_before_30', 
                'card_transaction_ratio_after_30', 'ratio_change', 'min_daily_trans', 
                'max_daily_trans', 'daily_transactions', 'normalized_trans_freq', 
                'normalized_daily_amount', 'difference_seconds', 'avg_interval', 
                'std_interval', 'transactions_per_mcc_x', 'mcc_total_amount_x', 
                'variance_transaction_amount_per_mcc_x', 'transactions_per_mcc_y', 
                'mcc_total_amount_y', 'variance_transaction_amount_per_mcc_y', 
                'mad_transaction_amount_per_mcc', 'transactions_per_mchno', 'mchno_total_amount', 
                'variance_transaction_amount_per_mchno', 'mad_transaction_amount_per_mchno', 
                'city_change', 'country_change', 'hour', 'loctm_seconds']

feature_importances = catboost_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# # 合併資料輸出

# In[ ]:


new_val_data = df_copy.merge(example[['txkey']], on='txkey', how='inner')
new_val_data = new_val_data.set_index('txkey')


# In[ ]:


columns_to_drop = ['chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change', 'unusual_3dsmk']

X = new_val_data.drop(columns=columns_to_drop)

for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
test_pool = Pool(X, cat_features=cat_features)


# In[ ]:


y_pred = catboost_model.predict(test_pool).astype(int)
new_val_data['pred']= y_pred
new_val_data =new_val_data.reset_index()

output_df = new_val_data[['txkey', 'pred']].set_index('txkey')
example = example.drop_duplicates(subset='txkey')

df2_sorted = example[['txkey']].merge(output_df, on='txkey', how='left')
df2_sorted = df2_sorted.set_index('txkey')


# In[ ]:


output_filename = 'dataset_2nd/predictions_secondround.csv'
df2_sorted.to_csv(output_filename, index='True')

