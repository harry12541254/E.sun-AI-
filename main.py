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


# Step 6 每張卡號在不同商品類別 mcc 下的交易情況

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


# In[11]:


# Step 7: 分析每張卡號在不同 mchno下的交易情況

# 計算每張卡號在不同商戶編碼下的交易數量
df_copy['transactions_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['txkey'].transform('count')

# 計算每個卡號每個商戶編碼的總交易金額
df_copy['mchno_total_amount'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform('sum')

# 計算每個卡號在不同商戶編碼下的交易金額的變異數
df_copy['variance_transaction_amount_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform('var')

# 使用之前定義的mad函數計算每個卡號在不同商戶編碼下的交易金額的MAD
df_copy['mad_transaction_amount_per_mchno'] = df_copy.groupby(['cano', 'mchno'])['conam'].transform(mad)


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


train_data = pd.concat([old_train,new_train],sort=False)
new_train_data = df_copy.merge(train_data[['txkey']], on='txkey', how='inner')



# In[16]:


columns_to_drop = ['label', 'txkey', 'chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change', 'unusual_3dsmk']

# 從數據框中刪除指定的列
X = new_train_data.drop(columns=columns_to_drop)
y = new_train_data['label']


# In[17]:


del old_train
del train_data


# In[18]:


X = new_train_data.drop(columns=columns_to_drop)
for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4, 
    random_state=40, 
    stratify=y
)

train_pool = Pool(X_train, y_train,cat_features=cat_features)
test_pool = Pool(X_test, y_test,cat_features=cat_features)


# In[19]:


feature_list = X_train.columns.tolist()
print(feature_list)


# In[61]:


catboost_model = CatBoostClassifier(
    iterations=6000,  
    learning_rate=0.035,
    depth=7,
    loss_function='Logloss',
    eval_metric='F1', 
    early_stopping_rounds=800,
    random_seed=42,
    verbose=100,
    l2_leaf_reg=3,
    leaf_estimation_iterations=10,
#     colsample_bylevel=0.8,
#     has_time = 'True',
#     subsample=0.85,
#     auto_class_weights='Balanced',
    max_ctr_complexity=10,
    task_type='GPU',
    scale_pos_weight = 9.4835, # 9.45
#     boosting_type='Ordered',
    random_strength=3,
#     rsm=0.8,
    grow_policy='Lossguide'
)

catboost_model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)
"""
0:	learn: 0.8280543	test: 0.8380455	best: 0.8380455 (0)	total: 9.36s	remaining: 7h 47m 46s
100:	learn: 0.9106877	test: 0.9144912	best: 0.9144912 (100)	total: 9m 41s	remaining: 4h 37m 59s
200:	learn: 0.9265892	test: 0.9280466	best: 0.9280466 (200)	total: 18m 48s	remaining: 4h 22m 1s
300:	learn: 0.9350423	test: 0.9359664	best: 0.9359664 (300)	total: 27m 58s	remaining: 4h 10m 48s
400:	learn: 0.9410402	test: 0.9401827	best: 0.9401827 (400)	total: 37m 12s	remaining: 4h 1m 7s
500:	learn: 0.9461936	test: 0.9442636	best: 0.9444322 (496)	total: 46m 11s	remaining: 3h 50m 24s
600:	learn: 0.9496379	test: 0.9468092	best: 0.9468133 (599)	total: 56m 41s	remaining: 3h 46m 19s
700:	learn: 0.9534834	test: 0.9488793	best: 0.9488793 (700)	total: 1h 7m 17s	remaining: 3h 40m 40s
800:	learn: 0.9563964	test: 0.9504029	best: 0.9504488 (797)	total: 1h 17m 58s	remaining: 3h 34m 2s
900:	learn: 0.9597543	test: 0.9518056	best: 0.9518931 (897)	total: 1h 27m 41s	remaining: 3h 24m 17s
1000:	learn: 0.9622792	test: 0.9533672	best: 0.9533672 (1000)	total: 1h 37m 6s	remaining: 3h 13m 56s
1100:	learn: 0.9648807	test: 0.9552601	best: 0.9552601 (1100)	total: 1h 47m 23s	remaining: 3h 5m 14s
1200:	learn: 0.9666886	test: 0.9565251	best: 0.9565251 (1200)	total: 1h 57m 34s	remaining: 2h 56m 7s
1300:	learn: 0.9689852	test: 0.9572601	best: 0.9572724 (1299)	total: 2h 6m 56s	remaining: 2h 45m 46s
1400:	learn: 0.9704295	test: 0.9582096	best: 0.9582510 (1398)	total: 2h 16m 23s	remaining: 2h 35m 39s
1500:	learn: 0.9717821	test: 0.9587664	best: 0.9589070 (1489)	total: 2h 25m 56s	remaining: 2h 25m 45s
1600:	learn: 0.9728419	test: 0.9594720	best: 0.9595670 (1591)	total: 2h 34m 50s	remaining: 2h 15m 18s
1700:	learn: 0.9740917	test: 0.9598222	best: 0.9599255 (1697)	total: 2h 44m 5s	remaining: 2h 5m 18s
1800:	learn: 0.9753392	test: 0.9604163	best: 0.9604740 (1797)	total: 2h 53m 25s	remaining: 1h 55m 27s
1900:	learn: 0.9766444	test: 0.9611915	best: 0.9611915 (1900)	total: 3h 2m 29s	remaining: 1h 45m 30s
2000:	learn: 0.9774685	test: 0.9613648	best: 0.9614636 (1941)	total: 3h 11m 29s	remaining: 1h 35m 36s
2100:	learn: 0.9785009	test: 0.9616618	best: 0.9616618 (2100)	total: 3h 20m 42s	remaining: 1h 25m 52s
2200:	learn: 0.9793928	test: 0.9617156	best: 0.9618475 (2184)	total: 3h 29m 58s	remaining: 1h 16m 13s
2300:	learn: 0.9801086	test: 0.9620622	best: 0.9620622 (2300)	total: 3h 39m 20s	remaining: 1h 6m 37s
2400:	learn: 0.9811091	test: 0.9621739	best: 0.9623057 (2363)	total: 3h 48m 14s	remaining: 56m 56s
2500:	learn: 0.9817564	test: 0.9624837	best: 0.9625496 (2496)	total: 3h 57m 15s	remaining: 47m 20s
2600:	learn: 0.9824483	test: 0.9628180	best: 0.9628180 (2598)	total: 4h 6m 15s	remaining: 37m 46s
2700:	learn: 0.9830343	test: 0.9630042	best: 0.9632057 (2668)	total: 4h 15m 17s	remaining: 28m 15s
2800:	learn: 0.9835630	test: 0.9633347	best: 0.9633759 (2795)	total: 4h 24m 25s	remaining: 18m 47s
2900:	learn: 0.9843055	test: 0.9634096	best: 0.9634837 (2893)	total: 4h 33m 31s	remaining: 9m 20s
2999:	learn: 0.9849618	test: 0.9635996	best: 0.9637851 (2988)	total: 4h 42m 37s	remaining: 0us

bestTest = 0.963785117
bestIteration = 2988

Shrink model to first 2989 iterations.
"""


# In[62]:


y_pred = catboost_model.predict(test_pool)

precision = precision_score(y_test, y_pred, average='binary', pos_label=1)

print(f"Precision: {precision}")


# In[63]:


recall = recall_score(y_test, y_pred, average='binary', pos_label=1)

print(f"Recall: {recall}")



# In[60]:


# Get the feature names from the preprocessed dataset

feature_names =['locdt', 'loctm', 'contp', 'etymd', 'mchno', 
                'acqic', 'mcc', 'conam', 'ecfg', 'insfg', 'flam1', 
                'stocn', 'scity', 'stscd', 'hcefg', 'csmcu', 'csmam', 
                'flg_3dsmk', 'card_transaction_count', 'customer_total_transactions', 
                'card_transaction_ratio_before_30', 'card_transaction_ratio_after_30', 
                'ratio_change', 'min_daily_trans', 'max_daily_trans', 'daily_transactions', 
                'normalized_trans_freq', 'normalized_daily_amount', 'difference_seconds', 
                'avg_interval', 'std_interval', 'transactions_per_mcc', 'mcc_total_amount', 
                'variance_transaction_amount_per_mcc', 'mad_transaction_amount_per_mcc', 
                'transactions_per_mchno', 'mchno_total_amount', 'variance_transaction_amount_per_mchno', 
                'mad_transaction_amount_per_mchno', 'hcefg_change', 'unusual_3dsmk', 'city_change', 
                'country_change', 'hour', 'loctm_seconds']


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

# In[64]:


new_val_data = df_copy.merge(example[['txkey']], on='txkey', how='inner')
new_val_data = new_val_data.set_index('txkey')


# In[67]:


columns_to_drop = ['chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change', 'unusual_3dsmk']

X = new_val_data.drop(columns=columns_to_drop)

for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
test_pool = Pool(X, cat_features=cat_features)


# In[68]:


y_pred = catboost_model.predict(test_pool).astype(int)
new_val_data['pred']= y_pred
new_val_data =new_val_data.reset_index()
output_df = new_val_data[['txkey', 'pred']].set_index('txkey')

output_df


# In[70]:


example = example.drop_duplicates(subset='txkey')
df2_sorted = example[['txkey']].merge(output_df, on='txkey', how='left')
df2_sorted


# In[2]:


output_filename = 'dataset_2nd/predictions_secondround.csv'
df2_sorted.to_csv(output_filename, index='True')


# In[ ]:




