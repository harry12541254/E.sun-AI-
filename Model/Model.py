#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
這份檔案要用來執行訓練行的部分，主要利用catboost來訓練模型，可透過參數利用GPU進行運算

input：特徵處理完的training dataset(processed_data.parquet)、validation dataset(val_data.parquet)

output：最終final_prediction 


"""


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


# # 訓練模型

# In[ ]:


new_train_data = pd.read_parquet('processed_data.parquet')
new_val_data = pd.read_parquet('val_data.parquet')
example = pd.read_csv('dataset_1st/31_範例繳交檔案.csv')





columns_to_drop = ['label', 'txkey', 'chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change', 'unusual_3dsmk']

X = new_train_data.drop(columns=columns_to_drop)
y = new_train_data['label']


# In[ ]:


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


# In[ ]:


"""
註記：若電腦為MAC系列，則須無法使用 task_type='GPU'，可將其註解並且開啟subsample，接受另外兩個參數再進行訓練，超參數皆無須調整

"""
catboost_model = CatBoostClassifier(
    iterations=6000,  
    learning_rate=0.035,
    depth=7,
    loss_function='Logloss',
    eval_metric='F1',  
    early_stopping_rounds=500,
    random_seed=42,
    verbose=100,
    l2_leaf_reg=3,
    leaf_estimation_iterations=10,
#     colsample_bylevel=0.8,
#     subsample=0.85,
    max_ctr_complexity=10,
    task_type='GPU',
    scale_pos_weight =  9.483525,
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


recall = recall_score(y_test, y_pred, average='binary', pos_label=1)

print(f"Recall: {recall}")



# 查看特徵

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

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print(feature_importance_df)


# # 預測和輸出資料

# In[ ]:


new_val_data = new_val_data.set_index('txkey')


# In[ ]:


columns_to_drop = ['chid', 'cano', 'bnsfg', 'flbmk', 'ovrlt', 'iterm']

cat_features = ['contp', 'etymd', 'mcc', 'ecfg', 'stocn', 'scity', 'insfg', 'mchno', 'acqic',
                'stscd', 'hcefg', 'csmcu', 'flg_3dsmk', 'hour','city_change', 'country_change', 'unusual_3dsmk']

X = new_val_data.drop(columns=columns_to_drop)

for feature in cat_features:
    X[feature] = X[feature].astype(str)
    
test_pool = Pool(X, cat_features=cat_features)

y_pred = catboost_model.predict(test_pool).astype(int)
new_val_data['pred']= y_pred
new_val_data =new_val_data.reset_index()

output_df = new_val_data[['txkey', 'pred']].set_index('txkey')
example = example.drop_duplicates(subset='txkey')

df2_sorted = example[['txkey']].merge(output_df, on='txkey', how='left')
output_filename = 'dataset_2nd/predictions_secondround.csv'
df2_sorted.to_csv(output_filename, index='True')