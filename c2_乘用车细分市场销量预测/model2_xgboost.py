"""
4、避免误差传递，试着用一下一等奖的方案
3、加入'popularity', 'carCommentVolum', 'newsReplyVolum'

2、score： 0.57074112000
在1的基础上对salesVolume取log，可以看到分数有了显著的提高
Stopping. Best iteration:
[51]	validation_0-rmse:0.244479	validation_1-rmse:0.262602
1、score： 0.45401776000 
使用了前n个月销量、前n个月累计值、前n个月差值，没有对salesVolume取log
Stopping. Best iteration:
[40]	validation_0-rmse:165.613	validation_1-rmse:208.512
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_sales = pd.read_csv('data_origin/train/train_sales_data.csv')
df_search = pd.read_csv('data_origin/train/train_search_data.csv')
df_user_reply = pd.read_csv('data_origin/train/train_user_reply_data.csv')

df = pd.merge(df_sales, df_search, on='province adcode model regYear regMonth'.split())

df = pd.merge(df, df_user_reply, on='model regYear regMonth'.split())

df_test = pd.read_csv('data_origin/evaluation_public.csv')
# 添加bodyType
dft = df[['bodyType', 'model']].drop_duplicates()
df_test = pd.merge(df_test, dft, on='model', how='left')
del dft

df_test = df_test.rename(columns={'forecastVolum': 'salesVolume'})

df = pd.concat([df, df_test], sort=False, ignore_index=True)
del df['adcode']

cols_cat = ['province', 'model', 'bodyType', 'regYear', 'regMonth']
cols_num = ['popularity', 'carCommentVolum', 'newsReplyVolum']
label = 'salesVolume'

df['t'] = 0
df.loc[df.regYear==2017, 't'] = 12
df.loc[df.regYear==2018, 't'] = 24
df['time_id'] = df.regMonth + df.t
df['time_id'] = df.time_id.map(int)
del df['t']

for i in range(6):
    df.loc[:, f'salesVolume_last_{i+1}'] = df.groupby('province model'.split())['salesVolume'].shift(i+1)
for i in range(2, 6):
    df.loc[:, f'salesVolume_sum_{i}'] = df.groupby('province model'.split())['salesVolume'].rolling(i).sum().reset_index('province model'.split(), drop=True)
    df.loc[:, f'salesVolume_sum_{i}'] = df.groupby('province model'.split())[f'salesVolume_sum_{i}'].shift(1)
for i in range(1, 6):
    df.loc[:, f'salesVolume_diff_{i}_{i+1}'] = df[f'salesVolume_last_{i}'] - df[f'salesVolume_last_{i+1}']
    df.loc[:, f'salesVolume_diff_{i}_{i+1}_ratio'] = df.loc[:, f'salesVolume_diff_{i}_{i+1}']/(df[f'salesVolume_last_{i+1}']+1)

del df['regYear']

cat_cols = ['province', 'model', 'bodyType', 'regMonth']

from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    le = LabelEncoder()
    le.fit(df[i])
    df.loc[:, i] = le.transform(df[i])    
    
cols = ['province', 'model', 'bodyType', 'regMonth', 'time_id', 'salesVolume',
       'salesVolume_last_1', 'salesVolume_last_2', 'salesVolume_last_3',
       'salesVolume_last_4', 'salesVolume_last_5', 'salesVolume_last_6',
       'salesVolume_sum_2', 'salesVolume_sum_3', 'salesVolume_sum_4',
       'salesVolume_sum_5', 'salesVolume_diff_1_2',
       'salesVolume_diff_1_2_ratio', 'salesVolume_diff_2_3',
       'salesVolume_diff_2_3_ratio', 'salesVolume_diff_3_4',
       'salesVolume_diff_3_4_ratio', 'salesVolume_diff_4_5',
       'salesVolume_diff_4_5_ratio', 'salesVolume_diff_5_6',
       'salesVolume_diff_5_6_ratio']

df_train = df.loc[(df.time_id<21)&(df.time_id>6), cols]
df_val = df.loc[(df.time_id<25)&(df.time_id>20), cols]

label = 'salesVolume'
y_train = df_train.pop(label).values
y_val = df_val.pop(label).values

import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=4, n_estimators=5000, objective='reg:squarederror')

# xgb_model.fit(df_train, y_train,
#         eval_set=[(df_train, y_train), (df_val, y_val)], early_stopping_rounds=300)     
xgb_model.fit(df_train, np.log(y_train+1),
        eval_set=[(df_train, np.log(y_train+1)), (df_val, np.log(y_val+1))], early_stopping_rounds=300)  


print(xgb_model.feature_importances_)

dfr = pd.DataFrame({k: v['rmse'] for k, v in xgb_model.evals_result_.items()})

# %matplotlib inline

dfr.plot()  


# test 按照每个月
for ii in [25, 26, 27, 28]:
    for i in range(6):
        df.loc[:, f'salesVolume_last_{i+1}'] = df.groupby('province model'.split())['salesVolume'].shift(i+1)
    for i in range(2, 6):
        df.loc[:, f'salesVolume_sum_{i}'] = df.groupby('province model'.split())['salesVolume'].rolling(i).sum().reset_index('province model'.split(), drop=True)
        df.loc[:, f'salesVolume_sum_{i}'] = df.groupby('province model'.split())[f'salesVolume_sum_{i}'].shift(1)
    for i in range(1, 6):
        df.loc[:, f'salesVolume_diff_{i}_{i+1}'] = df[f'salesVolume_last_{i}'] - df[f'salesVolume_last_{i+1}']
        df.loc[:, f'salesVolume_diff_{i}_{i+1}_ratio'] = df.loc[:, f'salesVolume_diff_{i}_{i+1}']/(df[f'salesVolume_last_{i+1}']+1)
    
    df_test = df.loc[(df.time_id==ii), cols]
    del df_test[label]

#     df.loc[(df.time_id==ii), label] = np.round(xgb_model.predict(df_test)) 
    df.loc[(df.time_id==ii), label] = np.round(np.exp(xgb_model.predict(df_test))-1) 
    
df['forecastVolum'] = df['salesVolume']

df.loc[df.time_id.isin([25, 26, 27, 28]), ['id','forecastVolum']].applymap(int).sort_values('id').to_csv('/Users/luoyonggui/Downloads/model2_xgboost.csv', index=False)    