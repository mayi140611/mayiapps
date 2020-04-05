"""
训练5000个周期，线上得分0.00010000000
Stopping. Best iteration:
[3501]	validation_0-rmse:0.067525	validation_1-rmse:0.467177
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_sales = pd.read_csv('data_origin/train/train_sales_data.csv')
df_search = pd.read_csv('data_origin/train/train_search_data.csv')
df_user_reply = pd.read_csv('data_origin/train/train_user_reply_data.csv')

df = pd.merge(df_sales, df_search, on='province adcode model regYear regMonth'.split())

df = pd.merge(df, df_user_reply, on='model regYear regMonth'.split())

del df['adcode']

y = df.pop('salesVolume').values

X_train = df[~((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_train = y[:X_train.shape[0]]

X_test = df[((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_test = y[X_train.shape[0]:]

del X_train['regYear'], X_test['regYear']

cat_cols = ['province', 'model', 'bodyType', 'regMonth']

df_test1 = pd.read_csv('data_origin/evaluation_public.csv')
del df_test1['regYear']


dff = df[['bodyType', 'model']].drop_duplicates()

df_test = pd.merge(df_test1, dff, on='model', how='left')

def t(s):
    return int(s.mean())
dfg = df.groupby('model regMonth'.split())[['popularity', 'carCommentVolum', 'newsReplyVolum']].agg(t)

df_test = pd.merge(df_test, dfg, left_on='model regMonth'.split(), right_index=True, how='left')

from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[i])
    X_train.loc[:, i] = le.transform(X_train[i])
    X_test.loc[:, i] = le.transform(X_test[i])
    df_test.loc[:, i] = le.transform(df_test[i])

import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=5000, objective='reg:squarederror')

xgb_model.fit(X_train, np.log(y_train+1),
        eval_set=[(X_train, np.log(y_train+1)), (X_test, np.log(y_test+1))], early_stopping_rounds=300)

dfr = pd.DataFrame({k: v['rmse'] for k, v in xgb_model.evals_result_.items()})

# %matplotlib inline

dfr.plot()

df_test.loc[:, 'forecastVolum'] = np.exp(xgb_model.predict(df_test[['province', 'model', 'bodyType', 'regMonth', 'popularity', 'carCommentVolum', 'newsReplyVolum']])) - 1
df_test.loc[:, 'forecastVolum'] = df_test.forecastVolum.map(round)

print(df_test.loc[df_test.forecastVolum<0, 'forecastVolum'].shape)
df_test.loc[df_test.forecastVolum<0, 'forecastVolum']=2

df_test[['id','forecastVolum']].sort_values('id').to_csv('/Users/luoyonggui/Downloads/baseline_xgboost2.csv', index=False)


