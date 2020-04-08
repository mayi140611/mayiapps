"""
update4: 减去趋势影响 用2016年regYear province regMonth bodyType均值-2017年regYear province regMonth bodyType均值
线上得分0.36078984000

得分反而下降了，有点出人意料

还可以一试的是去掉bodyType

然后基本上这种方法就没啥搞头了，分数应该在0.4左右


update3: 减去趋势影响 用2016年regMonth均值-2017年regMonth均值
线上得分0.39556193000
update2: 减去趋势影响 用2016年均值-2017年均值
线上得分0.36172235000
update1: 去除类别影响因子 province regMonth model bodyType
训练5000个周期，线上得分0.27212214000

Stopping. Best iteration:
[1164]	validation_0-rmse:0.234945	validation_1-rmse:0.185176

待改进：
没有考虑regYear, 2018得分明细应该偏低一些

只用了train_sales_data.csv的数据，不知道'popularity', 'carCommentVolum', 'newsReplyVolum'如何处理，之前取的是均值，明显不合理，因为2017年相较2016年有显著提高，

类别变量编码简单

未构建其它特征
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_sales = pd.read_csv('data_origin/train/train_sales_data.csv')
df_search = pd.read_csv('data_origin/train/train_search_data.csv')
df_user_reply = pd.read_csv('data_origin/train/train_user_reply_data.csv')
df_sales['salesVolume_log'] = np.log(df_sales['salesVolume'])
dfgt = df_sales.groupby('province regMonth model bodyType'.split()).salesVolume_log.mean()
dfgt.name = 't'

df_sales = pd.merge(df_sales, dfgt, left_on='province regMonth model bodyType'.split(), right_index=True)

df_sales['salesVolume_log_even'] = df_sales.salesVolume_log - df_sales.t
# df_sales.loc[df_sales.regYear==2017, 'salesVolume_log_even'] = df_sales.loc[df_sales.regYear==2017, 'salesVolume_log_even']+0.072852*2
dfgt1 = df_sales.groupby('regYear province regMonth bodyType'.split()).salesVolume_log_even.mean()
dfgt1.name = 't1'

df_sales = pd.merge(df_sales, dfgt1, left_on='regYear province regMonth bodyType'.split(), right_index=True)

df_sales['salesVolume_log_even'] = df_sales.salesVolume_log_even - df_sales.t1


# df = pd.merge(df_sales, df_search, on='province adcode model regYear regMonth'.split())

# df = pd.merge(df, df_user_reply, on='model regYear regMonth'.split())
df = df_sales
df.drop(columns='adcode salesVolume salesVolume_log t t1'.split(), inplace=True)

label = 'salesVolume_log_even'

y = df.pop(label).values

X_train = df[~((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_train = y[:X_train.shape[0]]

X_test = df[((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_test = y[X_train.shape[0]:]

del X_train['regYear'], X_test['regYear']


df_test = pd.read_csv('data_origin/evaluation_public.csv')



dff = df[['bodyType', 'model']].drop_duplicates()

df_test = pd.merge(df_test, dff, on='model', how='left')

df_test = pd.merge(df_test, dfgt, left_on='province regMonth model bodyType'.split(), right_index=True)
dfgt1 = dfgt1.reset_index('regYear')
dfs = dfgt1.loc[dfgt1.regYear==2017, 't1']-dfgt1.loc[dfgt1.regYear==2016, 't1']

df_test = pd.merge(df_test, dfs, left_on='province regMonth bodyType'.split(), right_index=True)
del df_test['regYear']

cat_cols = ['province', 'model', 'bodyType', 'regMonth']

from sklearn.preprocessing import LabelEncoder
for i in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[i])
    X_train.loc[:, i] = le.transform(X_train[i])
    X_test.loc[:, i] = le.transform(X_test[i])
    df_test.loc[:, i] = le.transform(df_test[i])
    
import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=5000, objective='reg:squarederror')

xgb_model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=300)     
    
print(xgb_model.feature_importances_)

dfr = pd.DataFrame({k: v['rmse'] for k, v in xgb_model.evals_result_.items()})

# %matplotlib inline

dfr.plot()    

df_test.loc[:, 'forecastVolum'] = xgb_model.predict(df_test[['province', 'model', 'bodyType', 'regMonth']])


df_test.loc[:, 'forecastVolum'] = np.exp(df_test.forecastVolum + df_test.t+df_test.t1)
df_test.loc[:, 'forecastVolum'] = df_test.forecastVolum.map(round)

print(df_test.loc[df_test.forecastVolum<0, 'forecastVolum'].shape)

df_test.loc[df_test.forecastVolum<0, 'forecastVolum']=2

df_test[['id','forecastVolum']].sort_values('id').to_csv('/Users/luoyonggui/Downloads/model1_xgboost.csv', index=False)




    
    
    
    
    
