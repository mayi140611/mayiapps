import pandas as pd
from sklearn.model_selection import train_test_split

df_sales = pd.read_csv('data_origin/train/train_sales_data.csv')
df_search = pd.read_csv('data_origin/train/train_search_data.csv')
df_user_reply = pd.read_csv('data_origin/train/train_user_reply_data.csv')

df = pd.merge(df_sales, df_search, on='province adcode model regYear regMonth'.split())

df = pd.merge(df, df_user_reply, on='model regYear regMonth'.split())

del df['adcode']

y = df.pop('salesVolume').values



# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

X_train = df[~((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_train = y[:X_train.shape[0]]

X_test = df[((df.regYear==2017)&(df.regMonth.isin([9,10,11,12])))]
y_test = y[X_train.shape[0]:]

del X_train['regYear'], X_test['regYear']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train['popularity carCommentVolum newsReplyVolum'.split()])

X_test.loc[:, 'popularity carCommentVolum newsReplyVolum'.split()] = scaler.transform(X_test['popularity carCommentVolum newsReplyVolum'.split()])

cat_cols = ['province', 'model', 'bodyType', 'regMonth']

from catboost import CatBoostRegressor

model = CatBoostRegressor(
#     iterations=5000,
#     learning_rate=
)

model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test), plot=True)

model.feature_importances_

model.feature_names_

df_test1 = pd.read_csv('data_origin/evaluation_public.csv')
del df_test1['regYear']


dff = df[['bodyType', 'model']].drop_duplicates()

df_test = pd.merge(df_test1, dff, on='model', how='left')

def t(s):
    return int(s.mean())
dfg = df.groupby('model regMonth'.split())[['popularity', 'carCommentVolum', 'newsReplyVolum']].agg(t)

df_test = pd.merge(df_test, dfg, left_on='model regMonth'.split(), right_index=True, how='left')

df_test.loc[:, 'popularity carCommentVolum newsReplyVolum'.split()] = scaler.transform(df_test['popularity carCommentVolum newsReplyVolum'.split()])

df_test.loc[:, 'forecastVolum'] = model.predict(df_test[['province', 'bodyType', 'model', 'regMonth', 'popularity', 'carCommentVolum', 'newsReplyVolum']])
df_test.loc[:, 'forecastVolum'] = df_test.forecastVolum.map(round)


df_test.loc[df_test.forecastVolum<0, 'forecastVolum']=2

df_test[['id','forecastVolum']].sort_values('id').to_csv('data_gen/baseline_catboost2.csv', index=False)

print('complete!')