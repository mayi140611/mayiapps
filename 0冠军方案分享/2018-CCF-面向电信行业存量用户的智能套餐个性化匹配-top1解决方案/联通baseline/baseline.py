import time
import pickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold


############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 统计转化率
def bys_rate(data,cate,cate2,label):
    temp = data.groupby(cate2,as_index=False)[label].agg({'count':'count','sum':'sum'}).rename(columns={'2_total_fee':'1_total_fee'})
    temp['rate'] = temp['sum']/temp['count']
    data_temp = data[[cate]].copy()
    data_temp = data_temp.merge(temp[[cate,'rate']],on=cate,how='left')
    return data_temp['rate']

# 统计转化率
def mul_rate(data,cate,label):
    temp1 = data.groupby([cate,label],as_index=False).size().unstack().fillna(0)
    temp2 = data.groupby([cate], as_index=False).size()
    temp2.loc[temp2 < 20] = np.nan
    temp3 = (temp1.T/temp2).T
    temp3.columns = [cate+'_'+str(c)+'_conversion' for c in temp3.columns]
    temp3 = temp3.reset_index()
    data = data.merge(temp3,on=cate,how='left')
    return data

# 相同的个数
def get_same_count(li):
    return pd.Series(li).value_counts().values[0]

# 相同的个数
def get_second_min(li):
    return sorted(li)[1]

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True, min_count=100,inplace=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    result = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in result.columns if c not in original_columns]
    cat_columns = [c for c in original_columns if c not in result.columns]
    if not inplace:
        for c in cat_columns:
            result[c] = df[c]
    for c in new_columns:
        if (result[c].sum()<100) or ((result.shape[0]-result[c].sum())<100):
            del result[c]
            new_columns.remove(c)
    return result, new_columns

# 连续特征离散化
def one_hot_encoder_continus(df, col, n_scatter=10,nan_as_category=True):
    df[col+'_scatter'] = pd.qcut(df[col],n_scatter)
    result = pd.get_dummies(df, columns=[col+'_scatter'], dummy_na=nan_as_category)
    return result

# count encoding
def count_encoding(li):
    temp = pd.Series(li)
    result = temp.map(temp.value_counts())
    return result

# 分组标准化
def grp_standard(data,key,names,drop=False):
    for name in names:
        new_name = name if drop else name + '_' + key + '_' + 'standardize'
        mean_std = data.groupby(key, as_index=False)[name].agg({'mean': 'mean',
                                                               'std': 'std'})
        data = data.merge(mean_std, on=key, how='left')
        data[new_name] = ((data[name]-data['mean'])/data['std']).fillna(0).astype(np.float32)
        data[new_name] = data[new_name].replace(-np.inf, 0).fillna(0)
        data.drop(['mean','std'],axis=1,inplace=True)
    return data

# 多分类F1值
def multi_f1(true,pred,silent=False):
    true_dummy = pd.get_dummies(pd.Series(true))
    pred_dummy = pd.get_dummies(pd.Series(pred))
    scores = []
    for c in true_dummy.columns:
        score = f1_score(true_dummy[c],pred_dummy[c])
        if not silent:
            print('{}       :   {}'.format(c,score))
        scores.append(score)
    return np.mean(scores)

def astype(x,t):
    try:
        return t(x)
    except:
        return np.nan

def have_0(x):
    try:
        r = x.split('.')[1][-1]
        return 0 if r=='0' else 1
    except:
        return 1

def xgb_cv(params, train_feat, test_feat, predictors, label='label',groups=None,cv=5,stratified=True):
    print('开始CV 5折训练...')
    t0 = time.time()
    train_preds = np.zeros((len(train_feat), train_feat[label].nunique()))
    test_preds = np.zeros((len(test_feat), train_feat[label].nunique()))
    xgb_test = xgb.DMatrix(test_feat[predictors])
    models = []
    kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf):
    # group_kfold = GroupKFold(n_splits=cv).get_n_splits(train_preds, train_preds[label], groups, random_state=66)
    # for i, train_index, test_index in group_kfold.split(train_preds,  train_preds[label], groups):
        xgb_train = xgb.DMatrix(train_feat[predictors].iloc[train_index], train_feat[label].iloc[train_index])
        xgb_eval = xgb.DMatrix(train_feat[predictors].iloc[test_index], train_feat[label].iloc[test_index])

        print('开始第{}轮训练...'.format(i))
        params = {'objective': 'multi:softprob',
                 'eta': 0.1,
                 'max_depth': 6,
                 'silent': 1,
                 'num_class': 11,
                 'eval_metric': "mlogloss",
                 'min_child_weight': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 'seed': 66
                 } if params is None else params
        watchlist = [(xgb_train, 'train'), (xgb_eval, 'val')]

        clf = xgb.train(params,
                        xgb_train,
                        num_boost_round=3000,
                        evals=watchlist,
                        verbose_eval=50,
                        early_stopping_rounds=50)

        train_preds[test_index] += clf.predict(xgb_eval)
        test_preds += clf.predict(xgb_test)
        models.append(clf)
    pickle.dump(models,open('xgb_{}.model'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),'+wb'))
    print('用时{}秒'.format(time.time()-t0))
    return train_preds,test_preds/5


############################### 特征函数 ###########################


def make_feat(data,data_key):
    t0 = time.time()
    month_fee = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
    data['total_fee_mean4'] = data[month_fee[:4]].mean(axis=1)
    data['total_fee_mean3'] = data[month_fee[:3]].mean(axis=1)
    data['total_fee_mean2'] = data[month_fee[:2]].mean(axis=1)
    data['total_fee_std4'] = data[month_fee[:4]].std(axis=1)
    data['total_fee_Standardization'] = data['total_fee_std4'] / (data['total_fee_mean4'] + 0.1)
    data['1_total_fee_rate12'] = data['1_total_fee'] / (data['2_total_fee'] + 0.1)
    data['1_total_fee_rate23'] = data['2_total_fee'] / (data['3_total_fee'] + 0.1)
    data['1_total_fee_rate34'] = data['3_total_fee'] / (data['4_total_fee'] + 0.1)
    data['1_total_fee_rate24'] = data['total_fee_mean2'] / (data['total_fee_mean4'] + 0.1)
    data['total_fee_max4'] = data[month_fee[:4]].max(axis=1)
    data['total_fee_min4'] = data[month_fee[:4]].min(axis=1)
    data['total_fee_second_min4'] = data[month_fee[:4]].apply(get_second_min, axis=1)
    data['service_caller_time_diff'] = data['service2_caller_time'] - data['service1_caller_time']
    data['service_caller_time_sum'] = data['service2_caller_time'] + data['service1_caller_time']
    data['service_caller_time_min'] = data[['service1_caller_time', 'service2_caller_time']].min(axis=1)
    data['service_caller_time_max'] = data[['service1_caller_time', 'service2_caller_time']].max(axis=1)

    data['1_total_fee_last0_number'] = count_encoding(
        data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
    data['1_total_fee_last1_number'] = count_encoding(
        data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
    data['1_total_fee_last2_number'] = count_encoding(
        data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
    data['1_total_fee_last3_number'] = count_encoding(data['1_total_fee'].fillna(-1) // 10)
    data['2_total_fee_last0_number'] = count_encoding(
        data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
    data['2_total_fee_last1_number'] = count_encoding(
        data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
    data['2_total_fee_last2_number'] = count_encoding(
        data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
    data['2_total_fee_last3_number'] = count_encoding(data['2_total_fee'].fillna(-1) // 10)
    data['3_total_fee_last0_number'] = count_encoding(
        data['3_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
    data['3_total_fee_last1_number'] = count_encoding(
        data['3_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
    data['3_total_fee_last2_number'] = count_encoding(
        data['3_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
    data['3_total_fee_last3_number'] = count_encoding(data['3_total_fee'].fillna(-1) // 10)
    data['4_total_fee_last0_number'] = count_encoding(
        data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
    data['4_total_fee_last1_number'] = count_encoding(
        data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
    data['4_total_fee_last2_number'] = count_encoding(
        data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
    data['4_total_fee_last3_number'] = count_encoding(data['4_total_fee'].fillna(-1) // 10)

    for fee in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']:
        data['{}_1'.format(fee)] = ((data[fee] % 1 == 0) & (data[fee] != 0))
        data['{}_01'.format(fee)] = ((data[fee] % 0.1 == 0) & (data[fee] != 0))

    data['pay_number_last_2'] = data['pay_num'] * 100 % 100

    data['1_total_fee_log'] = np.log(data['1_total_fee'] + 2)
    data['2_total_fee_log'] = np.log(data['2_total_fee'] + 2)
    data['3_total_fee_log'] = np.log(data['3_total_fee'] + 2)
    data['4_total_fee_log'] = np.log(data['4_total_fee'] + 2)
    data = grp_standard(data, 'contract_type', ['1_total_fee_log'], drop=False)
    data = grp_standard(data, 'contract_type', ['service_caller_time_min'], drop=False)
    data = grp_standard(data, 'contract_type', ['service_caller_time_max'], drop=False)
    data = grp_standard(data, 'contract_type', ['online_time'], drop=False)
    data = grp_standard(data, 'contract_type', ['age'], drop=False)
    data = grp_standard(data, 'net_service', ['1_total_fee_log'], drop=False)
    data = grp_standard(data, 'net_service', ['service_caller_time_min'], drop=False)
    data = grp_standard(data, 'net_service', ['service_caller_time_max'], drop=False)
    data = grp_standard(data, 'net_service', ['online_time'], drop=False)
    data = grp_standard(data, 'net_service', ['age'], drop=False)
    data['age_scatter'] = pd.qcut(data['age'], 5)
    data = grp_standard(data, 'age_scatter', ['1_total_fee_log'], drop=False)
    data = grp_standard(data, 'age_scatter', ['service_caller_time_min'], drop=False)
    data = grp_standard(data, 'age_scatter', ['service_caller_time_max'], drop=False)
    data = grp_standard(data, 'age_scatter', ['online_time'], drop=False)
    data = grp_standard(data, 'age_scatter', ['age'], drop=False)
    data['online_time_scatter'] = pd.qcut(data['online_time'], 5)
    data = grp_standard(data, 'online_time_scatter', ['1_total_fee_log'], drop=False)
    data = grp_standard(data, 'online_time_scatter', ['service_caller_time_min'], drop=False)
    data = grp_standard(data, 'online_time_scatter', ['service_caller_time_max'], drop=False)
    data = grp_standard(data, 'online_time_scatter', ['online_time'], drop=False)
    data = grp_standard(data, 'online_time_scatter', ['age'], drop=False)
    data = grp_standard(data, 'service_type', ['1_total_fee_log'], drop=False)
    data = grp_standard(data, 'service_type', ['service_caller_time_min'], drop=False)
    data = grp_standard(data, 'service_type', ['service_caller_time_max'], drop=False)
    data = grp_standard(data, 'service_type', ['online_time'], drop=False)
    data = grp_standard(data, 'service_type', ['age'], drop=False)

    del data['1_total_fee_log'], data['2_total_fee_log'], data['3_total_fee_log'], data['4_total_fee_log'], \
        data['age_scatter'], data['online_time_scatter']

    data['month_traffic_last_month_traffic_sum'] = data['month_traffic'] + data['last_month_traffic']
    data['month_traffic_last_month_traffic_diff'] = data['month_traffic'] - data['last_month_traffic']
    data['month_traffic_last_month_traffic_rate'] = data['month_traffic'] / (data['last_month_traffic'] + 0.01)
    data['outer_trafffic_month'] = data['month_traffic'] - data['local_trafffic_month']
    data['local_trafffic_month_month_traffic_rate'] = data['local_trafffic_month'] / (data['month_traffic'] + 0.01)

    data['month_traffic_last_month_traffic_sum_1_total_fee_rate'] = data['month_traffic_last_month_traffic_sum'] / (
        data['1_total_fee'] + 0.01)
    data['month_traffic_local_caller_time'] = data['month_traffic'] / (data['local_caller_time'] + 0.01)
    data['pay_num_per'] = data['pay_num'] / (data['pay_times'] + 0.01)
    data['total_fee_mean4_pay_num_rate'] = data['pay_num'] / (data['total_fee_mean4'] + 0.01)
    data['local_trafffic_month_spend'] = data['local_trafffic_month'] - data['last_month_traffic']
    data['month_traffic_1_total_fee_rate'] = data['month_traffic'] / (data['1_total_fee'] + 0.01)

    for traffic in ['month_traffic', 'last_month_traffic', 'local_trafffic_month']:
        data['{}_1'.format(traffic)] = ((data[traffic] % 1 == 0) & (data[traffic] != 0))
        data['{}_50'.format(traffic)] = ((data[traffic] % 50 == 0) & (data[traffic] != 0))
        data['{}_1024'.format(traffic)] = ((data[traffic] % 1024 == 0) & (data[traffic] != 0))
        data['{}_1024_50'.format(traffic)] = ((data[traffic] % 1024 % 50 == 0) & (data[traffic] != 0))

    data['service_caller_time'] = data['service1_caller_time'] + data['service2_caller_time']
    data['outer_caller_time'] = data['service_caller_time'] - data['local_caller_time']
    data['local_caller_time_rate'] = data['local_caller_time'] / (data['service_caller_time'] + 0.01)
    data['service1_caller_time_rate'] = data['service1_caller_time'] / (data['service_caller_time'] + 0.01)
    data['local_caller_time_service2_caller_time_rate'] = data['local_caller_time'] / (
        data['service2_caller_time'] + 0.01)
    data['service1_caller_time_1_total_fee_rate'] = data['service_caller_time'] / (data['1_total_fee'] + 0.01)

    data['contract_time_count'] = count_encoding(data['contract_time'])
    data['pay_num_count'] = count_encoding(data['pay_num'])
    data['pay_num_last0_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-1]).astype(int))
    data['pay_num_last1_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-2]).astype(int))
    data['pay_num_last2_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-4]).astype(int))
    data['pay_num_count'] = count_encoding(data['pay_num'] // 10)
    data['age_count3'] = count_encoding(data['age'] // 3)
    data['age_count6'] = count_encoding(data['age'] // 6)
    data['age_count10'] = count_encoding(data['age'] // 10)

    # 转化率
    data = mul_rate(data, 'pay_num', 'current_service')

    data = pd.get_dummies(data, columns=['contract_type'], dummy_na=-1)
    data = pd.get_dummies(data, columns=['net_service'], dummy_na=-1)
    data = pd.get_dummies(data, columns=['complaint_level'], dummy_na=-1)
    data.reset_index(drop=True, inplace=True)

    print('特征矩阵大小：{}'.format(data.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return data




d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}
d1 = {0:0,4:1,8:2}
rd1 = {0:0,1:4,2:8}
d4 = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7}
rd4 = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10}

print('数据预处理...')
str_dict = {'1_total_fee': 'str',
 '2_total_fee': 'str',
 '3_total_fee': 'str',
 '4_total_fee': 'str',
 'last_month_traffic': 'str',
 'local_caller_time': 'str',
 'local_trafffic_month': 'str',
 'month_traffic': 'str',
 'pay_num': 'str',
 'service1_caller_time': 'str',
 'service2_caller_time': 'str'}

have_0_c = ['1_total_fee',
'2_total_fee',
'3_total_fee',
'4_total_fee',
'month_traffic',
'last_month_traffic',
'local_trafffic_month',
'local_caller_time',
'service1_caller_time',
'service2_caller_time',
'pay_num']

train = pd.read_csv('train.csv',dtype=str_dict)
test = pd.read_csv( 'test.csv',dtype=str_dict)
train['label'] = train['current_service'].map(d)

def deal(data):
    for c in have_0_c:
        data['have_0_{}'.format(c)] = data[c].apply(have_0)
        try:
            data[c] = data[c].astype(float)
        except:
            pass
    data['2_total_fee'] = data['2_total_fee'].apply(lambda x: astype(x,float))
    data['3_total_fee'] = data['3_total_fee'].apply(lambda x: astype(x,float))
    data['age'] = data['age'].apply(lambda x: astype(x,int))
    data['gender'] = data['gender'].apply(lambda x: astype(x,int))
    data.loc[data['age']==0,'age'] = np.nan
    data.loc[data['1_total_fee'] < 0, '1_total_fee'] = np.nan
    data.loc[data['2_total_fee'] < 0, '2_total_fee'] = np.nan
    data.loc[data['3_total_fee'] < 0, '3_total_fee'] = np.nan
    data.loc[data['4_total_fee'] < 0, '4_total_fee'] = np.nan
    for c in [
    '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
    'month_traffic', 'last_month_traffic', 'local_trafffic_month',
    'local_caller_time', 'service1_caller_time', 'service2_caller_time',
    'many_over_bill', 'contract_type', 'contract_time', 'pay_num', ]:
        data[c] = data[c].round(4)
    return data

train = deal(train)
train = train[train['current_service'] != 999999].copy()
test = deal(test)
data = train.append(test)

print('构造特征...')
data_feat = make_feat(data,'online')
test_feat = data_feat[data_feat['user_id'].isin(test['user_id'])].copy()
train_feat = data_feat[data_feat['user_id'].isin(train['user_id'])].copy()

train_feat1 = train_feat[train_feat['service_type']==1].copy()
test_feat1 = test_feat[test_feat['service_type']==1].copy()
train_feat1['label'] = train_feat1['label'].map(d1)
predictors1 = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label']) and
               ('contract_type' not in c) and ('service_type' not in c)]
params = {'objective': 'multi:softprob',
         'eta': 0.5,
         'max_depth': 6,
         'silent': 1,
         'num_class': 3,
         'eval_metric': "mlogloss",
         'min_child_weight': 3,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'seed': 66
         }
train_preds1,test_preds1 = xgb_cv(params,train_feat1,test_feat1,predictors1)
int_train_preds1 = train_preds1.argmax(axis=1)
int_test_preds1 = test_preds1.argmax(axis=1)
print('线下第一类的得分为：  {}'.format(multi_f1(train_feat1['label'],int_train_preds1)**2))
train_preds1 = pd.DataFrame(train_preds1)
train_preds1['user_id'] = train_feat1['user_id'].values
test_preds1 = pd.DataFrame(test_preds1)
test_preds1['user_id'] = test_feat1['user_id'].values
data_pred1 = train_preds1.append(test_preds1)
data_pred1.columns = [rd1[i] if i in rd1 else i for i in data_pred1.columns]


train_feat4 = train_feat[train_feat['service_type']!=1].copy()
test_feat4 = test_feat[test_feat['service_type']!=1].copy()
train_feat4['label'] = train_feat4['label'].map(d4)
predictors4 = [c for c in train_feat.columns if (c not in ['user_id', 'current_service', 'label'])]
params = {'objective': 'multi:softprob',
                 'eta': 0.1,
                 'max_depth': 6,
                 'silent': 1,
                 'num_class': 8,
                 'eval_metric': "mlogloss",
                 'min_child_weight': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.7,
                 'seed': 66
                 }
train_preds4,test_preds4 = xgb_cv(params,train_feat4,test_feat4,predictors4)
int_train_preds4 = train_preds4.argmax(axis=1)
int_test_preds4 = test_preds4.argmax(axis=1)
print('线下第四类的得分为：  {}'.format(multi_f1(train_feat4['label'],int_train_preds4)**2))
train_preds4 = pd.DataFrame(train_preds4)
train_preds4['user_id'] = train_feat4['user_id'].values
test_preds4 = pd.DataFrame(test_preds4)
test_preds4['user_id'] = test_feat4['user_id'].values
data_pred4 = train_preds4.append(test_preds4)
data_pred4.columns = [rd4[i] if i in rd4 else i for i in data_pred4.columns]

# 输出预测概率，做stacking使用
data_pred = data_pred1.append(data_pred4).fillna(0)
data_pred.to_csv( 'data_preds_xgb1_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

# 计算cv得分
int_train_preds1 = pd.DataFrame({'user_id':train_feat1['user_id'].values,'pred':[rd1[i] for i in int_train_preds1 ]})
int_train_preds4 = pd.DataFrame({'user_id':train_feat4['user_id'].values,'pred':[rd4[i] for i in int_train_preds4 ]})
int_train_preds = int_train_preds1.append(int_train_preds4)
train_feat = train_feat.merge(int_train_preds,on='user_id',how='left')
train_feat['label'] = train_feat['current_service'].map(d)
print('线下F1得分为：  {}'.format(multi_f1(train_feat['label'],train_feat['pred'])**2))

int_test_preds1 = pd.DataFrame({'user_id':test_feat1['user_id'].values,'current_service':[rd1[i] for i in int_test_preds1]})
int_test_preds4 = pd.DataFrame({'user_id':test_feat4['user_id'].values,'current_service':[rd4[i] for i in int_test_preds4]})
test_preds = int_test_preds1.append(int_test_preds4)
test_preds['current_service'] = test_preds['current_service'].map(rd)
submission = test_feat[['user_id']].merge(test_preds,on='user_id',how='left')
submission[['user_id','current_service']].to_csv('xindai_sumbmission_xgb1_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index=False)
























