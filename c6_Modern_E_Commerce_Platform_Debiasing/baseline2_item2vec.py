"""
https://zhuanlan.zhihu.com/p/126735057

日期:2020-04-22 19:56:59排名: 无

score:0.0156
hitrate_50_full:0.0403
ndcg_50_full:0.0156
hitrate_50_half:0.0033
ndcg_50_half:0.0009

日期:2020-04-22 19:35:44排名: 无

score:0.0129
hitrate_50_full:0.0324
ndcg_50_full:0.0129
hitrate_50_half:0.0011
ndcg_50_half:0.0002
"""
from gensim.models.word2vec import *
import pandas as pd
from time import *
from tqdm import *
import numpy as np
import os
import numpy as np
from tqdm import *
import re

t0 = time.time()

def load_click_data(now_phase, gen_val_set=False):
    """
    gen_val_set = 是否产生线性验证集
    """
#     now_phase = 2  
    train_path = './data_origin/underexpose_train'  
    test_path = './data_origin/underexpose_test'  
    recom_item = []  

    whole_click = pd.DataFrame()  
    click_train = pd.DataFrame()   
    click_test = pd.DataFrame()  
    test_qtime = pd.DataFrame()  
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train1 = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time']) 
        if gen_val_set:
            print('gen val set...')
            dft = click_test1.sort_values('time').drop_duplicates('user_id', keep='last')
            click_test1 = click_test1[~click_test1.index.isin(dft.index.tolist())]
            dft.to_csv(f'data_gen/underexpose_test_qtime_with_answer-{c}.csv', index=False, header=None)
            del dft
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'])  

        click_train = click_train.append(click_train1) 
    #     all_click = click_train.append(click_test1)  
        click_test = click_test.append(click_test1) 
        test_qtime = test_qtime.append(test_qtime1) 

    # 去掉 train中time>query_time的数据    
#     click_train = pd.merge(click_train, test_qtime, how='left').fillna(10)  
#     click_train = click_train[click_train.time <= click_train.query_time]
#     del click_train['query_time']
#     whole_click = click_train.append(click_test)  
#     whole_click = whole_click.drop_duplicates()
#     whole_click = whole_click.sort_values('time').reset_index(drop=True)
    return click_train, click_test, test_qtime

TrainUserFeat = pd.read_csv(train_dir + 'underexpose_user_feat.csv',header=None)
TrainUserFeat.columns = ['userid','age','gender','city']
TrainItemFeat = pd.read_csv(train_dir + 'underexpose_item_feat.csv',header=None)
TrainItemFeat.iloc[:,1] = TrainItemFeat.iloc[:,1].apply(lambda x:float(x[1:]))
TrainItemFeat.iloc[:,-1] = TrainItemFeat.iloc[:,-1].apply(lambda x:float(x[:-1]))
feat_col = ['eb'+str(i) for i in TrainItemFeat.columns[1:]]
feat_col.insert(0,'items')
TrainItemFeat.columns = feat_col

TrainClick,TestClick,TestQtime = load_click_data(2)

TrainClick['flag'] = 0
TestQtime['flag'] = 1
TestClick['flag'] = 2
click = pd.concat([TrainClick,TestClick,TestQtime],axis=0 ,sort=False)

click.columns = ['userid','items','time', 'flag']

# 用户的点击物品以及对应的时间，此时，我们更加考虑是否推荐一些与用户曾经点击过的物品类似的东西，这样更有可能点击。
# 试想一下，假如我们最近喜欢新出的iphone，是否很大概率喜欢iphone的一些配件，按照这个道理，我们建立模型。
df = click.merge(TrainUserFeat,on = 'userid',how = 'left')
df = df.merge(TrainItemFeat,on='items',how = 'left')
df = df.sort_values(['userid','time']).reset_index(drop = True)

#重新划分训练集和测试集，并按时间排序，反应用户的点击行为。
train = df[df['flag']!=1].copy()
train = train.sort_values(['userid','time']).reset_index(drop = True)
test = df[df['flag']==1].copy()
test = test.sort_values(['userid','time']).reset_index(drop = True)

train.head(2)

test.head(2)

#训练模型，因为Word2vec的输入是string格式，需要提前处理，同时，把数据格式处理成uid=['itme1',itme2',...itmen']这种格式，
# 其中items_last为用户点击的最后3个物品，因为跟时间有关系，我们更加会推与用户最近点击的相关物品
tr = train.copy()
tr['items']  = tr['items'].astype(str)
items_all = tr['items'].unique()
tr = tr.groupby('userid')['items'].apply(lambda x:list(x)).reset_index()
tr['items_last'] = tr['items'].apply(lambda x:x[-3:])

tr.head()

def train_model(data):
    """训练模型"""
    begin_time = time()
    model = Word2Vec(data['items'].values, size=128, window=3, min_count=1, workers=4)
    end_time = time()
    run_time = end_time-begin_time
    print ('该循环程序运行时间：',round(run_time,2)) #该循环程序运行时间： 1.4201874732
    return model

model = train_model(tr)

def get_top_similar(items,k = 50):
    """计算item的相似度"""
    re_list = list(map(lambda x:[x[0],x[1]],model.most_similar(positive=[items],topn=k)))
    return re_list

#训练完模型后，我们需要计算用户点击过的物品相似度，一般而言，物品相似度是根据用户点击过的物品序列，计算embedding，从而计算相似度
recommendation_items = dict()
print('获取相似item') 
for i in tqdm(items_all):
    recommendation_items[i] = get_top_similar(i)

#计算用户最后点击的2个物品以及他们所对应的物品，每个物品包括物品代号以及相似度['itmes','sim']
tr['items_last_1'] = tr['items_last'].apply(lambda x:recommendation_items[x[-1]])
tr['items_last_2'] = tr['items_last'].apply(lambda x:recommendation_items[x[-2]])
tr['items_last_all'] = tr['items_last_1']+tr['items_last_2']
#根据相似度排序，越是相似的，优先推荐
tr['items_last_all'] = tr['items_last_all'].apply(lambda x:sorted(x,key = lambda x:x[1],reverse=True))

tr.head()

def melt_data():
    z = test_df.groupby(['userid'])['items_last_all'].apply(lambda x:np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['userid'], row['items_last_all']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['items_new','userid']
    i['items'] = i['items_new'].apply(lambda x:x[0])
    i['weights'] = i['items_new'].apply(lambda x:x[1])
    return i.iloc[:,1:]

test_df = test[['userid']].merge(tr,on = 'userid',how = 'left')
test_df.head(2)

z = test_df.groupby(['userid'])['items_last_all'].apply(lambda x:np.concatenate(list(x))).reset_index()
z.head(2)

i = pd.concat([pd.Series(row['userid'], row['items_last_all']) for _, row in z.iterrows()]).reset_index()
i.columns = ['items_new','userid']
i['items'] = i['items_new'].apply(lambda x:x[0])
i['weights'] = i['items_new'].apply(lambda x:x[1])
i.head(2)

test_df = i.iloc[:,1:]
test_df['items'] = test_df['items'].astype(float)
test_df = test_df.merge(TrainUserFeat,on = 'userid',how = 'left')
test_df = test_df.merge(TrainItemFeat,on='items',how = 'left')
test_df.head(2)

test_df = test_df.sort_values(['userid','weights'],ascending=False).reset_index()
test_df.drop_duplicates(['userid', 'items'], keep='first', inplace=True)

submit = test_df.groupby(['userid'])['items'].apply(lambda x: list(x)[:50]).reset_index()

submit.head(2)

sub = pd.DataFrame(list(submit['items'].values))
sub.columns = ['item_id_'+str(i).zfill(2) for i in range(1,51)]

sub.index = submit.userid

sub = sub.applymap(int)

sub.to_csv('/Users/luoyonggui/Downloads/baseline1_itemcf5.csv', header=None)

t1 = time.time()
print(f'{t1-t0}s')
print('complete!!!')