import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

def load_user_feat():
    train_user_df = pd.read_csv(path+'underexpose_train/underexpose_user_feat.csv', names=['user_id','user_age_level','user_gender','user_city_level'])
    return train_user_df

def load_item_feat():
    train_item_df = pd.read_csv(path+'underexpose_train/underexpose_item_feat.csv', sep=r',\s+|,\[|\],\[',names=['item_id']+list(range(256)))
    train_item_df.iloc[:, -1] = train_item_df.iloc[:, -1].str.replace(']', '').map(float)
    return train_item_df

def load_whole_click_data(now_phase):
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
    click_test_val2 = pd.DataFrame()  
    all_click_df = []
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train1 = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,  names=['user_id', 'item_id', 'time']) 
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','time'])  
        
#         print(click_train.shape)
        whole_click = whole_click.append(click_train1.copy()).append(click_test1.copy())
        click_test1_val = click_test1.sort_values(['user_id', 'time']).drop_duplicates('user_id', keep='last')
        click_test_val2 = click_test_val2.append(click_test1_val, ignore_index=True)
        click_test1 = click_test1[~click_test1.index.isin(click_test1_val.index)] 
        click_train = click_train.append(click_train1).append(click_test1).sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id','item_id','time'],keep='last').reset_index(drop=True)  
#         print(click_train.shape)
        
       
        all_click_df.append((click_train.copy(), click_test1_val.copy(), test_qtime1.copy()))
        
        test_qtime = test_qtime.append(test_qtime1) 

    
    # 统计每个item_id出现的次数(item_deg)
    item_deg_count = whole_click.groupby('item_id')['time'].count().reset_index()
    item_deg_count.columns = ['item_id', 'item_deg']
    
#     whole_click = whole_click.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id','item_id','time'],keep='last').reset_index(drop=True)
    # 只保留一个user_id购买最后一次的item_id
    whole_click = whole_click.sort_values(['user_id', 'time']).drop_duplicates(subset=['user_id','item_id'],keep='last').reset_index(drop=True)
    
    click_test_val2 = pd.merge(click_test_val2, item_deg_count, how='left')
    whole_click = pd.merge(whole_click, item_deg_count).sort_values(['user_id', 'time'])
    
    # 线下验证集 注：取的是所有的click数据中属于test的user_id的最后一次的点击时间，并没有取click_test中每个user_id的最后时间
    click_test_val = whole_click[whole_click.user_id.isin(test_qtime.user_id.unique())].sort_values(['user_id', 'time']).drop_duplicates('user_id', keep='last') 
    whole_click_train = whole_click[~whole_click.index.isin(click_test_val.index.tolist())] 
    
    whole_click_train = whole_click_train.sort_values(['user_id', 'time'])
    return whole_click_train, click_test_val, test_qtime, all_click_df, click_test_val2