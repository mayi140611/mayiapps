import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
# https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.6c3f5619NDeQ04&postId=102089
# the higher scores, the better performance
def evaluate_each_phase(predictions: dict, answers: dict):
    """
    
    predictions: dict
        key是user_id, value是list [pred_item_id_1,pred_item_id_2...]
    answers: dict
        key是user_id, value是(item_id, item_degree)
    """
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]  # item_degree 应该是内置的表示新颖性的指标，越小，越新颖
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        # 把正确的item_id和 预测值 依次 比对，
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.
def _create_answer_file_for_evaluation(answer_fname='data_gen/debias_track_answer.csv'):
    train = './data_origin/underexpose_train/underexpose_train_click-%d.csv'
    test = './data_origin/underexpose_test/underexpose_test_click-%d/underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, item_id>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = 'data_gen/underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    now_phase = 2
    with open(answer_fname, 'w') as fout:
        for phase_id in range(now_phase+1):
            
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % (phase_id, phase_id)) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)


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
        test_qtime1 = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,  names=['user_id','query_time'])  

        click_train = click_train.append(click_train1) 
    #     all_click = click_train.append(click_test1)  
        click_test = click_test.append(click_test1) 
        test_qtime = test_qtime.append(test_qtime1) 

    # 去掉 train中time>query_time的数据    
    click_train = pd.merge(click_train, test_qtime, how='left').fillna(10)  
    click_train = click_train[click_train.time <= click_train.query_time]
    del click_train['query_time']
    whole_click = click_train.append(click_test)  
    whole_click = whole_click.drop_duplicates()
    whole_click = whole_click.sort_values('time').reset_index(drop=True)
    return whole_click, click_train, click_test, test_qtime