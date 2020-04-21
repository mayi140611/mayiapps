import pandas as pd

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
    whole_click = whole_click.sort_values('time')
    return whole_click, click_train, click_test, test_qtime