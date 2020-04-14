"""
日期:2020-04-13 18:37:27排名: 无

score:0.0006
hitrate_50_full:0.0030
ndcg_50_full:0.0006
hitrate_50_half:0.0011
ndcg_50_half:0.0002

"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

t0 = time.time()

path = './data_origin/'

train_click_0_df = pd.read_csv(path+'underexpose_train/underexpose_train_click-0.csv',names=['user_id','item_id','time'])
train_click_1_df = pd.read_csv(path+'underexpose_train/underexpose_train_click-1.csv',names=['user_id','item_id','time'])
train_click_df = pd.concat([train_click_0_df, train_click_1_df], ignore_index=True)

test_click_0_df = pd.read_csv(path+'underexpose_test/underexpose_test_click-0/underexpose_test_click-0.csv', names=['user_id','item_id','time'])
test_click_1_df = pd.read_csv(path+'underexpose_test/underexpose_test_click-1/underexpose_test_click-1.csv', names=['user_id','item_id','time'])

test_click_df = pd.concat([test_click_0_df, test_click_1_df], ignore_index=True)

click_df = pd.concat([train_click_df, test_click_df], ignore_index=True)

# 删除重复的数据
click_df = click_df.drop_duplicates()
click_df.shape

n_users = click_df.user_id.nunique()

n_items = click_df.item_id.nunique()

print('gen train_data_matrix begin...')
train_data_matrix = np.zeros((n_users, n_items))
user_id_dict, item_id_dict = dict(), dict()
user_id_dict_inv, item_id_dict_inv = dict(), dict()
u_cnt, i_cnt = 0, 0
for line in click_df.itertuples():
    if line[1] not in user_id_dict:
        user_id_dict[line[1]] = u_cnt
        user_id_dict_inv[u_cnt] = line[1]
        u_cnt += 1
    if line[2] not in item_id_dict:
        item_id_dict[line[2]] = i_cnt
        item_id_dict_inv[i_cnt] = line[2]
        i_cnt += 1
    train_data_matrix[user_id_dict[line[1]], item_id_dict[line[2]]] += 1

print('cal cosine_similarity begin...')    
tt = [item_id_dict[i] for i in test_click_df.item_id.unique().tolist()]    
    
t = cosine_similarity(train_data_matrix.T, train_data_matrix[:, tt].T)

print('cal cosine_similarity end')   
dft = pd.DataFrame(t)

from tqdm import tqdm

rdict = dict()
for i in tqdm(range(dft.shape[1])):
    rdict[item_id_dict_inv[tt[i]]] = [item_id_dict_inv[i] for i in dft.iloc[i].sort_values(ascending=False).iloc[1:51].index.tolist()]


test_click_df = test_click_df.sort_values('time').drop_duplicates('user_id')

test_click_df['items'] = test_click_df.item_id.map(lambda x: rdict[x])

df_sub = pd.DataFrame(test_click_df.user_id)

for i in range(50):
    df_sub[f'item_{i}'] = test_click_df['items'].str.get(i)

    
df_sub.to_csv('/Users/luoyonggui/Downloads/underexpose_submit-1.csv', index=False, header=None)

t1 = time.time()
print(f'{t1-t0}s')
print('complete!!!')