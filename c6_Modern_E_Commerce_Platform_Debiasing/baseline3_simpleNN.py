"""
很简单的想法，把用户特征和item特征都输进去，label是下一个item的index 
softmax

训练了十几个周期，效果不理想，而且自己也没有信心这样的模型会收敛！！！
"""
from myutils import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

t0 = time.time()

path = './data_origin/'
train_user_df = pd.read_csv(path+'underexpose_train/underexpose_user_feat.csv', names=['user_id','user_age_level','user_gender','user_city_level'])
train_user_df = train_user_df.drop_duplicates('user_id')
train_user_df.head()

train_user_df = cfe.build_one_hot_features(train_user_df, ['user_age_level', 'user_gender', 'user_city_level'])

train_user_df = train_user_df.reset_index(drop=True)

train_item_df = pd.read_csv(path+'underexpose_train/underexpose_item_feat.csv', sep=r',\s+|,\[|\],\[',names=['item_id']+list(range(256)))
train_item_df.iloc[:, -1] = train_item_df.iloc[:, -1].str.replace(']', '').map(float)

scaler = StandardScaler().fit(train_item_df.iloc[:, 1:])

train_item_df.iloc[:, 1:] = scaler.transform(train_item_df.iloc[:, 1:])

train_item_df.head()

whole_click = pd.read_pickle('data_gen/whole_click_no_val_set.pkl')

test_qtime = pd.read_pickle('data_gen/test_qtime.pkl')

# 统计 test_qtime.user_id 购买过的 item_id
def t(s):
    return s.unique().tolist()
user_items = whole_click[whole_click.user_id.isin(test_qtime.user_id.tolist())].groupby('user_id')['item_id'].agg(t)

def fe_train(train_user_df, train_item_df, whole_click):
    
    item_id_max = whole_click.item_id.nunique()
    uid_max = whole_click.user_id.nunique()
    
    le_item_id = LabelEncoder()
    whole_click.loc[:, 'item_id1'] = le_item_id.fit_transform(whole_click.loc[:, 'item_id'])

    le_uid = LabelEncoder()

    whole_click.loc[:, 'user_id1'] = le_uid.fit_transform(whole_click.loc[:, 'user_id'])
    
    
    ## label ONE 
    # Note: 作为label的item_id并没有经过le_item_id编码!!!
    one = OneHotEncoder()
    one.fit(whole_click.loc[:, 'item_id'].values.reshape((-1, 1)))
    
    whole_click['item_id_before'] = whole_click.groupby('user_id')['item_id1'].shift(1)

    whole_click.dropna(inplace=True)
    whole_click.columns = ['user_id', 'label', 'time', 'item_id1', 'user_id1', 'item_id']

    whole_click.loc[:, 'item_id'] = whole_click.loc[:, 'item_id'].map(int)

    whole_click = pd.merge(whole_click, train_user_df, how='left')

    whole_click = pd.merge(whole_click, train_item_df, how='left')

    whole_click.fillna(0, inplace=True)
    whole_click = whole_click.reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train_index, val_index = train_test_split(list(range(whole_click.shape[0])), test_size=0.02, random_state=14)
    
    whole_click_train = whole_click[whole_click.index.isin(train_index)]

    whole_click_val = whole_click[whole_click.index.isin(val_index)]
    
    
    y_train  = one.transform(whole_click_train.label.values.reshape((-1, 1)))
    y_val  = one.transform(whole_click_val.label.values.reshape((-1, 1)))
    return whole_click_train, whole_click_val, y_train, y_val, item_id_max, uid_max, le_item_id, le_uid, one

def fe_test(train_user_df, train_item_df, whole_click, test_qtime, le_item_id, le_uid):
    # 线下测试集
    whole_click = whole_click[whole_click.user_id.isin(test_qtime.user_id.tolist())].drop_duplicates('user_id', keep='last')
    
    whole_click.loc[:, 'item_id'] = le_item_id.transform(whole_click.loc[:, 'item_id'])

    whole_click.loc[:, 'user_id1'] = le_uid.transform(whole_click.loc[:, 'user_id'])
    
    whole_click.loc[:, 'item_id'] = whole_click.loc[:, 'item_id'].map(int)

    whole_click = pd.merge(whole_click, train_user_df, how='left')

    whole_click = pd.merge(whole_click, train_item_df, how='left')

    whole_click.fillna(0, inplace=True)
    whole_click = whole_click.reset_index(drop=True)

    return whole_click

whole_click_train, whole_click_val, y_train, y_val, item_id_max, uid_max, le_item_id, le_uid, one = fe_train(train_user_df, train_item_df, whole_click.copy())

test_set = fe_test(train_user_df, train_item_df, whole_click.copy(), test_qtime, le_item_id, le_uid)

print('gen model input')
num1 = whole_click.shape[0]
# num1 = 5
features_train1 = [
    whole_click_train['user_id1'].iloc[:num1].values,
    whole_click_train[['user_gender_F','user_gender_M']].iloc[:num1].values.reshape((-1, 1, 2)),
    whole_click_train[['user_age_level_1.0',
                 'user_age_level_2.0',
                 'user_age_level_3.0',
                 'user_age_level_4.0',
                 'user_age_level_5.0',
                 'user_age_level_6.0',
                 'user_age_level_7.0',
                 'user_age_level_8.0']].iloc[:num1].values.reshape((-1, 1, 8)),
    whole_click_train[['user_city_level_1.0',
                 'user_city_level_2.0',
                 'user_city_level_3.0',
                 'user_city_level_4.0',
                 'user_city_level_5.0',
                 'user_city_level_6.0']].iloc[:num1].values.reshape((-1, 1, 6)),
    whole_click_train['item_id'].iloc[:num1].values,
    whole_click_train.iloc[:, -256:-128].iloc[:num1].values.reshape((-1, 1, 128)),
    whole_click_train.iloc[:, -128:].iloc[:num1].values.reshape((-1, 1, 128)),
]
y_train1 = y_train.toarray()[:num1].reshape((-1, 1, item_id_max))




features_val1 = [
    whole_click_val['user_id1'].iloc[:].values,
    whole_click_val[['user_gender_F','user_gender_M']].iloc[:].values.reshape((-1, 1, 2)),
    whole_click_val[['user_age_level_1.0',
                 'user_age_level_2.0',
                 'user_age_level_3.0',
                 'user_age_level_4.0',
                 'user_age_level_5.0',
                 'user_age_level_6.0',
                 'user_age_level_7.0',
                 'user_age_level_8.0']].iloc[:].values.reshape((-1, 1, 8)),
    whole_click_val[['user_city_level_1.0',
                 'user_city_level_2.0',
                 'user_city_level_3.0',
                 'user_city_level_4.0',
                 'user_city_level_5.0',
                 'user_city_level_6.0']].iloc[:].values.reshape((-1, 1, 6)),
    whole_click_val['item_id'].iloc[:].values,
    whole_click_val.iloc[:, -256:-128].iloc[:].values.reshape((-1, 1, 128)),
    whole_click_val.iloc[:, -128:].iloc[:].values.reshape((-1, 1, 128)),
]
y_val1 = y_val.toarray()[:].reshape((-1, 1, item_id_max))
    
    
features_test = [
        test_set['user_id1'].iloc[:].values,
        test_set[['user_gender_F','user_gender_M']].iloc[:].values.reshape((-1, 1, 2)),
        test_set[['user_age_level_1.0',
                 'user_age_level_2.0',
                 'user_age_level_3.0',
                 'user_age_level_4.0',
                 'user_age_level_5.0',
                 'user_age_level_6.0',
                 'user_age_level_7.0',
                 'user_age_level_8.0']].iloc[:].values.reshape((-1, 1, 8)),
        test_set[['user_city_level_1.0',
                 'user_city_level_2.0',
                 'user_city_level_3.0',
                 'user_city_level_4.0',
                 'user_city_level_5.0',
                 'user_city_level_6.0']].iloc[:].values.reshape((-1, 1, 6)),
        test_set['item_id'].iloc[:].values,
        test_set.iloc[:, -256:-128].iloc[:].values.reshape((-1, 1, 128)),
        test_set.iloc[:, -128:].iloc[:].values.reshape((-1, 1, 128)),
]
print('build nn')
embed_dim = 32
def get_inputs():
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')  
#     user_gender = tf.keras.layers.Input(shape=(2,), dtype='int32', name='user_gender')  
#     user_age = tf.keras.layers.Input(shape=(8,), dtype='int32', name='user_age') 
#     user_city = tf.keras.layers.Input(shape=(6,), dtype='int32', name='user_city')
    user_gender = tf.keras.layers.Input(shape=(1,2,), name='user_gender')  
    user_age = tf.keras.layers.Input(shape=(1,8,), name='user_age') 
    user_city = tf.keras.layers.Input(shape=(1,6,), name='user_city')

    item_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_id') 
    item_text = tf.keras.layers.Input(shape=(1, 128,), name='item_text') 
    item_img = tf.keras.layers.Input(shape=(1, 128,), name='item_img') 
#     item_text = tf.keras.layers.Input(shape=(128,), name='item_text') 
#     item_img = tf.keras.layers.Input(shape=(128,), name='item_img') 
    return uid, user_gender, user_age, user_city, item_id, item_text, item_img

def get_user_embedding(uid):
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1, name='uid_embed_layer')(uid)
    return uid_embed_layer

def get_user_feature_layer(uid_embed_layer, user_gender, user_age, user_city):
    #第一层全连接
#     uid_fc_layer = tf.keras.layers.Dense(embed_dim, name="uid_fc_layer", activation='relu')(uid_embed_layer)
#     gender_fc_layer = tf.keras.layers.Dense(embed_dim, name="gender_fc_layer", activation='relu')(gender_embed_layer)
#     age_fc_layer = tf.keras.layers.Dense(embed_dim, name="age_fc_layer", activation='relu')(age_embed_layer)
#     job_fc_layer = tf.keras.layers.Dense(embed_dim, name="job_fc_layer", activation='relu')(job_embed_layer)

    #第二层全连接
#     user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, user_gender, user_age, user_city], 2)  #(?, 1, 128)
    user_combine_layer = tf.keras.layers.concatenate([uid_embed_layer, user_gender, user_age, user_city], 2)  #(?, 1, 128)
    user_combine_layer = tf.keras.layers.Dense(200, activation='relu')(user_combine_layer)  #(?, 1, 200)

    user_combine_layer_flat = tf.keras.layers.Reshape([200], name="user_combine_layer_flat")(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat

def get_item_id_embed_layer(item_id):
    item_id_embed_layer = tf.keras.layers.Embedding(item_id_max, embed_dim, input_length=1, name='item_id_embed_layer')(item_id)
    return item_id_embed_layer

# def get_item_id_embed_layer(item_id, item_id):
#     t = tf.keras.layers.Embedding(item_id_max, embed_dim, input_length=1, name='item_id_embed_layer')
#     item_id_embed_layer = t(item_id)
#     item_id_embed_layer1 = t(item_id)
#     item_id_embed_layer = tf.keras.layers.concatenate([item_id_embed_layer, item_id_embed_layer1], 2)
    return item_id_embed_layer

def get_item_feature_layer(item_id_embed_layer, item_text, item_img):
    #第一层全连接
#     item_id_fc_layer = tf.keras.layers.Dense(embed_dim, name="item_id_fc_layer", activation='relu')(item_id_embed_layer)
#     item_categories_fc_layer = tf.keras.layers.Dense(embed_dim, name="item_categories_fc_layer", activation='relu')(item_categories_embed_layer)

    #第二层全连接
    item_combine_layer = tf.keras.layers.concatenate([item_id_embed_layer, item_text, item_img], 2)  
    item_combine_layer = tf.keras.layers.Dense(200, activation='relu')(item_combine_layer)

    item_combine_layer_flat = tf.keras.layers.Reshape([200], name="item_combine_layer_flat")(item_combine_layer)
    return item_combine_layer, item_combine_layer_flat

def merge_layer(user_combine_layer, item_combine_layer):
    user_item_combine_layer = tf.keras.layers.concatenate([user_combine_layer, item_combine_layer], 2)  
    user_item_combine_layer = tf.keras.layers.Dense(200, activation='relu')(user_item_combine_layer)
    return user_item_combine_layer

uid, user_gender, user_age, user_city, item_id, item_text, item_img = get_inputs()
uid_embed_layer = get_user_embedding(uid)
user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, user_gender, user_age, user_city)

item_id_embed_layer = get_item_id_embed_layer(item_id)
item_combine_layer, item_combine_layer_flat = get_item_feature_layer(item_id_embed_layer, item_text, item_img)

user_item_combine_layer = merge_layer(user_combine_layer, item_combine_layer)

predictions = tf.keras.layers.Dense(item_id_max, activation='softmax')(user_item_combine_layer)

model = tf.keras.Model(
    inputs=[uid, user_gender, user_age, user_city, item_id, item_text, item_img],
    outputs=predictions)

model.summary()
# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

answers = pd.read_csv('./data_gen/debias_track_answer.csv', header=None, index_col=0)
answers = answers.set_index(1)
answers['p'] = answers.values.tolist()

answers = answers['p'].to_dict()

class my_callbacks(tf.keras.callbacks.Callback):
#     def __init__(self, epochs, name):
#         self.epochs = epochs
#         self.name = name

# #     def on_train_begin(self, logs={}):
# #         self.losses = []
# #         print('starttrain')

    def on_epoch_end(self, epoch, logs={}):
        print('\nval begin')
        dfr = pd.DataFrame(model.predict(features_test).reshape((5079, 61903))).T

        top_k = 500
        sim_item_corr = dict()
        u_list = test_set.user_id.tolist()
        dfr_topk = pd.DataFrame(columns = u_list)
        for i in range(dfr.shape[1]):
            st = dfr.iloc[:, i].sort_values(ascending=False).iloc[:top_k].index.tolist()
            st = pd.Series(le_item_id.inverse_transform(st))
            st = st[~st.isin(user_items.loc[u_list[i]])]
            dfr_topk.iloc[:, i] = st.iloc[:50].tolist()
        predictions = dfr_topk.T
        predictions.to_csv(f'data_gen/submit_{epoch}_.csv', header=None)
        predictions['p'] = predictions.values.tolist()
        predictions = predictions['p'].to_dict()
        print(evaluate_each_phase(predictions, answers))
#     def on_train_end(self, logs={}):
#         print("endtrain")


callbacks = [
    my_callbacks(),
  tf.keras.callbacks.ModelCheckpoint(
    'model/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch'
  ),
  # Interrupt training if `val_loss` stops improving for over 2 epochs
#   tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# model.load_weights('model/weights.07-13.33.hdf5')

# Trains for 5 epochs

model.fit(features_train1, y_train1, validation_data=(features_val1, y_val1), epochs=10, batch_size=256, 
          callbacks=callbacks)

# sub.to_csv('/Users/luoyonggui/Downloads/baseline1_itemcf5.csv', header=None)

t1 = time.time()
print(f'{t1-t0}s')
print('complete!!!')