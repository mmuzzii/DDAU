def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


from random import shuffle,randint,choice,sample
import numpy as np


def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    # print(training_data)
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    # 循环生成批次数据
    while ptr < data_size:
        # 计算本批次结束的索引
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        # 为每个用户的正样本选择负样本(不在用户的历史交互中)
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                # neg在用户历史交互中则重新选择
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        # 产出一批用户ID、正样本物品ID和负样本物品ID(n_neg=1,一个正样本对应一个负样本)
        yield u_idx, i_idx, j_idx

