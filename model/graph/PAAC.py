import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, InfoNCE_i
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random


class PAAC(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(PAAC, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['PAAC'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = PAAC_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.popular = self.pop_train()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            train_res = {
                'bpr_loss': 0.0,
                'emb_loss': 0.0,
                'cl_loss': 0.0,
                'batch_loss': 0.0,
                'align_loss': 0.0,
            }
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss, rec_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss = self.batch_loss(user_emb, pos_item_emb, neg_item_emb, user_idx, pos_idx)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                train_res['bpr_loss'] += rec_loss.item()
                train_res['emb_loss'] += l2_loss.item()
                train_res['batch_loss'] += batch_loss.item()
                train_res['cl_loss'] += cl_loss.item()

                #if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())


            train_res['bpr_loss'] = train_res['bpr_loss'] / math.ceil(len(self.data.training_data) / self.batch_size)
            train_res['emb_loss'] = train_res['emb_loss'] / math.ceil(len(self.data.training_data) / self.batch_size)
            train_res['batch_loss'] = train_res['batch_loss'] / math.ceil(len(self.data.training_data) / self.batch_size)
            train_res['cl_loss'] = train_res['cl_loss'] / math.ceil(len(self.data.training_data) / self.batch_size)
            print('training:', epoch + 1, 'rec_loss:', train_res['bpr_loss'], 'emb_loss:', train_res['emb_loss'],'cl_loss:', train_res['cl_loss'], 'batch_loss:', train_res['batch_loss'])

            self.user_emb, self.item_emb = self.model()

            G1, G2 = self.split_group_items(self.data, self.popular)
            align_loss = self.alignment_user(self.item_emb[G1], self.item_emb[G2]) * 1000
            optimizer.zero_grad()
            align_loss.backward()
            optimizer.step()
            train_res['align_loss'] += align_loss.item()

            print('align_loss:', train_res['align_loss'])

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def batch_loss(self, user_emb, pos_item_emb, neg_item_emb, u_idx, i_idx):
        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        # l2_loss = l2_reg_loss(0.0001, user_emb, pos_item_emb)
        l2_loss = (torch.norm(user_emb, p=2) + torch.norm(pos_item_emb, p=2)) * 0.0001
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx)
        batch_loss = rec_loss + l2_loss + cl_loss
        return batch_loss, rec_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss

    def cl_loss(self, u_idx, pos_idx):
        # batch里采样
        u_idx = torch.tensor(u_idx)
        batch_pop, batch_unpop = self.split_batch_items(pos_idx, self.popular)
        # print("len:", len(batch_pop), len(batch_unpop))

        batch_users = torch.unique(u_idx).type(torch.long).cuda()
        batch_pop = torch.tensor(batch_pop)
        batch_pop = torch.unique(batch_pop).type(torch.long).cuda()
        batch_unpop = torch.tensor(batch_unpop)
        batch_unpop = torch.unique(batch_unpop).type(torch.long).cuda()

        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        user_cl_loss = InfoNCE(user_view_1[batch_users], user_view_2[batch_users], 0.2) * 10
        item_cl_pop = 0.8 * InfoNCE_i(item_view_1[batch_pop], item_view_2[batch_pop], item_view_2[batch_unpop], 0.2, 0.8)
        item_cl_unpop = 0.2 * InfoNCE_i(item_view_1[batch_unpop], item_view_2[batch_unpop],item_view_2[batch_pop], 0.2, 0.8)
        item_cl_loss = (item_cl_pop + item_cl_unpop) * 10
        cl_loss = user_cl_loss + item_cl_loss

        return cl_loss, user_cl_loss, item_cl_loss

    def alignment_user(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def split_batch_items(self, items, popular):
        g1, g2 = [], []
        # 根据items的受欢迎程度对items排序
        items_sorted = list(np.array(items)[np.argsort(np.array(popular)[items])])
        num = int(len(items_sorted) / 2)
        g1.extend(items_sorted[0:num])
        g2.extend(items_sorted[num:])
        return np.array(g1), np.array(g2)

    def split_group_items(self, data, popular):
        g1, g2 = [], []
        for u in data.training_set_u.keys():
            items = []
            for item in data.training_set_u[u]:
                item = int(item)
                items.append(item)
            # print(items)
            items_sorted = list(np.array(items)[np.argsort(np.array(popular)[items])])
            if len(items) % 2 != 0:
                items_sorted = np.delete(items_sorted, random.sample(range(len(items_sorted)), 1)[0])
            num = int(len(items_sorted) / 2)
            g1.extend(items_sorted[0:num])
            g2.extend(items_sorted[num:])
        return np.array(g1), np.array(g2)

    def pop_train(self):
        # 返回item的交互次数
        train_interation = self.data.item
        ps = pd.Series(train_interation)
        # 计算ps中每个item出现的次数
        vc = ps.value_counts(sort=False)
        vc.sort_index(inplace=True)
        pop_train = []
        num_items = self.data.item_num

        if num_items == len(np.unique(np.array(train_interation))):
            for item in range(num_items):
                # 将vc中当前item的出现次数添加到pop_train里
                pop_train.append(vc[item])
        else:
            for item in range(num_items):
                if item not in list(vc.index):
                    pop_train.append(0)
                else:
                    pop_train.append(vc[item])
        return pop_train

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class PAAC_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(PAAC_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            # 逐层更新embedding
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:
                # 生成和embedding形状相同的噪声，从均匀分布[0,1)中取值
                random_noise = torch.rand_like(ego_embeddings).cuda()
                # 对噪声进行L2归一化*扰动系数(eps超球半径,控制大小);torch.sign(ego_embeddings)控制方向
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        return user_all_embeddings, item_all_embeddings