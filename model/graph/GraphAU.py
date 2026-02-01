import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22



class GraphAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(GraphAU, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['GraphAU'])

        self.gamma = float(args['-gamma_au'])
        self.decay_base = float(args['-decaying_base'])
        ls = [1.0]
        # self.layers - 1
        for l in range(3 - 1):
            ls.append(ls[-1] * self.decay_base)
        self.decay_weight = torch.tensor(ls).cuda()


        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = GraphAU_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.decay_weight, self.gamma)



    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            uniform_batch = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                batch_loss = model.calculate_loss(user_idx, pos_idx)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model.get_embedding_aggregation(self.n_layers)


            # 评估当前模型
            measure = self.fast_evaluation(epoch)
        # print(uniform_list)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.get_embedding_aggregation(self.n_layers)

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class GraphAU_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, decay_weight, gamma):
        super(GraphAU_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.decay_weight = decay_weight
        self.gamma = gamma



    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def get_embedding_aggregation(self, hops, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(hops):
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

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, user_idx, pos_idx):
        user_e = self.embedding_dict['user_emb'][user_idx]
        item_e = self.embedding_dict['item_emb'][pos_idx]

        align = [self.alignment(user_e, item_e)]
        uniform = [(self.uniformity(user_e) + self.uniformity(item_e)) / 2]

        for i in range(2,self.n_layers+1):
            user_emb_agg , item_emb_agg = self.get_embedding_aggregation(i)
            user_e_i = user_emb_agg[user_idx]
            item_e_i = item_emb_agg[pos_idx]
            align.append((self.alignment(user_e, item_e_i) + self.alignment(item_e, user_e_i))/2)
        align = torch.mean(self.decay_weight * torch.stack(align))

        loss = align + self.gamma * uniform[0]
        return loss






