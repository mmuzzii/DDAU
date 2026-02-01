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



class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)



    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            uniform_batch = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])

                # 统计不同cl loss系数下uniform的变化
                uniform = self.calculate_loss(user_emb, pos_item_emb)


                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)

                batch_loss = cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + rec_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'cl_loss:', cl_loss.item(), 'rec_loss:', rec_loss.item())
                    uniform_batch += uniform
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            print(uniform_batch/(n/100))

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
        # print(uniform_list)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return uniform

    def cal_cl_loss(self, idx, batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.1)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.1)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
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

        # COSTA
        '''if perturbed:
            #print('d:', all_embeddings.shape[1])
            k = torch.tensor(int(all_embeddings.shape[1] * 0.5))
            p = (1 / torch.sqrt(k)) * torch.randn(all_embeddings.shape[1], k).cuda()
            all_embeddings = all_embeddings @ p'''

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        return user_all_embeddings, item_all_embeddings



