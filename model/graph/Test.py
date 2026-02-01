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
from torch.nn import Parameter
import faiss

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



class Test(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(Test, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['Test'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        hidden_size = 64


    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        NDCG = []
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                # print(len(user_idx),",",len(pos_idx),",",len(neg_idx))
                # 一个计算原始representation的encoder

                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                cl_loss, sub_loss = self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb)

                # batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + self.cl_rate * cl_loss + self.cl_rate * sub_loss
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + self.cl_rate * cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss',cl_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            recall = float(v)
            NDCG.append(recall)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def draw(self, loss, epoch, label):
        x1 = range(0, epoch + 1)
        y1 = loss
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, '.-', label=label, markevery=20)
        plt.xlabel('epoch')
        # plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + 1 * uniform
        return loss, align, uniform

    def cal_cl_loss(self, idx, user_emb, pos_item_emb):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        '''user_view_sub = self.model.subgraph_user(u_idx)
        item_view_sub = self.model.subgraph_item(i_idx)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        sub_user = InfoNCE(user_view_1[u_idx],user_view_sub,0.2)
        sub_item = InfoNCE(item_view_1[i_idx], item_view_sub,0.2)'''

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        sub_user = 0
        sub_item = 0

        return user_cl_loss + item_cl_loss,  sub_user + sub_item

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

        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

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

    '''def forward_with_subgraph_user(self, user_ids):
        """
        基于子图采样和嵌入更新的前向计算
        - user_ids: 一批用户的 ID
        - perturbed: 是否加入对比学习中的扰动
        """
        all_subgraph_user_embeddings = []
        user_ids = user_ids.cpu()
        for user_id in user_ids:

            # 构建子图邻接矩阵
            subgraph_adj,sampled_items = self.data.sample_subgraph_user(user_id)
            subgraph_adj_tensor = TorchGraphInterface.convert_sparse_mat_to_tensor(subgraph_adj).cuda()

            # 初始化子图上的节点嵌入
            user_emb = self.embedding_dict['user_emb'][user_id].unsqueeze(0)  # 当前用户嵌入
            item_embs = self.embedding_dict['item_emb'][sampled_items]  # 子图物品嵌入
            subgraph_ego_embeddings = torch.cat([user_emb, item_embs], dim=0)


            subgraph_ego_embeddings = torch.sparse.mm(subgraph_adj_tensor, subgraph_ego_embeddings)

            # 提取用户和物品的嵌入
            updated_user_emb = subgraph_ego_embeddings[0]  # 子图中第一个节点是用户
            all_subgraph_user_embeddings.append(updated_user_emb)


        all_subgraph_user_embeddings = torch.stack(all_subgraph_user_embeddings, dim=0)

        return all_subgraph_user_embeddings

    def forward_with_subgraph_item(self, item_ids):
        """
        基于子图采样和嵌入更新的前向计算（以 item_id 为中心）
        - item_ids: 一批物品的 ID
        """
        all_subgraph_item_embeddings = []  # 存储所有子图的物品嵌入
        item_ids = item_ids.cpu()
        for item_id in item_ids:
            # 构建子图邻接矩阵
            subgraph_adj, sampled_users = self.data.sample_subgraph_item(item_id)  # 基于 item_id 采样子图
            subgraph_adj_tensor = TorchGraphInterface.convert_sparse_mat_to_tensor(subgraph_adj).cuda()

            # 初始化子图上的节点嵌入
            # global_item_id = item_id + self.data.user_num  # 全局物品 ID
            item_emb = self.embedding_dict['item_emb'][item_id].unsqueeze(0)  # 当前物品嵌入
            user_embs = self.embedding_dict['user_emb'][sampled_users]  # 子图用户嵌入
            subgraph_ego_embeddings = torch.cat([user_embs, item_emb], dim=0)

            # 在子图上进行图卷积
            subgraph_ego_embeddings = torch.sparse.mm(subgraph_adj_tensor, subgraph_ego_embeddings)

            # 提取物品的嵌入（子图中最后一个节点是物品）
            updated_item_emb = subgraph_ego_embeddings[-1]  # 子图的最后一个节点是中心物品
            all_subgraph_item_embeddings.append(updated_item_emb)

        # 将所有物品嵌入合并成一个张量
        all_subgraph_item_embeddings = torch.stack(all_subgraph_item_embeddings, dim=0)

        return all_subgraph_item_embeddings'''
    def subgraph_user(self,user_idx):
        subgraph_representations = []  # 存储每个用户子图的表征
        user_idx = user_idx.cpu()
        for user_id in user_idx:
            # 1. 采样用户子图
            sampled_items = self.data.sample_subgraph_user(user_id)

            # 2. 获取子图的初始嵌入
            user_emb = self.embedding_dict['user_emb'][user_id].unsqueeze(0)  # 当前用户嵌入
            item_embs = self.embedding_dict['item_emb'][sampled_items]  # 采样物品嵌入
            subgraph_ego_embeddings = torch.cat([user_emb, item_embs], dim=0)  # 拼接用户和物品的嵌入

            # 4. 使用 readout 函数聚合子图表征

            subgraph_representation = subgraph_ego_embeddings.mean(dim=0)  # 取平均
            subgraph_representations.append(subgraph_representation)

        # 5. 返回所有子图的表征
        return torch.stack(subgraph_representations, dim=0)

    def subgraph_item(self,item_idx):
        subgraph_representations = []  # 存储每个用户子图的表征
        item_idx = item_idx.cpu()
        for item_id in item_idx:
            sampled_users = self.data.sample_subgraph_item(item_id)

            # 2. 获取子图的初始嵌入
            item_emb = self.embedding_dict['item_emb'][item_id].unsqueeze(0)  # 当前用户嵌入
            user_embs = self.embedding_dict['user_emb'][sampled_users]  # 采样物品嵌入
            subgraph_ego_embeddings = torch.cat([item_emb, user_embs], dim=0)  # 拼接用户和物品的嵌入

            # 4. 使用 readout 函数聚合子图表征

            subgraph_representation = subgraph_ego_embeddings.mean(dim=0)  # 取平均
            subgraph_representations.append(subgraph_representation)

        # 5. 返回所有子图的表征
        return torch.stack(subgraph_representations, dim=0)





