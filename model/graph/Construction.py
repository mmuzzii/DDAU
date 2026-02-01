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

from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

from data.graph import Graph
import numpy as np
from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22



class Construction(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(Construction, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['Construction'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        hidden_size = 64

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.noise_scale = torch.ones(1, 64).cuda()
        self.noise_shift = torch.zeros(1, 64).cuda()


    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        '''optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': self.lRate},
            {'params': [self.log_tau1, self.noise_shift, self.noise_scale], 'lr': self.lRate}  # 将 log_tau1 添加到优化器
        ])'''

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, mean, std, prototypes = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss, ui_pro_loss = self.cal_cl_loss([user_idx, pos_idx], mean, std, user_emb, pos_item_emb, prototypes)
                # ui_pro_loss = self.cal_cl_loss([user_idx, pos_idx], mean, std, user_emb, pos_item_emb, prototypes)
                kl_loss = self.kl_regularizer(mean, std)
                '''uni_loss = self.uniformity(user_emb)+self.uniformity(pos_item_emb)
                ui_align = self.alignment(user_emb, pos_item_emb)
                rec_loss = 4 * uni_loss + ui_align'''
                # uni_loss = (self.uniformity(user_emb) + self.uniformity(pos_item_emb))/2
                batch_loss = self.cl_rate * cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + rec_loss + kl_loss + ui_pro_loss * 0.1
                # batch_loss = l2_reg_loss(self.reg, user_emb,pos_item_emb) + rec_loss + kl_loss + ui_pro_loss * 0.1 + uni_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'pro_loss:', ui_pro_loss.item(), 'cl_loss', cl_loss.item())
                    # user_emb, item_emb, _, _,_ = self.model()

            '''_,_,_,_,prototypes = self.model()
            pro_uni_loss = self.uniformity(prototypes)
            optimizer.zero_grad()
            pro_uni_loss.backward()
            optimizer.step()
            print('training:', epoch + 1,'pro_uni_loss',pro_uni_loss.item())'''

            with torch.no_grad():
                self.user_emb, self.item_emb,_,_,_= self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def reparameter(self, mean, std, scale=0.01):


        random_noise = torch.randn_like(std)  # 生成标准正态分布噪声
        distribution_noise = (random_noise * self.noise_scale) + self.noise_shift


        embedding = mean + std * distribution_noise * scale

        user_embeddings, item_embeddings = torch.split(embedding, [self.data.user_num, self.data.item_num])
        return user_embeddings, item_embeddings

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = 2 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + 1 * uniform
        return loss, align, uniform

    def cal_cl_loss(self, idx, mean, std, user_emb, item_emb, pro):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder

        user_embeddings_1, item_embeddings_1 = self.reparameter(mean, std)
        user_embeddings_2, item_embeddings_2 = self.reparameter(mean, std)

        user_view_1 = user_embeddings_1[u_idx]
        item_view_1 = item_embeddings_1[i_idx]
        user_view_2 = user_embeddings_2[u_idx]
        item_view_2 = item_embeddings_2[i_idx]


        cl_u = InfoNCE(user_view_1,user_view_2,0.2)
        cl_i = InfoNCE(item_view_1,item_view_2,0.2)
        '''cl_align_u = self.alignment(user_view_1,user_view_2)
        cl_align_i = self.alignment(item_view_1,item_view_2)'''

        ui_pro_loss = self.prototype_loss(user_view_1, item_view_1, user_view_2, item_view_2, user_emb, item_emb, pro)

        return cl_u + cl_i, ui_pro_loss

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        cosine_loss = 1 - (x * y).sum(dim=1).mean()
        return (x - y).norm(p=2, dim=1).pow(2).mean() + cosine_loss

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log() + ((1 - x) ** 2).mean()

    def align_uni_loss(self, user_e, item_e, pro):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = 2 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2 + self.uniformity(pro)
        # cluster = self.clustering_loss(user_e, user_prototypes) + self.clustering_loss(item_e, item_prototypes)
        loss = align + 4 * uniform
        # loss = align
        return loss, align, uniform

    def prototype_loss(self,user_emb1, item_emb1, user_emb2, item_emb2, user_emb, item_emb, pro):

        pro = F.normalize(pro, dim=-1)

        user_view1 = F.normalize(user_emb1, dim=-1)
        user_view2 = F.normalize(user_emb2, dim=-1)
        item_view1 = F.normalize(item_emb1, dim=-1)
        item_view2 = F.normalize(item_emb2, dim=-1)

        '''logits_u1= F.normalize(logits_u1, dim=-1)
        logits_i1 = F.normalize(logits_i1, dim=-1)
        logits_u2 = F.normalize(logits_u2, dim=-1)
        logits_i2 = F.normalize(logits_i2, dim=-1)'''

        logits_u1 = torch.mm(user_view1, pro.T)
        logits_i1 = torch.mm(item_view1, pro.T)
        logits_u2 = torch.mm(user_view2, pro.T)
        logits_i2 = torch.mm(item_view2, pro.T)

        u_emb = F.normalize(user_emb, dim=-1)
        i_emb = F.normalize(item_emb, dim=-1)
        '''user_pro = F.normalize(user_pro, dim=-1)
        item_pro = F.normalize(item_pro, dim=-1)'''
        logit1 = torch.mm(u_emb, pro.T)
        logit2 = torch.mm(i_emb, pro.T)

        with torch.no_grad():
            qu1 = self.distributed_sinkhorn(logits_u1).detach()
            qu2 = self.distributed_sinkhorn(logits_u2).detach()
            qi1 = self.distributed_sinkhorn(logits_i1).detach()
            qi2 = self.distributed_sinkhorn(logits_i2).detach()

            qu = self.distributed_sinkhorn(logit1).detach()
            qi = self.distributed_sinkhorn(logit2).detach()

        loss_u = 0
        subloss1 = 0
        u1 = logits_u1 / 0.1
        subloss1 -= torch.mean(torch.sum(qu2 * F.log_softmax(u1, dim=1), dim=1))
        subloss2 = 0
        u2 = logits_u2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu1 * F.log_softmax(u2, dim=1), dim=1))
        loss_u += (subloss1 + subloss2) / 2

        loss_i = 0
        subloss1 = 0
        i1 = logits_i1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi2 * F.log_softmax(i1, dim=1), dim=1))
        subloss2 = 0
        i2 = logits_i2 / 0.1
        subloss2 -= torch.mean(torch.sum(qi1 * F.log_softmax(i2, dim=1), dim=1))
        loss_i += (subloss1 + subloss2) / 2

        loss_cf = 0
        subloss1 = 0
        u = logit1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi * F.log_softmax(u, dim=1), dim=1))
        subloss2 = 0
        i = logit2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu * F.log_softmax(i, dim=1), dim=1))
        loss_cf += (subloss1 + subloss2) / 2


        return (loss_cf + loss_i + loss_u)/3


    def kl_regularizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return regu_loss.sum(1).mean() / self.batch_size

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        # print(out.shape)
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q
        # sinkhorn_iterations = 3
        sinkhorn_iterations = 3
        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_,_,_ = self.model.forward()

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


        self.eps_weight = torch.nn.Parameter(torch.randn(64,64), requires_grad=True)
        self.eps_bias = torch.nn.Parameter(torch.zeros(64), requires_grad=True)

        self.prototype = self._init_prototypes()


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def _init_prototypes(self):
        initializer = nn.init.xavier_uniform_
        prototypes = nn.Parameter(initializer(torch.empty(2000, self.emb_size)))

        return prototypes

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

        low = self.calculate_low_frequency(all_embeddings,self.norm_adj)
        high = self.calculate_high_frequency(all_embeddings,self.norm_adj)
        all_embeddings = self.combine_signals(low, high,self.norm_adj)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        mean = all_embeddings
        logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        std = torch.exp(logstd)

        prototypes = self.prototype

        return user_all_embeddings, item_all_embeddings, mean, std, prototypes

    def calculate_low_frequency(self, emb, adjacency):
        omega = 1


        laplacian =  adjacency

        # 特征分解：获取特征值 (Lambda) 和特征向量 (Phi)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # 低频信号计算：S^L = Phi[(omega + 1)I - Lambda]Phi^T x
        low_frequency_signal = eigenvectors @ ((omega + 1 - eigenvalues).unsqueeze(1) * (eigenvectors.T @ emb))
        return low_frequency_signal

    def calculate_high_frequency(self, emb, adjacency):
        omega = 1

        laplacian = adjacency

        # 特征分解：获取特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        # 高频信号计算：S^H = Phi[(omega - 1)I + Lambda]Phi^T x
        high_frequency_signal = eigenvectors @ ((omega - 1 + eigenvalues).unsqueeze(1) * (eigenvectors.T @ emb))
        return high_frequency_signal

    def combine_signals(self, emb, low_frequency_signal, high_frequency_signal, adjacency):

        omega = 0.5
        # q_matrix: 软标签
        degree = torch.sum(adjacency, dim=1)  # 计算每个节点的度数
        degree_inv = 1.0 / degree

        combined_signal = []
        low_frequency_contrib = adjacency @ low_frequency_signal  # 直接邻接矩阵乘特征
        high_frequency_contrib = adjacency @ high_frequency_signal
        low_frequency_contrib = low_frequency_contrib / degree.unsqueeze(1)  # 按度归一化
        high_frequency_contrib = high_frequency_contrib / degree.unsqueeze(1)

        combined_signal = omega * emb + low_frequency_contrib + high_frequency_contrib
        return combined_signal




