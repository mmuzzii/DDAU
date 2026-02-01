##########################
# This code take SGL, implemented by Coder-Yu on Github, as the backbone.
##########################


from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from sklearn.decomposition import NMF
import numpy as np


# Paper: self-supervised graph learning for recommendation. SIGIR'21


class pretrain(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(pretrain, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['pretrain'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        hidden_size = 64

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.noise_scale = nn.Parameter(torch.ones(1, 64)).cuda()  # 可学习缩放
        self.noise_shift = nn.Parameter(torch.zeros(1, 64)).cuda()

    def _pre_train(self):
        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(15):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, mean, std, prototypes = pre_trained_model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], mean, std)
                cl_loss, pro_cl_loss = self.cal_cl_loss([user_idx, pos_idx], mean, std, user_emb, pos_item_emb, prototypes)

                batch_loss = cl_loss + pro_cl_loss + self.kl_regularizer(mean, std)

                # Backward and optimize
                optimizer.zero_grad()
                # 最后一个周期保留训练图
                if epoch == self.maxEpoch - 1:
                    batch_loss.backward(retain_graph=True)
                else:
                    batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

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

        cl_u = InfoNCE(user_view_1, user_view_2, 0.2)
        cl_i = InfoNCE(item_view_1, item_view_2, 0.2)
        '''cl_align_u = self.alignment(user_view_1,user_view_2)
        cl_align_i = self.alignment(item_view_1,item_view_2)'''

        ui_pro_loss = self.prototype_loss(user_view_1, item_view_1, user_view_2, item_view_2, user_emb, item_emb, pro)

        return cl_u + cl_i, ui_pro_loss

    def reparameter(self, mean, std, scale=0.01):


        random_noise = torch.randn_like(std)  # 生成标准正态分布噪声
        distribution_noise = (random_noise * self.noise_scale) + self.noise_shift


        embedding = mean + std * distribution_noise * scale

        user_embeddings, item_embeddings = torch.split(embedding, [self.data.user_num, self.data.item_num])
        return user_embeddings, item_embeddings

    def kl_regularizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return regu_loss.sum(1).mean() / self.batch_size

    def train(self):
        self._pre_train()

        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                rec_user_emb, rec_item_emb, mean, std, prototypes = model()
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:',directAU_loss.item(), 'align_loss:', align_loss.item(), 'uniform_loss:', uni_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb,_,_,_ = self.model()

            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def prototype_loss(self, user_emb1, item_emb1, user_emb2, item_emb2, user_emb, item_emb, pro):

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

        return (loss_cf + loss_i + loss_u) / 3

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
            self.best_user_emb, self.best_item_emb, _, _, _ = self.model.forward()

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
        prototypes = nn.Parameter(initializer(torch.empty(4000, self.emb_size)))

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

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        mean = all_embeddings
        logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        std = torch.exp(logstd)

        prototypes = self.prototype

        return user_all_embeddings, item_all_embeddings, mean, std, prototypes