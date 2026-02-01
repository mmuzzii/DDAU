import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, clip_loss
import faiss
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import os
from data.augmentor import GraphAugmentor


class swav3(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(swav3, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['swav3'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.drop_rate = float(args['-droprate'])
        self.model = swav_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.drop_rate)
        self.batch_size = int(args['-batch_size'])

        self.device = torch.device("cuda")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

        self.noise_scale = torch.ones(1, 64).cuda()
        self.noise_shift = torch.zeros(1, 64).cuda()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                # rec_user_emb, rec_item_emb, user_prototypes, item_prototypes, mean, std = model()
                rec_user_emb, rec_item_emb, user_prototypes, item_prototypes = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                directAU_loss, align_loss, uni_loss = self.calculate_loss(user_emb, pos_item_emb, user_prototypes, item_prototypes)
                # loss = self.loss(user_emb,pos_item_emb)
                loss_p , loss_n= self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb, user_prototypes, item_prototypes)
                # all_uni = (self.uniformity(user_prototypes) + self.uniformity(item_prototypes))/2
                # batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb)/self.batch_size + 0.01*uni_loss + 10 * orth_loss
                batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size + directAU_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'loss_p:',loss_p.item(),'loss_n',loss_n.item())
                    # print('training:', epoch + 1, 'batch', n,'cl_align:',cl_align.item(),'bpr_align:',bpr_align.item(),'uniform:',uniform.item(),'loss_p',loss_p.item(),'batch_loss:',batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb,_,_ = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def kl_regularizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return regu_loss.sum(1).mean() / self.batch_size

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        cosine_loss = 1 - (x * y).sum(dim=1).mean()
        return (x - y).norm(p=2, dim=1).pow(2).mean() + cosine_loss


    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log() + ((1 - x) ** 2).mean()

    def calculate_loss(self, user_e, item_e, user_prototypes, item_prototypes):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = 2 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2 + (self.uniformity(user_prototypes) + self.uniformity(item_prototypes)) /2
        # cluster = self.clustering_loss(user_e, user_prototypes) + self.clustering_loss(item_e, item_prototypes)
        loss = align + 2 * uniform
        # loss = align
        return loss, align, uniform

    def reparameter(self, mean, std, scale=0.01):


        random_noise = torch.randn_like(std)  # 生成标准正态分布噪声
        distribution_noise = (random_noise * self.noise_scale) + self.noise_shift


        embedding = mean + std * distribution_noise * scale

        user_embeddings, item_embeddings = torch.split(embedding, [self.data.user_num, self.data.item_num])
        return user_embeddings, item_embeddings

    def cal_cl_loss(self, idx, user_emb, pos_item_emb, user_pro, item_pro):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_pro = F.normalize(user_pro, dim=-1)
        item_pro = F.normalize(item_pro, dim=-1)

        # 两个计算增强representation的encoder
        user_view_1, item_view_1, _, _ = self.model(perturbed=True)
        user_view_2, item_view_2, _, _,= self.model(perturbed=True)
        '''user_view_1, item_view_1 = self.reparameter(mean, std)
        user_view_2, item_view_2 = self.reparameter(mean, std)'''

        user_view1 = F.normalize(user_view_1, dim=-1)
        user_view2 = F.normalize(user_view_2, dim=-1)
        item_view1 = F.normalize(item_view_1, dim=-1)
        item_view2 = F.normalize(item_view_2, dim=-1)

        logits_u1 = torch.mm(user_view1[u_idx], user_pro.T)
        logits_i1 = torch.mm(item_view1[i_idx], item_pro.T)
        logits_u2 = torch.mm(user_view2[u_idx], user_pro.T)
        logits_i2 = torch.mm(item_view2[i_idx], item_pro.T)


        u_emb = F.normalize(user_emb, dim=-1)
        i_emb = F.normalize(pos_item_emb, dim=-1)
        logit1 = torch.mm(u_emb, user_pro.T)
        logit2 = torch.mm(i_emb, item_pro.T)

        sim_u, indices_u = torch.topk(logit1, k=1, dim=1, largest=True)
        sim_i, indices_i = torch.topk(logit2, k=1, dim=1, largest=True)

        u_p = user_pro[indices_u]
        i_p = item_pro[indices_i]
        p_align = self.alignment(u_p, i_p)


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
        loss_u += subloss1 + subloss2 /2

        loss_i = 0
        subloss1 = 0
        i1 = logits_i1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi2 * F.log_softmax(i1, dim=1), dim=1))
        subloss2 = 0
        i2 = logits_i2 / 0.1
        subloss2 -= torch.mean(torch.sum(qi1 * F.log_softmax(i2, dim=1), dim=1))
        loss_i += subloss1 + subloss2 /2

        '''loss_cf = 0
        subloss1 = 0
        u = logit1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi * F.log_softmax(u, dim=1), dim=1))
        subloss2 = 0
        i = logit2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu * F.log_softmax(i, dim=1), dim=1))
        loss_cf += subloss1 + subloss2 /2'''

        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)



        # return loss_cf + loss_i + loss_u, user_cl_loss + item_cl_loss
        return p_align + loss_i + loss_u, user_cl_loss + item_cl_loss



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_,_ = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


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

class swav_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, drop_rate):
        super(swav_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.drop_rate = drop_rate

        '''self.prototypes_i = nn.Linear(self.emb_size, 3000, bias=False)
        self.prototypes_u = nn.Linear(self.emb_size, 3000, bias=False)'''
        self.prototype_dict = self._init_prototypes()

        '''self.eps_weight = torch.nn.Parameter(torch.randn(64, 64), requires_grad=True)
        self.eps_bias = torch.nn.Parameter(torch.zeros(64), requires_grad=True)'''

        # self.bn = GroupWhitening1d(num_features=64, num_groups=32)


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def _init_prototypes(self):
        initializer = nn.init.xavier_uniform_
        prototypes_dict = nn.ParameterDict({
            'user_prototypes': nn.Parameter(initializer(torch.empty(2000, self.emb_size))),
            'item_prototypes': nn.Parameter(initializer(torch.empty(2000, self.emb_size))),
        })
        return prototypes_dict


    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):


            # 逐层更新embedding
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:

                # 生成和embedding形状相同的噪声，从均匀分布[0,1)中取值
                random_noise = torch.rand_like(ego_embeddings).cuda()   ## 看一下L2归一化会不会影响均值方差
                # 对噪声进行L2归一化*扰动系数(eps超球半径,控制大小);torch.sign(ego_embeddings)控制方向
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps



            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        user_prototypes = self.prototype_dict['user_prototypes']
        item_prototypes = self.prototype_dict['item_prototypes']

        '''mean = all_embeddings
        logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        std = torch.exp(logstd)'''

        return user_all_embeddings, item_all_embeddings, user_prototypes, item_prototypes
