import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

import faiss
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import os
from data.augmentor import GraphAugmentor

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


class swav2(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(swav2, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['swav2'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.drop_rate = float(args['-droprate'])
        self.model = swav_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.drop_rate)
        self.batch_size = int(args['-batch_size'])

        self.device = torch.device("cuda")




    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        loss_all = []
        NDCG = []
        for epoch in range(self.maxEpoch):
            loss = 0
            '''kmeans_training_data = []
            kmeans_training_data_i = []
            print("Training Clusters:")
            self.model.eval()
            rec_user_emb, rec_item_emb,_,_ = model()
            rec_user_emb_out = rec_user_emb.detach().cpu().numpy()
            rec_item_emb_out = rec_item_emb.detach().cpu().numpy()
            kmeans_training_data.append(rec_user_emb_out)
            kmeans_training_data_i.append(rec_item_emb_out)
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
            kmeans_training_data_i = np.concatenate(kmeans_training_data_i, axis=0)
            for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                cluster.train(kmeans_training_data)
                self.clusters[i] = cluster
            for i, cluster in tqdm(enumerate(self.clusters_i), total=len(self.clusters_i)):
                cluster.train(kmeans_training_data_i)
                self.clusters_i[i] = cluster

            del kmeans_training_data, kmeans_training_data_i
            import gc
            gc.collect()'''

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, user_pro, item_pro = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                directAU_loss, align_loss, uni_loss = self.calculate_loss(user_emb, pos_item_emb)
                loss = self.loss(user_emb,pos_item_emb)
                loss_p , loss_n = self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb, user_pro, item_pro)
                # all_uni = (self.uniformity(user_prototypes) + self.uniformity(item_prototypes))/2
                # batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb)/self.batch_size + 0.01*uni_loss + 10 * orth_loss
                batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size + directAU_loss
                '''cl_align, bpr_align, uniform, loss_p = self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb,user_prototypes, item_prototypes)
                batch_loss = 0.3 * cl_align + bpr_align + uniform + 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb)/self.batch_size'''
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'loss_p:',loss_p.item(),'loss_n',loss_n.item())
                    # print('training:', epoch + 1, 'batch', n,'cl_align:',cl_align.item(),'bpr_align:',bpr_align.item(),'uniform:',uniform.item(),'loss_p',loss_p.item(),'batch_loss:',batch_loss.item())

                loss = loss + loss_p.item()
            loss = loss / n
            loss_all.append(loss)
            with torch.no_grad():
                self.user_emb, self.item_emb,_,_ = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            recall = float(v)
            NDCG.append(recall)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        '''torch.save(self.user_emb, '/emb/user_embeddings_swav222.pt')
        torch.save(self.item_emb, '/emb/item_embeddings_swav222.pt')'''
        print(loss_all)
        print(NDCG)


    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def alignment2(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)

        cosine_loss = 1 - (x * y).sum(dim=1).mean()
        orth_loss = (torch.matmul(x, x.T) - torch.eye(x.size(0), device=x.device)).pow(2).mean() + (
                torch.matmul(y, y.T) - torch.eye(y.size(0), device=y.device)).pow(2).mean()
        return (x - y).norm(p=2, dim=1).pow(2).mean() + cosine_loss

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment2(user_e, item_e)
        uniform = 2 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + 2 * uniform
        # loss = align
        return loss, align, uniform

    def loss(self,user_e, item_e):
        repr_loss = F.mse_loss(user_e, item_e)

        x = user_e - user_e.mean(dim=0)
        y = item_e - item_e.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(64) + \
                   self.off_diagonal(cov_y).pow_(2).sum().div(64)

        loss = (
                25 * repr_loss
                + 25 * std_loss
                + 1 * cov_loss
        )
        # print('repr_loss',repr_loss,'std_loss',std_loss,'cov_loss',cov_loss)
        return loss

    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_cl_loss(self, idx, user_emb, pos_item_emb, user_pro, item_pro):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_pro = F.normalize(user_pro, dim=-1)
        item_pro = F.normalize(item_pro, dim=-1)
        # pro = F.normalize(pro,dim=-1)

        # 两个计算增强representation的encoder
        user_view_1, item_view_1, _,_ = self.model(perturbed=True)
        '''dropped_adj1 = self.model.graph_reconstruction()
        user_view_2, item_view_2, _, _ = self.model(perturbed_adj=dropped_adj1)'''
        user_view_2, item_view_2, _,_ = self.model(perturbed=True)

        user_view1 = F.normalize(user_view_1, dim=-1)
        user_view2 = F.normalize(user_view_2, dim=-1)
        item_view1 = F.normalize(item_view_1, dim=-1)
        item_view2 = F.normalize(item_view_2, dim=-1)

        '''logits_u1= F.normalize(logits_u1, dim=-1)
        logits_i1 = F.normalize(logits_i1, dim=-1)
        logits_u2 = F.normalize(logits_u2, dim=-1)
        logits_i2 = F.normalize(logits_i2, dim=-1)'''

        logits_u1 = torch.mm(user_view1[u_idx], user_pro.T)
        logits_i1 = torch.mm(item_view1[i_idx], user_pro.T)
        logits_u2 = torch.mm(user_view2[u_idx], item_pro.T)
        logits_i2 = torch.mm(item_view2[i_idx], item_pro.T)
        '''logits_u1 = F.softmax(logits_u1, dim=-1)
        logits_i1 = F.softmax(logits_i1, dim=-1)
        logits_u2 = F.softmax(logits_u2, dim=-1)
        logits_i2 = F.softmax(logits_i2, dim=-1)'''


        '''zhelogits_u1 = self.model.prototypes_u(user_view_11[u_idx])
        logits_i1 = self.model.prototypes_i(item_view_11[i_idx])
        logits_u2 = self.model.prototypes_u(user_view_22[u_idx])
        logits_i2 = self.model.prototypes_i(item_view_22[i_idx])'''
        u_emb = F.normalize(user_emb, dim=-1)
        i_emb = F.normalize(pos_item_emb, dim=-1)
        '''user_pro = F.normalize(user_pro, dim=-1)
        item_pro = F.normalize(item_pro, dim=-1)'''
        logit1 = torch.mm(u_emb, user_pro.T)
        logit2 = torch.mm(i_emb, item_pro.T)
        '''logit1 = F.softmax(logit1, dim=-1)
        logit2 = F.softmax(logit2, dim=-1)'''

        with torch.no_grad():
            qu1 = self.distributed_sinkhorn(logits_u1).detach()
            qu2 = self.distributed_sinkhorn(logits_u2).detach()
            qi1 = self.distributed_sinkhorn(logits_i1).detach()
            qi2 = self.distributed_sinkhorn(logits_i2).detach()

            qu = self.distributed_sinkhorn(logit1).detach()
            qi = self.distributed_sinkhorn(logit2).detach()

        '''loss_u = 0
        subloss1 = 0
        u1 = logits_u1 / 0.1
        subloss1 -= torch.mean(torch.sum(qu1 * F.log_softmax(u1, dim=1), dim=1))
        subloss2 = 0
        u2 = logits_u2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu2 * F.log_softmax(u2, dim=1), dim=1))
        loss_u += subloss1 + subloss2 / 2

        loss_i = 0
        subloss1 = 0
        i1 = logits_i1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi1 * F.log_softmax(i1, dim=1), dim=1))
        subloss2 = 0
        i2 = logits_i2 / 0.1
        subloss2 -= torch.mean(torch.sum(qi2 * F.log_softmax(i2, dim=1), dim=1))
        loss_i += subloss1 + subloss2 / 2'''

        loss_u = 0
        subloss1 = 0
        u1 = logits_u1 / 0.1
        subloss1 -= torch.mean(torch.sum(qu2 * F.log_softmax(u1, dim=1), dim=1))
        subloss2 = 0
        u2 = logits_u2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu1 * F.log_softmax(u2, dim=1), dim=1))
        loss_u += subloss1 + subloss2 / 2

        loss_i = 0
        subloss1 = 0
        i1 = logits_i1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi2 * F.log_softmax(i1, dim=1), dim=1))
        subloss2 = 0
        i2 = logits_i2 / 0.1
        subloss2 -= torch.mean(torch.sum(qi1 * F.log_softmax(i2, dim=1), dim=1))
        loss_i += subloss1 + subloss2 / 2

        loss_cf = 0
        subloss1 = 0
        u = logit1 / 0.1
        subloss1 -= torch.mean(torch.sum(qi * F.log_softmax(u, dim=1), dim=1))
        subloss2 = 0
        i = logit2 / 0.1
        subloss2 -= torch.mean(torch.sum(qu * F.log_softmax(i, dim=1), dim=1))
        loss_cf += subloss1 + subloss2 / 2

        '''icl_loss_u = self.pcl_criterion(user_view_1[u_idx1], user_view_2[u_idx1], intents=intents_user, intent_ids=None)
        icl_loss_i = self.pcl_criterion(item_view_1[i_idx1], item_view_2[i_idx1], intents=intents_item,intent_ids=None)'''



        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        user_orth_loss = self.orthogonality_loss(user_pro)
        item_orth_loss = self.orthogonality_loss(item_pro)


        '''u_cl_align = self.alignment(user_view_1[u_idx], user_view_2[u_idx])
        i_cl_align = self.alignment(item_view_1[i_idx], item_view_2[i_idx])'''

        # bpr角度对齐
        '''bpr_align = self.alignment(user_emb, pos_item_emb)

        uniform = (self.uniformity(user_emb) + self.uniformity(pos_item_emb)) / 2'''

        # return u_cl_align + i_cl_align , bpr_align, uniform, loss_i + loss_u + loss_cf
        return (loss_i + loss_u + loss_cf)/3, user_cl_loss + item_cl_loss

    def orthogonality_loss(self, prototypes):
        # prototypes: (num_prototypes, emb_size)
        prototypes_norm = prototypes / prototypes.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.T)
        identity_matrix = torch.eye(similarity_matrix.size(0)).cuda()
        orth_loss = ((similarity_matrix - identity_matrix) ** 2).mean()
        return orth_loss

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

        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

        self.drop_rate = drop_rate

        '''self.prototypes_i = nn.Linear(self.emb_size, 3000, bias=False)
        self.prototypes_u = nn.Linear(self.emb_size, 3000, bias=False)'''
        self.prototype = self._init_prototypes()

        # self.bn = GroupWhitening1d(num_features=64, num_groups=32)
        # self.ii_adj = data.ii_adj
        # self.sparse_ii_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.ii_adj).cuda()

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
        # prototypes = nn.Parameter(initializer(torch.empty(4000, self.emb_size)))
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
        all_embeddings = torch.mean(all_embeddings, dim=1) ## jiaquan

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        '''ii_embeddings = torch.sparse.mm(self.sparse_ii_adj, item_all_embeddings)
        item_all_embeddings = 0.5 * item_all_embeddings + 0.5 * ii_embeddings'''


        user_prototypes = self.prototype['user_prototypes']
        item_prototypes = self.prototype['item_prototypes']

        return user_all_embeddings, item_all_embeddings, user_prototypes,  item_prototypes

class GroupWhitening1d(nn.Module):
    def __init__(self, num_features, num_groups=32, shuffle=False, momentum=0.9):
        super(GroupWhitening1d, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        #self.momentum = momentum
        # self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer('running_mean', None)
        # self.register_buffer("running_covariance", torch.eye(self.num_features))
        self.register_buffer('running_covariance', None)
        self.x_last_batch = None
        self.shuffle = shuffle

    def forward(self, x):
        #import ipdb;ipdb.set_trace()
        G, N, D = self.num_groups, *x.shape
        if self.shuffle:
            new_idx = torch.randperm(x.shape[1])
            reverse_shuffle = torch.argsort(new_idx)
            x = x.t()[new_idx].t()
        x = x.view(N, G, D//G)
        x = x - x.mean(dim=0, keepdim=True)
        x = x.transpose(0,1) # G, N, D//G
        covs = x.transpose(1,2).bmm(x) / (x.size(1) - 1)
        #eigenvalues, eigenvectors = torch.symeig(covs.cpu(), eigenvectors=True, upper=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(covs.cpu())
        S, U = eigenvalues.to(x.device), eigenvectors.to(x.device)
        self.eig = eigenvalues.min()
        whitening_transform = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
        x = x.bmm(whitening_transform)
        if self.shuffle:
            return x.transpose(1,2).flatten(0,1)[reverse_shuffle].t()
        else:
            return x.transpose(0,1).flatten(1)

