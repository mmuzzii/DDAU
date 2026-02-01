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
import faiss
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import os
from data.augmentor import GraphAugmentor


class DDAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DDAU, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DDAU'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.drop_rate = float(args['-droprate'])
        self.model = DDAU_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.drop_rate)
        self.batch_size = int(args['-batch_size'])

        self.device = torch.device("cuda")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        loss_cl_all = []
        loss_rec_all = []
        NDCG = []
        for epoch in range(self.maxEpoch):
            loss_cl = 0
            loss_rec = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, user_prototypes, item_prototypes = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                directAU_loss, align_loss, uni_loss = self.calculate_loss(user_emb, pos_item_emb, user_prototypes, item_prototypes)
                loss_p, p_au = self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb, user_prototypes, item_prototypes)
                batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb,pos_item_emb) / self.batch_size + directAU_loss + p_au   # + self.cl_rate * loss_n
                # batch_loss = 0.1 * loss_p + l2_reg_loss(self.reg, user_emb,pos_item_emb) / self.batch_size + rec_loss + p_au
                '''loss_cl = self.instance_CL([user_idx,pos_idx])
                batch_loss = self.cl_rate * loss_cl + directAU_loss'''
                '''cl_align, bpr_align, uniform, loss_p = self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb,user_prototypes, item_prototypes)
                batch_loss = 0.3 * cl_align + bpr_align + uniform + 0.1 * loss_p + l2_reg_loss(self.reg, user_emb, pos_item_emb)/self.batch_size'''
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'loss_p:',loss_p.item(), 'p_au', p_au.item())
                    # print('training:', epoch + 1, 'batch', n,'cl_align:',cl_align.item(),'bpr_align:',bpr_align.item(),'uniform:',uniform.item(),'loss_p',loss_p.item(),'batch_loss:',batch_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss',loss_cl.item())
                loss_cl = loss_cl + loss_p.item()
                loss_rec = loss_rec + align_loss.item()

            loss_cl = loss_cl / n
            loss_cl_all.append(loss_cl)
            loss_rec = loss_rec / n
            loss_rec_all.append(loss_rec)

            with torch.no_grad():
                self.user_emb, self.item_emb,_,_ = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            recall = float(v)
            NDCG.append(recall)


        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        print('loss_cl_all', loss_cl_all)
        print('loss_rec_all', loss_rec_all)
        print('NDCG', NDCG)
        '''epochs = range(1, len(loss_cl_all) + 1)
        print('loss_cl_all', loss_cl_all)
        print('loss_rec_all', loss_rec_all)
        print('NDCG', NDCG)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_cl_all, label="CL Loss", marker='o', linestyle='-', color='blue')
        plt.plot(epochs, loss_rec_all, label="Rec Loss", marker='o', linestyle='-', color='red')
        plt.plot(epochs, NDCG, label="NDCG", marker='o', linestyle='-', color='orange')
        # 添加标题和坐标轴标签
        plt.title("Loss and NDCG over Epochs of DDAU in douban", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        # 添加图例
        plt.legend(fontsize=12)
        # 显示网格
        plt.grid(alpha=0.5)
        # 显示图表
        plt.tight_layout()
        plt.show()'''


    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        cosine_loss = 1 - (x * y).sum(dim=1).mean()
        return (x - y).norm(p=2, dim=1).pow(2).mean() + cosine_loss


    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log() + ((1 - x) ** 2).mean()

    def alignment_ori(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()


    def uniformity_ori(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e, user_prototypes, item_prototypes):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = 2 * (self.uniformity(user_e) + self.uniformity(item_e)) / 2 # + (self.uniformity(user_prototypes) + self.uniformity(item_prototypes)/2.)
        # cluster = self.clustering_loss(user_e, user_prototypes) + self.clustering_loss(item_e, item_prototypes)
        loss = align + 0.5 * uniform
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

        # 两个计算增强representation的encoder
        user_view_1, item_view_1, _, _ = self.model(perturbed=True)
        '''dropped_adj1 = self.model.graph_reconstruction()
        user_view_2, item_view_2, _, _ = self.model(perturbed_adj=dropped_adj1)'''
        user_view_2, item_view_2, _, _ = self.model(perturbed=True)

        user_view1 = F.normalize(user_view_1, dim=-1)
        user_view2 = F.normalize(user_view_2, dim=-1)
        item_view1 = F.normalize(item_view_1, dim=-1)
        item_view2 = F.normalize(item_view_2, dim=-1)

        '''logits_u1= F.normalize(logits_u1, dim=-1)
        logits_i1 = F.normalize(logits_i1, dim=-1)
        logits_u2 = F.normalize(logits_u2, dim=-1)
        logits_i2 = F.normalize(logits_i2, dim=-1)'''

        logits_u1 = torch.mm(user_view1[u_idx], user_pro.T)
        logits_i1 = torch.mm(item_view1[i_idx], item_pro.T)
        logits_u2 = torch.mm(user_view2[u_idx], user_pro.T)
        logits_i2 = torch.mm(item_view2[i_idx], item_pro.T)


        u_emb = F.normalize(user_emb, dim=-1)
        i_emb = F.normalize(pos_item_emb, dim=-1)
        '''user_pro = F.normalize(user_pro, dim=-1)
        item_pro = F.normalize(item_pro, dim=-1)'''
        logit1 = torch.mm(u_emb, user_pro.T)
        logit2 = torch.mm(i_emb, item_pro.T)

        sim_u, indices_u = torch.topk(logit1, k=1, dim=1, largest=True)
        sim_i, indices_i = torch.topk(logit2, k=1, dim=1, largest=True)
        u_p = user_pro[indices_u]
        i_p = item_pro[indices_i]
        p_align = self.alignment(u_p, i_p)

        indices_u_flat = indices_u.view(-1)
        unique_indices_u = torch.unique(indices_u_flat)
        indices_i_flat = indices_i.view(-1)
        unique_indices_i = torch.unique(indices_i_flat)
        u_p = user_pro[unique_indices_u]
        i_p = item_pro[unique_indices_i]
        p_uni = (self.uniformity(u_p) + self.uniformity(i_p)) * 0.5

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


        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        '''user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)'''


        return loss_i + loss_u + loss_cf , p_uni + p_align

    def instance_CL(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1,_,_ = self.model(perturbed=True)
        user_view_2, item_view_2,_,_ = self.model(perturbed=True)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

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

class DDAU_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, drop_rate):
        super(DDAU_Encoder, self).__init__()
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
        self.prototype_dict = self._init_prototypes()

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
            'user_prototypes': nn.Parameter(initializer(torch.empty(4000, self.emb_size))),
            'item_prototypes': nn.Parameter(initializer(torch.empty(4000, self.emb_size))),
        })
        return prototypes_dict


    def forward(self, perturbed=False):

        '''user_prototypes = self.prototype_dict['user_prototypes']
        item_prototypes = self.prototype_dict['item_prototypes']

        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']

        sim_u = user_emb @ user_prototypes.transpose(0,1)
        sim_u5, top5_indices = torch.topk(sim_u, k=2, dim=1, largest=True)
        user_sp = user_prototypes[top5_indices] * F.sigmoid(sim_u5).unsqueeze(-1)#.view(sim_u.shape[0], -1)
        user_sp = torch.sum(user_sp, dim=1)
        # user_sp = self.glu(torch.concat([user_emb.unsqueeze(-1), user_sp.unsqueeze(-1)], dim=-1)).squeeze(-1)
        user_emb = user_emb + user_sp

        sim_i = item_emb @ item_prototypes.transpose(0,1)
        sim_i5, top5_indices = torch.topk(sim_i, k=2, dim=1, largest=True)
        item_sp = item_prototypes[top5_indices] * F.sigmoid(sim_i5).unsqueeze(-1)#.view(sim_u.shape[0], -1)
        item_sp = torch.sum(item_sp, dim=1)
        #item_sp = self.glu(torch.concat([item_emb.unsqueeze(-1), item_sp.unsqueeze(-1)], dim=-1)).squeeze(-1)
        item_emb = item_emb + item_sp

        ego_embeddings = torch.cat([user_emb, item_emb], 0)
        all_embeddings = []'''

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

                '''directional_noise = 0.8 * F.normalize(ego_embeddings, dim=-1) + (1 - 0.8) * F.normalize(random_noise, dim=-1)
                ego_embeddings += directional_noise * self.eps

                covariance = torch.cov(ego_embeddings.T)                                                                                                                                                                                                                                                                                                          
                adaptive_noise = torch.mm(F.normalize(random_noise, dim=-1), covariance)
                ego_embeddings += adaptive_noise * self.eps'''

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1) ## jiaquan

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        user_prototypes = self.prototype_dict['user_prototypes']
        item_prototypes = self.prototype_dict['item_prototypes']



        return user_all_embeddings, item_all_embeddings, user_prototypes, item_prototypes

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