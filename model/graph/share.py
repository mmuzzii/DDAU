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
import numpy as np
from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22



class share(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(share, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['share'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.emb_size = 64
        self.model = share_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.fuse = RepresentationFusion(self.emb_size)
        hidden_size = 64
        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.shared_project = self.shared_project.to(self.device)
        self.num_prototypes = 4000


    def train(self):
        model = self.model.cuda()
        fuse = self.fuse.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        loss_all = []
        NDCG = []

        # prototypes = torch.zeros((5000, self.emb_size), device=self.device)
        for epoch in range(self.maxEpoch):
            loss = 0

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                # print(len(user_idx),",",len(pos_idx),",",len(neg_idx))
                # 一个计算原始representation的encoder

                h_shared_user, h_shared_item, h_user_distinct, h_item_distinct = model()
                '''user_shared = self.shared_project(h_shared_user)
                item_shared = self.shared_project(h_shared_item)'''

                user_emb, pos_item_emb, neg_item_emb = h_user_distinct[user_idx], h_item_distinct [pos_idx], h_item_distinct [neg_idx]
                user_emb_share, pos_item_emb_share, neg_item_emb_share = h_shared_user[user_idx], h_shared_item[pos_idx], h_shared_item[neg_idx]

                jsd = self.alignment(user_emb_share,pos_item_emb_share)

                orth_user = self.orthogonality_loss(user_emb_share, user_emb)
                orth_item = self.orthogonality_loss(pos_item_emb_share, pos_item_emb)
                orth_loss = (orth_user + orth_item) * 10



                user, item_pos, item_neg = fuse(user_emb_share,user_emb,pos_item_emb_share,pos_item_emb,neg_item_emb_share,neg_item_emb)

                # loss_ccl = self.cal_ccl_loss([user_idx, pos_idx], user_emb, pos_item_emb, prototypes)

                rec_loss = bpr_loss(user, item_pos, item_neg)

                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)

                batch_loss = jsd  + orth_loss + l2_reg_loss(self.reg, user, item_pos) + rec_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss', rec_loss.item(), 'jsd', jsd.item(), 'orth', orth_loss.item())


            with torch.no_grad():
                user_emb_share, item_emb_share , user_distinct, item_distinct = self.model()
                self.user_emb, self.item_emb = fuse(user_emb_share, user_distinct , item_emb_share, item_distinct)

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            recall = float(v)
            NDCG.append(recall)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        torch.save(self.user_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_CL_book.pt')
        torch.save(self.item_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\item_embeddings_CL_book.pt')

    def orthogonality_loss(self,h_shared, h_distinct):
        # 余弦相似度的绝对值
        orth_loss = torch.abs(F.cosine_similarity(h_shared, h_distinct, dim=-1))
        return torch.mean(orth_loss)

    def jsd_loss(self, h_shared_user, h_shared_item):
        # 将共享表征映射为概率分布（通过 sigmoid）
        p_user = torch.sigmoid(h_shared_user)
        p_item = torch.sigmoid(h_shared_item)

        # 计算混合分布
        m = (p_user + p_item) / 2

        # 计算 KL 散度
        kl_user = F.kl_div(m.log(), p_user, reduction='batchmean')
        kl_item = F.kl_div(m.log(), p_item, reduction='batchmean')

        # JSD loss
        return 0.5 * (kl_user + kl_item)

    def cal_ccl_loss(self, idx, user_emb, pos_item_emb, prototypes):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        prototypes = F.normalize(prototypes, dim=-1)

        _, _, user_view_1, item_view_1 = self.model(perturbed=True)

        u_emb = F.normalize(user_emb, dim=-1)
        i_emb = F.normalize(pos_item_emb, dim=-1)
        user_view1 = F.normalize(user_view_1[u_idx], dim=-1)
        item_view1 = F.normalize(item_view_1[i_idx], dim=-1)

        logits_u1 = torch.mm(user_view1[u_idx], prototypes.T)
        logits_i1 = torch.mm(item_view1[i_idx], prototypes.T)
        logits_u = torch.mm(u_emb[u_idx], prototypes.T)
        logits_i = torch.mm(i_emb[i_idx], prototypes.T)

        with torch.no_grad():
            qu1 = self.distributed_sinkhorn(logits_u1).detach()
            qu = self.distributed_sinkhorn(logits_u).detach()
            qi1 = self.distributed_sinkhorn(logits_i1).detach()
            qi = self.distributed_sinkhorn(logits_i).detach()

        loss_u = 0
        u1 = logits_u1 / 0.1
        loss_u -= torch.mean(torch.sum(qu * F.log_softmax(u1, dim=1), dim=1))

        loss_i = 0
        i1 = logits_i1 / 0.1
        loss_i -= torch.mean(torch.sum(qi * F.log_softmax(i1, dim=1), dim=1))

        loss_ui = 0
        u = logits_u / 0.1
        loss_ui -= torch.mean(torch.sum(qi * F.log_softmax(u, dim=1), dim=1))

        return (loss_i + loss_u+ loss_ui)/3

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

    def cal_cl_loss(self, idx, batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        _,_,user_view_1, item_view_1 = self.model(perturbed=True)
        _,_,user_view_2, item_view_2 = self.model(perturbed=True)



        # user_view_1, item_view_1, user_view_2, item_view_2 = [self.model.project(x) for x in [user_view_1, item_view_1, user_view_2, item_view_2]]

        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        '''u1 = user_view_1[u_idx]
        # print(u1.shape)
        u1 = u1[:,:20]
        u2 = user_view_2[u_idx]
        u2 = u2[:, :20]
        i1 = item_view_1[i_idx]
        i1 = i1[:, :20]
        i2 = item_view_2[i_idx]
        i2 = i2[:, :20]
        user_cl_loss = (InfoNCE(u1, u2, 0.2) + InfoNCE(u2, u1, 0.2))/2
        item_cl_loss = (InfoNCE(i1, i2, 0.2) + InfoNCE(i2, i1, 0.2))/ 2'''
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

    def save(self):
        '''with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()'''

        with torch.no_grad():
            user_emb_share, item_emb_share, user_distinct, item_distinct = self.model()
            self.best_user_emb, self.best_item_emb = self.fuse(user_emb_share, user_distinct, item_emb_share, item_distinct)

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class AttentionFusion(nn.Module):
    def __init__(self, emb_size):
        super(AttentionFusion, self).__init__()
        self.attn_user = nn.Linear(emb_size * 2, 2)  # 注意力权重生成层（用户）
        self.attn_item_pos = nn.Linear(emb_size * 2, 2)  # 注意力权重生成层（正样本物品）
        self.attn_item_neg = nn.Linear(emb_size * 2, 2)  # 注意力权重生成层（负样本物品）

    def forward(self, h_shared_user, h_user_distinct,
                h_shared_item_pos, h_item_distinct_pos,
                h_shared_item_neg = None, h_item_distinct_neg = None):

        # 用户表征融合
        user_concat = torch.cat([h_shared_user, h_user_distinct], dim=-1)  # 拼接共享与特有表征
        attn_user = F.softmax(self.attn_user(user_concat), dim=-1)  # 注意力权重
        h_user_final = attn_user[:, 0:1] * h_shared_user + attn_user[:, 1:2] * h_user_distinct

        # 正样本物品表征融合
        item_pos_concat = torch.cat([h_shared_item_pos, h_item_distinct_pos], dim=-1)
        attn_item_pos = F.softmax(self.attn_item_pos(item_pos_concat), dim=-1)
        h_item_pos_final = attn_item_pos[:, 0:1] * h_shared_item_pos + attn_item_pos[:, 1:2] * h_item_distinct_pos

        if h_shared_item_neg is not None:
            # 负样本物品表征融合
            item_neg_concat = torch.cat([h_shared_item_neg, h_item_distinct_neg], dim=-1)
            attn_item_neg = F.softmax(self.attn_item_neg(item_neg_concat), dim=-1)
            h_item_neg_final = attn_item_neg[:, 0:1] * h_shared_item_neg + attn_item_neg[:, 1:2] * h_item_distinct_neg

            return h_user_final, h_item_pos_final, h_item_neg_final
        else:
            return h_user_final, h_item_pos_final

class RepresentationFusion(nn.Module):
    def __init__(self, emb_size):
        super(RepresentationFusion, self).__init__()
        # 可学习的权重参数
        self.alpha_user = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5
        self.alpha_item = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5
        self.fc_user = nn.Linear(emb_size, emb_size)  # 用户特征变换
        self.fc_item = nn.Linear(emb_size, emb_size)  # 物品特征变换

    def forward(self, h_shared_user, h_user_distinct,
                h_shared_item_pos, h_item_distinct_pos,
                h_shared_item_neg = None, h_item_distinct_neg = None):
        # 用户表征融合
        h_user_shared_transformed = self.fc_user(h_shared_user)
        h_user_final = self.alpha_user * h_user_shared_transformed + (1 - self.alpha_user) * h_user_distinct

        # 物品表征融合
        h_item_shared_transformed = self.fc_item(h_shared_item_pos)
        h_item_pos = self.alpha_item * h_item_shared_transformed + (1 - self.alpha_item) * h_item_distinct_pos

        if h_shared_item_neg is not None:
            item_neg_shared_transformed = self.fc_item(h_shared_item_neg)
            h_item_neg = self.alpha_item * item_neg_shared_transformed + (1 - self.alpha_item) * h_item_distinct_neg
            return h_user_final, h_item_pos, h_item_neg
        else:
            return h_user_final, h_item_pos





class share_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(share_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        # Shared Encoder
        self.shared_gcn1 = nn.Linear(emb_size, emb_size)
        self.shared_gcn2 = nn.Linear(emb_size, emb_size)

        # User Distinct Encoder
        self.user_fc1 = nn.Linear(emb_size, emb_size)
        self.user_fc2 = nn.Linear(emb_size, emb_size)

        # Item Distinct Encoder
        self.item_fc1 = nn.Linear(emb_size, emb_size)
        self.item_fc2 = nn.Linear(emb_size, emb_size)

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

        # 共享表示
        combined_embeddings = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
        h_shared = F.relu(self.shared_gcn1(combined_embeddings))
        h_shared = self.shared_gcn2(h_shared)

        # 用户特有表示
        h_user_distinct = F.relu(self.user_fc1(user_all_embeddings))
        h_user_distinct = self.user_fc2(h_user_distinct)

        # 物品特有表示
        h_item_distinct = F.relu(self.item_fc1(item_all_embeddings))
        h_item_distinct = self.item_fc2(h_item_distinct)

        # 拆分共享表示
        h_shared_user, h_shared_item = torch.split(h_shared, [self.data.user_num, self.data.item_num])

        return h_shared_user, h_shared_item, h_user_distinct, h_item_distinct



