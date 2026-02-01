import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

from util.MinNormSolver import MinNormSolver
import torch.optim as optim
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



class prototype(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(prototype, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['prototype'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = prototype_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                # print(len(user_idx),",",len(pos_idx),",",len(neg_idx))
                # 一个计算原始representation的encoder

                rec_user_emb, rec_item_emb, user_prototypes, item_prototypes = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                u_cl_align, i_cl_align, bpr_align, uniform, i_loss = self.align_uni_loss([user_idx, pos_idx],user_emb,pos_item_emb, rec_user_emb, rec_item_emb, user_prototypes, item_prototypes)

                grads_A = []
                grads_B = []
                grads_C = []
                optimizer.zero_grad()
                (u_cl_align+i_cl_align+bpr_align).backward(retain_graph=True)
                grads_A.append(model.embedding_dict['user_emb'].grad.clone())
                grads_A.append(model.embedding_dict['item_emb'].grad.clone())

                '''optimizer.zero_grad()
                bpr_align.backward(retain_graph=True)
                grads_B.append(model.embedding_dict['user_emb'].grad.clone())
                grads_B.append(model.embedding_dict['item_emb'].grad.clone())'''

                optimizer.zero_grad()
                uniform.backward(retain_graph=True)
                grads_C.append(model.embedding_dict['user_emb'].grad.clone())
                grads_C.append(model.embedding_dict['item_emb'].grad.clone())

                solver = MinNormSolver()
                sol, min_norm = solver.find_min_norm_element([grads_A, grads_C])
                # batch_loss = self.uncertainty_to_weigh_losses(total_loss) + l2_reg_loss(self.reg, user_emb,pos_item_emb) / self.batch_size

                # batch_loss = (u_cl_align + i_cl_align)*0.3 + uniform + bpr_align + 0.1*i_loss + l2_reg_loss(self.reg, user_emb,pos_item_emb) / self.batch_size

                # rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)

                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()

                combined_loss = sol[0] * (u_cl_align+i_cl_align+bpr_align) + sol[1] * uniform
                combined_loss.backward()


                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'u_cl_align:', u_cl_align.item(),  'i_cl_align:', i_cl_align.item(), 'bpr_align:', bpr_align.item(), 'uniform:', uniform.item(),'combine_Loss:',combined_loss.item())

                '''with torch.no_grad():
                    normalized_user_prototypes = F.normalize(
                        self.model.prototypes_dict['user_prototypes'], p=2, dim=1)
                    normalized_item_prototypes = F.normalize(
                        self.model.prototypes_dict['item_prototypes'], p=2, dim=1)

                    self.model.prototypes_dict['user_prototypes'] = nn.Parameter(normalized_user_prototypes)
                    self.model.prototypes_dict['item_prototypes'] = nn.Parameter(normalized_item_prototypes)'''

            with torch.no_grad():
                self.user_emb, self.item_emb,_,_ = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)


        '''self.draw(cl_loss_list,epoch,'cl_loss')
        self.draw(rec_loss_list, epoch,'rec_loss')
        self.draw(recall_list,epoch,'recall')'''
        # self.draw(align_list, epoch, 'align_loss')
        # self.draw(uni_list, epoch, 'uniform_loss')

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


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
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)


        # user_view_1, item_view_1, user_view_2, item_view_2 = [self.model.project(x) for x in [user_view_1, item_view_1, user_view_2, item_view_2]]

        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_ ,_ = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def align_uni_loss(self, idx, user_emb, item_emb,user_all, item_all, user_prototypes, item_prototypes):
        u_idx = torch.unique(torch.Tensor(idx[0])).type(torch.long).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1])).type(torch.long).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        # cl角度对齐
        u_cl_align = self.alignment(user_view_1[u_idx], user_view_2[u_idx])
        i_cl_align = self.alignment(item_view_1[i_idx], item_view_2[i_idx])

        user_z = torch.cat((user_all[u_idx], user_view_1[u_idx]), dim=0)
        item_z = torch.cat((item_all[i_idx], item_view_1[i_idx]), dim=0)

        '''iu_loss = self.proto_loss(user_z, user_prototypes, temperature=0.05)
        ii_loss = self.proto_loss(item_z, item_prototypes, temperature=0.05)'''
        iu_loss = self.proto_loss(user_z, user_prototypes, user_all[u_idx],temperature=0.05)
        ii_loss = self.proto_loss(item_z, item_prototypes, item_all[i_idx], temperature=0.05)

        # bpr角度对齐
        bpr_align = self.alignment(user_emb, item_emb)

        # cl角度均匀
        uniform = 2 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2



        return u_cl_align, i_cl_align, bpr_align, uniform, iu_loss + ii_loss

    def proto_loss(self, z, prototypes, e,temperature=0.1):
        # Compute scores between embeddings and prototypes
        # 3862x64 and 2000x64
        # print(prototypes.shape," ",z.shape)
        scores = torch.mm(z, prototypes.T)

        score_t = scores[: z.size(0) // 2]
        score_s = scores[z.size(0) // 2:]

        c_t = prototypes[score_t.max(dim=1)[1]]
        c_s = prototypes[score_s.max(dim=1)[1]]
        # 拉近原始视图和增强视图的原型
        # loss_au = self.alignment(c_s, c_t) + self.alignment(e, c_t) + self.uniformity(prototypes)
        loss_au = self.alignment(c_s, c_t) + self.uniformity(prototypes)
        # Apply the Sinkhorn-Knopp algorithm to get soft cluster assignments
        q_t = self.sinkhorn_knopp(score_t)
        q_s = self.sinkhorn_knopp(score_s)

        log_p_t = torch.log_softmax(score_t / temperature + 1e-7, dim=1)
        log_p_s = torch.log_softmax(score_s / temperature + 1e-7, dim=1)

        # Calculate cross-entropy loss
        loss_t = torch.mean(
            -torch.sum(
                q_s * log_p_t,
                dim=1,
            )
        )
        loss_s = torch.mean(
            -torch.sum(
                q_t * log_p_s,
                dim=1,
            )
        )
        # proto loss is the average of loss_t and loss_s
        proto_loss = (loss_t + loss_s) / 2
        return proto_loss + loss_au

    def sinkhorn_knopp(self, scores, epsilon=0.05, n_iters=3):
        with torch.no_grad():
            scores_max = torch.max(scores, dim=1, keepdim=True).values
            scores_stable = scores - scores_max
            Q = torch.exp(scores_stable / epsilon).t()
            Q /= (Q.sum(dim=1, keepdim=True) + 1e-8)

            K, B = Q.shape
            u = torch.zeros(K).to(scores.device)
            r = torch.ones(K).to(scores.device) / K
            c = torch.ones(B).to(scores.device) / B

            for _ in range(n_iters):
                u = Q.sum(dim=1)
                Q *= (r / (u + 1e-8)).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)

            Q = (Q / Q.sum(dim=0, keepdim=True)).t()
            return Q


class prototype_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(prototype_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.prototypes_dict = self._init_prototypes()
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

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
            'user_prototypes': nn.Parameter(initializer(torch.empty(1000, self.emb_size))),
            'item_prototypes': nn.Parameter(initializer(torch.empty(1000, self.emb_size))),
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

        user_prototypes = self.prototypes_dict['user_prototypes']
        item_prototypes = self.prototypes_dict['item_prototypes']
        if perturbed:
            return user_all_embeddings, item_all_embeddings
        else:
            return user_all_embeddings, item_all_embeddings, user_prototypes, item_prototypes


