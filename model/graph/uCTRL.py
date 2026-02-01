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

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class uCTRL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(uCTRL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['uCTRL'])
        self.n_layers = int(args['-n_layer'])
        # self.gamma = float(args['-gamma'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.model = uCTRL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

        self.users_pop, self.items_pop = self.data.get_user_item_pop_unpop()
        self.users_pop = self.users_pop.cuda()
        self.items_pop = self.items_pop.cuda()

        self.device = torch.device("cuda")

        self.projection_u = torch.empty((2, self.emb_size, self.emb_size), device=self.device)
        self.projection_u = torch.nn.Parameter(self.projection_u, requires_grad=True)
        nn.init.xavier_normal_(self.projection_u)

        self.projection_i = torch.empty((2, self.emb_size, self.emb_size), device=self.device)
        self.projection_i = torch.nn.Parameter(self.projection_i, requires_grad=True)
        nn.init.xavier_normal_(self.projection_i)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                # batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                align_relation, uniform_relation, align_unbias, uniform_unbias, uCTRL_loss = self.calculate_loss(user_emb, pos_item_emb, [user_idx,pos_idx])
                batch_loss = cl_loss + uCTRL_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'cl_loss:', cl_loss.item(),  'uCTRL_loss:', uCTRL_loss.item(),
                          'align_relation',align_relation.item(),'uniform_relation',uniform_relation.item(),'align_unbias',align_unbias.item(),'uniform_unbias',uniform_unbias.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        '''indices = np.random.choice(self.user_emb.shape[0], size=500, replace=False)
        user_point = self.user_emb[indices]
        indices = np.random.choice(self.item_emb.shape[0], size=15000, replace=False)
        item_point = self.item_emb[indices]
        all_emb = torch.cat((item_point, user_point), dim=0)
        print(all_emb.shape)

        tsne = TSNE(n_components=2, init='pca', random_state=42)
        embeddings_cpu = all_emb.cpu()
        vis_dims = tsne.fit_transform(embeddings_cpu.detach().numpy())
        # 创建一个默认颜色的数组
        colors = ['red'] * embeddings_cpu.shape[0]  # 假设默认颜色为蓝色
        # 设置特定点的颜色为红色
        for idx in range(15000):
            colors[idx] = 'blue'
        for i in range(len(colors)):
            plt.scatter(vis_dims[i, 0], vis_dims[i, 1], color=colors[i], alpha=0.5)

        plt.title('t-SNE Visualization')
        plt.show()'''

    def draw(self, loss, epoch, label):
        x1 = range(0, epoch + 1)
        y1 = loss
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, '.-', label=label, markevery=20)
        plt.xlabel('epoch')
        # plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def alignment(self,t_u, t_i, w_u, w_i, o_space=False, alpha=2):

        # t_u, t_i = F.normalize(t_u, dim=-1), F.normalize(t_i, dim=-1)
        if o_space:
            align_loss = (t_u - t_i).norm(p=2, dim=1).pow(alpha)
            return align_loss.mean()

        # relation space compute propensity score
        w = torch.mul(w_u, w_i).sum(dim=1)
        w = torch.sigmoid(w)
        # clipping:限制w的最小值从而限制ipw的最大值
        w = torch.clamp(w, min=0.1)

        align_loss = (1 / w) * (t_u - t_i).norm(p=2, dim=1).pow(alpha)
        return align_loss.mean()

    def uniformity(self, x):
        # x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb, idx):
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()
        # 投影到relation space
        users_relation_emb = torch.einsum("ik,ikj->ij",[user_emb.detach(), self.projection_u[self.users_pop[u_idx]]])
        items_relation_emb = torch.einsum("ik,ikj->ij",[item_emb.detach(), self.projection_i[self.items_pop[i_idx]]])
        users_relation_emb = F.normalize(users_relation_emb, dim=-1)
        items_relation_emb = F.normalize(items_relation_emb, dim=-1)

        o_space = True
        align_relation = self.alignment(t_u=users_relation_emb, t_i=items_relation_emb, w_u=user_emb.detach(),w_i=item_emb.detach(), o_space=o_space)
        uniform_relation = 0.5 * (self.uniformity(x=users_relation_emb) + self.uniformity(x=items_relation_emb)) / 2

        user_emb, item_emb = F.normalize(user_emb, dim=-1), F.normalize(item_emb, dim=-1)
        align_unbias = 5 * self.alignment(t_u=user_emb, t_i=item_emb, w_u=users_relation_emb.detach(),w_i=items_relation_emb.detach())
        uniform_unbias = 0.5 * (self.uniformity(x=user_emb) + self.uniformity(x=item_emb)) / 2

        return align_relation, uniform_relation, align_unbias, uniform_unbias, (align_unbias + uniform_unbias) + (align_relation + uniform_relation)

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class uCTRL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(uCTRL_Encoder, self).__init__()
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
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings