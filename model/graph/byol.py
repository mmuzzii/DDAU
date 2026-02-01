import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt




class byol(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(byol, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['byol'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = byol_Encoder(self.data, self.emb_size, self.eps, self.n_layers)



    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)
                cl_loss = self.byol_loss([user_idx,pos_idx], rec_user_emb, rec_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        '''torch.save(self.user_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_simgclclr.pt')
        torch.save(self.item_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\item_embeddings_simgclclr.pt')'''

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
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

    def byol_loss(self, idx, user_emb, item_emb):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        u_online, i_online = user_emb, item_emb
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, 0.5)
            i_target = F.dropout(i_target, 0.5)

        u_online, i_online = self.model.predictor(u_online), self.model.predictor(i_online)
        u_online = u_online[u_idx]
        i_online = i_online[i_idx]
        u_target = u_target[u_idx]
        i_target = i_target[i_idx]

        loss_u = 1 - cosine_similarity(u_online, u_target.detach(), dim=-1).mean()
        loss_i = 1 - cosine_similarity(i_online, i_target.detach(), dim=-1).mean()

        return loss_u + loss_i


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        user_emb = self.model.predictor(self.user_emb[u])
        item_emb = self.model.predictor(self.item_emb)
        # 内积计算得分
        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()


class byol_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(byol_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        nn.init.xavier_normal_(self.predictor.weight)

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



