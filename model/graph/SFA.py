import torch
import torch.nn as nn
import torch.nn.functional as F
from SimGCL.base.graph_recommender import GraphRecommender
from SimGCL.util.conf import OptionConf
from SimGCL.util.sampler import next_batch_pairwise
from SimGCL.base.torch_interface import TorchGraphInterface
from SimGCL.util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from SimGCL.data.augmentor import GraphAugmentor
import matplotlib.pyplot as plt
import numpy as np


class SFA(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SFA, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SFA'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        self.model = SFA_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()

                '''for id in user_idx:
                    if id == 0:
                        user_0_e = rec_user_emb[id].data.cpu().numpy()'''

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)

                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2,n)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            #print("count:", count)
            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        #user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + 1 * uniform
        return loss, align, uniform

class SFA_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SFA_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
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

    # 0or1进行一次增强，random walk 每层的增强视图不同
    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        # 将其转换为拉普拉斯矩阵
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            # 有扰动(增强)的邻接矩阵则用扰动的来更新embedding
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    # 扰动邻接矩阵为列表，即不同层有不同的邻接矩阵
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2,batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 获得两个增强视图的embedding
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        user_view_1 = user_view_1[u_idx]
        item_view_1 = item_view_1[i_idx]
        user_view_2 = user_view_2[u_idx]
        item_view_2 = item_view_2[i_idx]
        '''H1 = user_view_1.detach()
        H2 = item_view_1.detach()
        H3 = user_view_2.detach()
        H4 = item_view_2.detach()

        user_view_1 = self.spectral_feature_augmentation(H1.T,1).cuda()
        item_view_1 = self.spectral_feature_augmentation(H2.T,1).cuda()
        user_view_2 = self.spectral_feature_augmentation(H3.T,1).cuda()
        item_view_2 = self.spectral_feature_augmentation(H4.T,1).cuda()'''

        r1 = torch.from_numpy(np.random.normal(0, 1, (user_view_1.shape[1], 1))).to(torch.float32).cuda() # 64*1的向量
        # print(r1.dtype," ",user_view_1.dtype)
        r1 = user_view_1.T @ user_view_1 @ r1
        r1_norm=torch.norm(r1) ** 2
        user_view_1 = user_view_1 - ((user_view_1 @ r1) @ r1.T) / r1_norm ** 2

        r2 = torch.from_numpy(np.random.normal(0, 1, (item_view_1.shape[1], 1))).to(torch.float32).cuda()
        r2 = item_view_1.T @ item_view_1 @ r2
        r2_norm = torch.norm(r2) ** 2
        item_view_1 = item_view_1 - ((item_view_1 @ r2) @ r2.T) / r2_norm ** 2

        r3 = torch.from_numpy(np.random.normal(0, 1, (user_view_2.shape[1], 1))).to(torch.float32).cuda()
        r3 = user_view_2.T @ user_view_2 @ r3
        r3_norm = torch.norm(r3) ** 2
        user_view_2 = user_view_2 - ((user_view_2 @ r3) @ r1.T) / r3_norm ** 2

        r4 = torch.from_numpy(np.random.normal(0, 1, (item_view_2.shape[1], 1))).to(torch.float32).cuda()
        r4 = item_view_2.T @ item_view_2 @ r4
        r4_norm = torch.norm(r4) ** 2
        item_view_2 = item_view_2 - ((item_view_2 @ r4) @ r4.T) / r4_norm ** 2

        user_view_1, item_view_1, user_view_2, item_view_2 = [self.project(x) for x in [user_view_1, item_view_1, user_view_2, item_view_2]]

        view1 = torch.cat((user_view_1,item_view_1),0)
        view2 = torch.cat((user_view_2,item_view_2),0)


        return InfoNCE(view1,view2,self.temp)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)