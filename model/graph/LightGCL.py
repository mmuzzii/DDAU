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
from sklearn.manifold import TSNE

class LightGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCL'])
        self.cl_rate = float(args['-lambda'])
        #aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-dropout'])
        n_layers = int(args['-n_layer'])
        #temp = float(args['-temp'])
        svd_q=int(args['-svd_q'])

        #adj = self.data.norm_adj
        data=self.data
        user_num=self.data.user_num
        item_num=self.data.item_num
        norm_adj=data.norm_adj
        norm_adj=norm_adj[:user_num,user_num:]
        adj = TorchGraphInterface.convert_sparse_mat_to_tensor(norm_adj).cuda()
        print('Performing SVD...')
        # 低秩近似(q=svd_q指定了秩)
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
        print('svd_u:',svd_u.shape,'s:',s.shape,'svd_v:',svd_v.shape)
        # 每列左奇异向量都被相应的奇异值缩放。结果可以被视为原始数据在低维空间中的表示，其中奇异值作为缩放因子调整了各个维度的重要性
        u_mul_s = svd_u @ (torch.diag(s))
        v_mul_s = svd_v @ (torch.diag(s))
        del s
        print('SVD done.')

        self.model = LightGCL_Encoder(self.data, self.emb_size, drop_rate, n_layers, u_mul_s, v_mul_s, svd_u.T, svd_v.T,adj)

    def scipy_sparse_mat_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        recall_list = []
        cl_loss_list = []
        rec_loss_list = []
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb,rec_user_emb_g,rec_item_emb_g = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, rec_item_emb,rec_user_emb_g,rec_item_emb_g )

                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())

            cl_loss_list.append(cl_loss.item())
            rec_loss_list.append(rec_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb, user_emb_g, item_emb_g = self.model()

            measure = self.fast_evaluation(epoch)
            k, v = measure[2].strip().split(':')
            recall = float(v)
            recall_list.append(recall)

        self.draw(cl_loss_list, epoch, 'cl_loss')
        self.draw(rec_loss_list, epoch, 'rec_loss')
        self.draw(recall_list, epoch, 'recall')
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def draw(self, loss, epoch, label):
        x1 = range(0, epoch + 1)
        y1 = loss
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, '.-', label=label, markevery=20)
        plt.xlabel('epoch')
        # plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def cal_cl_loss(self, idx, user_emb, item_emb, user_emb_g, item_emb_g):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        #user_view_1, item_view_1 = self.model(aug=True)
        #user_view_2, item_view_2 = self.model()
        neg_score = torch.log(torch.exp(user_emb_g[u_idx] @ user_emb.T / 0.2).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(item_emb_g[i_idx] @ item_emb.T / 0.2).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((user_emb_g[u_idx] * user_emb[u_idx]).sum(1) / 0.2, -5.0, 5.0)).mean() + (
            torch.clamp((item_emb_g[i_idx] * item_emb[i_idx]).sum(1) / 0.2, -5.0, 5.0)).mean()

        #item_cl_loss = InfoNCE(item_view_1[i_idx], item_emb, 0.2)

        #return user_cl_loss + item_cl_loss
        return -pos_score + neg_score

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb , user_emb_g, item_emb_g= self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LightGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, u_mul_s, v_mul_s, ut, vt, adj):
        super(LightGCL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        #self.temp = temp
        #self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        #self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.n_u=data.user_num
        self.n_i=data.item_num
        self.adj=adj

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict


    def sparse_dropout(self,mat, dropout):
        if dropout == 0.0:
            return mat
        # 获取非0元素索引
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return torch.sparse.FloatTensor(indices, values, size)

    def forward(self,aug=False):
        #ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        #all_embeddings = [ego_embeddings]
        ego_user = self.embedding_dict['user_emb']
        ego_item = self.embedding_dict['item_emb']

        user_all_embeddings_g = []
        item_all_embeddings_g = []

        user_all_embeddings = []
        item_all_embeddings = []


        for k in range(self.n_layers):
            # GNN propagation
            #self.Z_u_list[k] = (torch.spmm(self.sparse_dropout(self.sparse_norm_adj, self.drop_rate), self.E_i_list[self.n_layers - 1]))
            #self.Z_i_list[k] = (torch.spmm(self.sparse_dropout(self.sparse_norm_adj, self.drop_rate).transpose(0, 1), self.E_u_list[self.n_layers - 1]))
            #print("aaaa",self.sparse_norm_adj.shape,"bbbb",self.adj.shape,"cccc",ego_user.shape,"dddd")
            #ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            #z_u = (torch.spmm(self.sparse_dropout(self.sparse_norm_adj, self.drop_rate), ego_user))
            #z_i = torch.sparse.mm(self.sparse_norm_adj, ego_item)
            #z_u,z_i=torch.split(ego_embeddings,[self.data.user_num, self.data.item_num])

            z_u = torch.sparse.mm(self.adj, ego_item)
            z_i = torch.sparse.mm(self.adj.T, ego_user)
            #print('z_u',z_u.shape,'z_i',z_i.shape)

            # svd_adj propagation
            '''vt_ei = self.vt @ self.E_i_list[self.n_layers - 1]
            self.G_u_list[self.n_layers] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[self.n_layers - 1]
            self.G_i_list[self.n_layers] = (self.v_mul_s @ ut_eu)'''

            vt_ei = self.vt @ ego_item
            g_u = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ ego_user
            g_i = self.v_mul_s @ ut_eu


            # aggregate
            #self.E_u_list[self.n_layers] = self.Z_u_list[self.n_layers]
            #self.E_i_list[self.n_layers] = self.Z_i_list[self.n_layers]
            ego_user = z_u
            ego_item = z_i

            #all_embeddings.append(ego_embeddings)
            user_all_embeddings_g.append(g_u)
            item_all_embeddings_g.append(g_i)
            user_all_embeddings.append(ego_user)
            item_all_embeddings.append(ego_item)

        '''self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)'''

        #all_embeddings = torch.stack(all_embeddings, dim=1)
        #all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings_g = torch.stack(user_all_embeddings_g, dim=1)
        user_all_embeddings_g = torch.mean(user_all_embeddings_g, dim=1)
        item_all_embeddings_g = torch.stack(item_all_embeddings_g, dim=1)
        item_all_embeddings_g = torch.mean(item_all_embeddings_g, dim=1)

        user_all_embeddings = torch.stack(user_all_embeddings, dim=1)
        user_all_embeddings = torch.mean(user_all_embeddings, dim=1)
        item_all_embeddings= torch.stack(item_all_embeddings, dim=1)
        item_all_embeddings = torch.mean(item_all_embeddings, dim=1)

        '''if aug==True:
            return user_all_embeddings_g,item_all_embeddings_g
        else:
            #user_all_embeddings, item_all_embeddings = torch.split(all_embeddings,[self.data.user_num, self.data.item_num])
            return user_all_embeddings, item_all_embeddings'''
        return user_all_embeddings,item_all_embeddings,user_all_embeddings_g,item_all_embeddings_g





