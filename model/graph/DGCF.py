import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class DGCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DGCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DGCF'])
        self.n_layers = int(args['-n_layer'])

        self.norm_adj = self.data.norm_adj

        self.model = DGCF_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()



class DGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(DGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers

        self.n_user = self.data.user_num
        self.n_item = self.data.item_num

        self.norm_adj = data.norm_adj
        self.ui_adj = self.data.ui_adj

        # all_h_list:行索引(u+i)(边的头节点索引)；all_t_list：列索引(边的尾节点索引)；all_v_list：data
        # all_h_list, all_t_list, self.all_v_list = self.load_adjacency_list_data(self.ui_adj)
        # generate intermediate data
        user_item_interaction = self.ui_adj[:self.n_user, self.n_user:self.n_user + self.n_item]
        print(user_item_interaction.shape)
        interaction_matrix = user_item_interaction.tocoo()
        row = interaction_matrix.row.tolist()
        col = interaction_matrix.col.tolist()
        col = [item_index + self.n_user for item_index in col]
        all_h_list = row + col  # row.extend(col)
        print("all_h_list",max(all_h_list))
        all_t_list = col + row  # col.extend(row)
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)

        self.all_h_list = torch.LongTensor(all_h_list).cuda()
        self.all_t_list = torch.LongTensor(all_t_list).cuda()
        # 边到头节点的映射，第一行为头节点索引，第二行为边索引，每一列表示头节点与对应的边
        self.edge2head = torch.LongTensor([all_h_list, edge_ids]).cuda()
        self.head2edge = torch.LongTensor([edge_ids, all_h_list]).cuda()
        self.tail2edge = torch.LongTensor([edge_ids, all_t_list]).cuda()
        val_one = torch.ones_like(self.all_h_list).float().cuda()
        num_node = self.n_user + self.n_item
        self.edge2head_mat = self._build_sparse_tensor(
            self.edge2head, val_one, (num_node, num_edge)
        )
        self.head2edge_mat = self._build_sparse_tensor(
            self.head2edge, val_one, (num_edge, num_node)
        )
        self.tail2edge_mat = self._build_sparse_tensor(
            self.tail2edge, val_one, (num_edge, num_node)
        )
        self.num_edge = num_edge
        self.num_node = num_node

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.n_factors = 4
        self.n_iterations = 2
        self.pick_level = 1e10
        self.softmax = torch.nn.Softmax(dim=1)

    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse.FloatTensor(indices, values, size).cuda()

    def build_matrix(self, A_values):
        # A_{hat} = D^{-0.5} * A * D^{-0.5}
        norm_A_values = self.softmax(A_values)
        factor_edge_weight = []
        for i in range(self.n_factors):
            # 每个边的第i个因子，unsqueeze(1) 增加一个维度，使其形状变为 (num_edge, 1)
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # 头节点的度 (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            # 确保 d_values 中的值不会低于 1e-8
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                print("d_values：", torch.min(d_values), torch.max(d_values))
            # D ^ {-0.5}
            d_values = 1.0 / torch.sqrt(d_values)

            # 分别计算与头节点和尾节点相关的边权重
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)

            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        A_values = torch.ones((self.num_edge, self.n_factors), device='cuda', requires_grad=True)
        for k in range(self.layers):
            layer_embeddings = []
            # split the input embedding table
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.n_factors, 1)

            # 每层t次迭代(embedding迭代+intent-aware graph迭代)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)

                for i in range(0, self.n_factors):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # 根据尾节点，更新边的embedding：(num_edge, dim / n_factors)
                    edge_val = torch.sparse.mm(
                        self.tail2edge_mat, ego_layer_embeddings[i]
                    )
                    # 应用权重调整embedding
                    edge_val = edge_val * edge_weight
                    # 使用边的embedding更新头节点的embedding
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)

                    iter_embeddings.append(factor_embeddings)
                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # print(factor_embeddings.shape," ",self.all_h_list.shape)
                    head_factor_embeddings = torch.index_select(
                        factor_embeddings, dim=0, index=self.all_h_list
                    )
                    tail_factor_embeddings = torch.index_select(
                        ego_layer_embeddings[i], dim=0, index=self.all_t_list
                    )

                    # .... constrain the vector length
                    head_factor_embeddings = F.normalize(
                        head_factor_embeddings, p=2, dim=1
                    )
                    tail_factor_embeddings = F.normalize(
                        tail_factor_embeddings, p=2, dim=1
                    )

                    # get the attentive weights: u_k * tanh(i_k)
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings),
                        dim=1,
                        keepdim=True,
                    )
                    A_iter_values.append(A_factor_values)
                # 沿factor维度整合：(num_edge, n_factors)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)
            ego_embeddings = side_embeddings
            # concatenate outputs of all layers，整合t次迭代
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        # 整合所有层
        # (num_node, n_layer + 1, embedding_size)
        all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, embedding_size)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)

        u_g_embeddings = all_embeddings[: self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        # print("u_embedding size:",u_g_embeddings.shape," i_embedding size:",i_g_embeddings.shape)

        return u_g_embeddings, i_g_embeddings





