import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import scipy.sparse as sp
import numpy as np
import random
import torch_geometric
import os.path as osp

U = 0
Lambda = 0
mean = 0
std = 0.1
direction = 0
edge1 = 0
weight1 = 0
edge2 = 0
weight2 = 0
use_spec_aug = []
thetas = []
pre_theta = 0
weight_decay_of_spec = 1
Lambda_1 = 0

def adj_to_edgeidx(adj):
    adj=sp.coo_matrix(adj.cpu())
    values=adj.data
    indices=np.vstack((adj.row,adj.col))
    edge_index=torch.LongTensor(indices).cuda()
    edge_weight=torch.FloatTensor(values).cuda()
    return edge_index,edge_weight


def augment_spe(epsilon, epoch, num_epochs):
    global mean, std, Lambda, U, direction, edge1, weight1, edge2, weight2, Lambda_1
    print("mean=", mean)
    positive = torch.where(Lambda > 0)[0]
    negative = torch.where(Lambda < 0)[0]
    print(positive.shape, negative.shape)
    print(torch.mean(direction[positive]), torch.mean(direction[negative]))
    mean += direction * epsilon * Lambda * ((num_epochs - epoch) / num_epochs)
    # noise_1=torch.tensor([torch.normal(m,std) for m in mean]).to(device)
    # print("noise_1==",noise_1)
    Lambda_1 = Lambda + mean

    Lambda_1 = torch.diag(Lambda_1)

    adj1 = torch.matmul(U, Lambda_1)
    adj1 = torch.matmul(adj1, U.T)

    zeros = torch.zeros_like(adj1)
    adj1 = torch.where(adj1 > 0.2, adj1, zeros)
    adj2 = adj1.clone()

    edge1, weight1 = adj_to_edgeidx(adj1)
    edge2, weight2 = adj_to_edgeidx(adj2)
    print("average weight=", torch.mean(weight1))

def augmentation(x,edge_degree,retain_edge,retain_feature,throw_edge,throw_feature,config):
    # drop_edge_rate_1 = 0.4, drop_edge_rate_2 = 0.5
    edge_index_1,edge_weight_1 = drop_edge(edge1, weight1, 0.4 ,0,retain_edge,throw_edge,edge_degree,True)
    edge_index_2,edge_weight_2 = drop_edge(edge2, weight2, 0.5 ,0,retain_edge,throw_edge,edge_degree,True)
    edge_index_1,edge_weight_1 = edge1,weight1
    edge_index_2,edge_weight_2 = edge2,weight2
    # drop_feature_rate_1 = 0.2,drop_feature_rate_2 = 0.4
    x_1=drop_feature(x,0.2 ,0,retain_feature,throw_feature)
    x_2=drop_feature(x,0.4 ,0,retain_feature,throw_feature)
    return x_1,edge_index_1,edge_weight_1,x_2,edge_index_2,edge_weight_2


def drop_edge(edge_index, edge_weight, rate, retain_prob, retain, throw_edge, edge_degree, use_degree_drop):
    num_edge = edge_index.size(1)
    edge_mask = torch.rand(num_edge, device=edge_index.device) >= rate

    index = torch.LongTensor(random.sample(range(len(retain)), (int)(len(retain) * retain_prob))).to(retain.device)
    retain = torch.index_select(retain, 0, index)
    edge_mask[retain] = True
    index = torch.LongTensor(random.sample(range(len(throw_edge)), (int)(len(retain) * 2))).to(retain.device)
    throw = torch.index_select(throw_edge, 0, index)
    edge_mask[throw] = False
    edge_index = edge_index[:, edge_mask]
    edge_weight = edge_weight[edge_mask]
    return edge_index, edge_weight

def drop_feature(x, drop_prob,retain_prob,retain,throw):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) >= drop_prob
    index = torch.LongTensor(random.sample(range(len(retain)), (int)(len(retain)*retain_prob))).to(retain.device)
    retain=torch.index_select(retain,0,index)
    drop_mask[retain]=True
    index = torch.LongTensor(random.sample(range(len(throw)), (int)(len(retain)*2))).to(retain.device)
    throw=torch.index_select(throw,0,index)
    drop_mask[throw]=False
    temp=torch.stack([drop_mask]*x.shape[0])
    return torch.mul(x,temp)

def scipy_to_torch(x):
    values=x.data
    indices=np.vstack((x.row,x.col))
    idx=torch.LongTensor(indices)
    val=torch.FloatTensor(values)
    shape=x.shape
    x=torch.sparse.FloatTensor(idx,val,torch.Size(shape))
    return x

def init_edge(x,edge_index,config):
    adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
    adj = scipy_to_torch(adj)

    global Lambda, U, mean, direction
    # path = osp.join('eigs', config['dataset'], 'lambda.pt')
    path = "C:\\Users\\23386\\PycharmProjects\\rec\\dataset\\yelp2018\\lambda.pt"
    if osp.exists(path) == False:
        adj = adj.cuda()
        (Lambda, U) = torch.eig(adj.to_dense(), eigenvectors=True)
        Lambda = torch.tensor(Lambda).cuda()
        Lambda = Lambda[:, 0].cuda()
        U = U.cuda()

        torch.save(Lambda, path)
        torch.save(U, "C:\\Users\\23386\\PycharmProjects\\rec\\dataset\\yelp2018\\U.pt")

    else:
        Lambda = torch.load(path).cuda()
        U = torch.load("C:\\Users\\23386\\PycharmProjects\\rec\\dataset\\yelp2018\\U.pt").cuda()
    mean = torch.zeros([Lambda.shape[0]]).cuda()
    global thetas, pre_theta
    thetas = []
    pre_theta = 0
    direction = torch.zeros([x.shape[0]]).cuda()
    # epsilon_1 = -0.04
    augment_spe(-0.04, 1, 200)


class spec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(spec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['spec'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = spec_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):

                user_idx, pos_idx, neg_idx = batch
                # print(len(user_idx),",",len(pos_idx),",",len(neg_idx))
                # 一个计算原始representation的encoder

                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], user_emb, pos_item_emb)

                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
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

    def cal_cl_loss(self, idx, user_emb, pos_item_emb):
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        k1 = torch.tensor(int(user_emb.shape[0] * 0.5))
        # 生成高斯随机矩阵
        p1 = (1 / torch.sqrt(k1)) * torch.randn(k1, user_emb.shape[0]).cuda()

        user_view_1 = p1 @ user_view_1[u_idx]
        user_view_2 = p1 @ user_view_2[u_idx]

        u1, u2 = [self.model.project(x) for x in [user_view_1, user_view_2]]

        k2 = torch.tensor(int(pos_item_emb.shape[0] * 0.5))
        # 生成高斯随机矩阵
        p2 = (1 / torch.sqrt(k2)) * torch.randn(k2, pos_item_emb.shape[0]).cuda()

        item_view_1 = p2 @ item_view_1[i_idx]
        item_view_2 = p2 @ item_view_2[i_idx]

        i1, i2 = [self.model.project(x) for x in [item_view_1, item_view_2]]


        # user_view_1, item_view_1, user_view_2, item_view_2 = [self.model.project(x) for x in [user_view_1, item_view_1, user_view_2, item_view_2]]

        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        '''user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)'''
        user_cl_loss = InfoNCE(u1, u2, 0.2)
        item_cl_loss = InfoNCE(i1, i2, 0.2)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class spec_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(spec_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
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

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

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



