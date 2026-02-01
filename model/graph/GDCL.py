import torch
import torch.nn as nn
import torch.nn.functional as F
from SimGCL.base.graph_recommender import GraphRecommender
from SimGCL.util.conf import OptionConf
from SimGCL.util.sampler import next_batch_pairwise
from SimGCL.base.torch_interface import TorchGraphInterface
from SimGCL.util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch.nn import Parameter

class GDCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(GDCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['GDCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = GDCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

        self.cluster_loss = nn.KLDivLoss(size_average=False)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.train_infonce(model,optimizer)
        self.train_bpr(model,optimizer)

    def train_infonce(self,model,optimizer):

        for epoch in range(10):
            self.print_model_parameters(model, "each_epoch_info")
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model() # 计算当前embedding
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                cl_loss = self.cal_cl_loss([user_idx, pos_idx], n)
                batch_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'cl_loss:', batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
            self.save()
        # self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        # torch.save(model.state_dict(), 'C:\\Users\\Mayn\\PycharmProjects\\test13\\SimGCL\\save\\infonce_trained_model.pth')
        self.print_model_parameters(model, "after_cl parameters")

        indices = np.random.choice(self.user_emb.shape[0], size=500, replace=False)
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

        plt.title('t-SNE Visualization_cl')
        plt.show()

    def train_bpr(self,model,optimizer):
        # model.load_state_dict(torch.load('C:\\Users\\Mayn\\PycharmProjects\\test13\\SimGCL\\save\\infonce_trained_model.pth'))
        self.print_model_parameters(model, "before_bpr parameters")
        for epoch in range(10):
            self.print_model_parameters(model, "each_epoch_bpr")
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss, align_loss, uni_loss = self.calculate_loss(user_emb, pos_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 11, 'batch', n, 'bpr_loss:', batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch + 5)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        self.print_model_parameters(model, "final parameters")

        indices = np.random.choice(self.user_emb.shape[0], size=500, replace=False)
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

        plt.title('t-SNE Visualization_bpr')
        plt.show()

    def print_model_parameters(self, model, label="Model parameters"):
        print(label)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

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

    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class GDCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(GDCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(6, 64))



        #self.fc1 = torch.nn.Linear(64, 64)
        #self.fc2 = torch.nn.Linear(64, 64)

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

