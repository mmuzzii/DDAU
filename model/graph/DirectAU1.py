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
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class DirectAU1(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DirectAU1, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DirectAU1'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = DirectAU1_Encoder(self.data, self.emb_size, self.eps, self.n_layers)


    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                '''cl_loss = 0.5 * self.cal_cl_loss([user_idx,pos_idx])
                # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                align_loss, uni_loss, directAU_loss = self.calculate_loss(user_emb, pos_item_emb)
                batch_loss = cl_loss + directAU_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size'''

                u_cl_align, i_cl_align, bpr_align, uniform = self.align_uni_loss([user_idx, pos_idx], user_emb, pos_item_emb)
                cl_loss = 0.1 * self.caw_cl_loss([user_idx, pos_idx])
                align, uniform, directAU_loss = self.calculate_loss(user_emb, pos_item_emb)
                batch_loss = cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size + directAU_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'au_loss:', directAU_loss.item(), 'cl_loss', cl_loss.item(), 'batch_loss', batch_loss.item())
                    # print('training:', epoch + 1, 'batch', n, 'u_cl_align:', u_cl_align.item(),  'i_cl_align:', i_cl_align.item(), 'bpr_align:', bpr_align.item(), 'uniform:', uniform.item(), 'batch loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            measure = self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        '''torch.save(self.user_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_directau1clr.pt')
        torch.save(self.item_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\item_embeddings_directau1clr.pt')'''
        '''tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        emb = self.item_emb.cpu()
        item_emb_2d = tsne.fit_transform(emb)
        item_emb_2d = normalize(item_emb_2d, axis=1, norm='l2')

        f, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        sns.kdeplot(x=[p[0] for p in item_emb_2d], y=[p[1] for p in item_emb_2d], bw_adjust=0.5, fill=True, cmap="coolwarm",
                    ax=axs[0])
        axs[0].set_title(f'2D KDE of AUCC', fontsize=12, fontweight="bold")
        # 提取数据点的坐标，用于计算角度
        x = [p[0] for p in item_emb_2d]
        y = [p[1] for p in item_emb_2d]

        # 计算角度并绘制一维KDE
        angles = np.arctan2(y, x)
        sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1], color='purple')
        axs[1].set_title(f'Angle KDE of AUCC', fontsize=12)

        # 显示图形
        plt.tight_layout()
        plt.show()'''

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

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        # return (x - y).norm(p=2, dim=1).pow(2).mean()
        pos_score = torch.diag((x @ y.T))
        return -pos_score.mean()


    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = 1 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align, uniform, align + uniform

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def align_uni_loss(self, idx, user_emb, item_emb):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        # cl角度对齐
        '''u_cl_align = self.alignment(user_view_1[u_idx], user_view_2[u_idx])
        i_cl_align = self.alignment(item_view_1[i_idx], item_view_2[i_idx])'''
        u_cl_align = self.alignment(user_view_1[u_idx], user_view_2[u_idx])
        i_cl_align = self.alignment(item_view_1[i_idx], item_view_2[i_idx])

        # bpr角度对齐
        bpr_align = self.alignment(user_emb, item_emb)

        '''u1 = user_view_1[u_idx]
        u1 = u1[:, :32]
        u2 = user_view_2[u_idx]
        u2 = u2[:, :32]
        i1 = item_view_1[i_idx]
        i1 = i1[:, :32]
        i2 = item_view_2[i_idx]
        i2 = i2[:, :32]
        u_cl_align = self.alignment(u1, u2)
        i_cl_align = self.alignment(i1, i2)'''

        '''u3 = user_emb[:, :16]
        i3 = item_emb[:, :16]
        bpr_align = self.alignment(u3, i3)'''

        # cl角度均匀
        uniform = 2 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2

        return u_cl_align, i_cl_align, bpr_align, uniform


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def feature_store(self, model):
        total = torch.tensor([]).cuda()
        for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            user_idx, pos_idx, neg_idx = batch
            rec_user_emb, rec_item_emb = model()
            user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
            feature = model.model(batch[0].cuda())
            feature = F.normalize(feature, dim=1)
            total = torch.cat((total, feature.detach()), dim=0)
        return total


class DirectAU1_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(DirectAU1_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.norm = NormLayer()

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

            # ego_embeddings = self.norm(ego_embeddings)

            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        return user_all_embeddings, item_all_embeddings


class NormLayer(nn.Module):
    def __init__(self):
        """
            mode:
              'None' : No normalization
              'PN'   : PairNorm
              'PN-SI'  : Scale-Individually version of PairNorm
              'PN-SCS' : Scale-and-Center-Simultaneously version of PairNorm
              'LN': LayerNorm
              'CN': ContraNorm
        """
        super(NormLayer, self).__init__()
        # self.mode = args.norm_mode
        # self.scale = args.norm_scale

    def forward(self, x):

        x = x - x.mean(dim=1, keepdim=True)
        x = nn.functional.normalize(x, dim=1)

        return x