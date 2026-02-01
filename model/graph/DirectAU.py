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
from model.graph.LightGCN import LGCN_Encoder
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize


class DirectAU(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DirectAU, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DirectAU'])
        self.gamma = float(args['-gamma'])
        self.n_layers = int(args['-n_layers'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                align_loss, uni_loss, directAU_loss = self.calculate_loss(user_emb, pos_item_emb)
                batch_loss = directAU_loss + l2_reg_loss(self.reg, user_emb,pos_item_emb)/self.batch_size
                # batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'align_loss:', align_loss.item(), 'uniform_loss:', uni_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        '''torch.save(self.user_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_directau.pt')
        torch.save(self.item_emb, 'C:\\Users\\23386\\PycharmProjects\\rec\\emb\\item_embeddings_directau.pt')'''

        '''tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        emb = self.item_emb.cpu()
        item_emb_2d = tsne.fit_transform(emb)
        item_emb_2d = normalize(item_emb_2d, axis=1, norm='l2')

        f, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        sns.kdeplot(x=[p[0] for p in item_emb_2d], y=[p[1] for p in item_emb_2d],bw_adjust=0.5, fill=True, cmap="GnBu", ax=axs[0])
        axs[0].set_title(f'2D KDE of DirectAU', fontsize=12, fontweight="bold")
        # 提取数据点的坐标，用于计算角度
        x = [p[0] for p in item_emb_2d]
        y = [p[1] for p in item_emb_2d]

        # 计算角度并绘制一维KDE
        angles = np.arctan2(y, x)
        sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1], color='green')
        axs[1].set_title(f'Angle KDE of DirectAU', fontsize=12)

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
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = self.gamma * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align,uniform,align + uniform

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()