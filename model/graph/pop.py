import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import math
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22



class pop(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(pop, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['pop'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = pop_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

        self.training_set_u = self.data.training_set_u
        self.item = self.data.item

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        l_align_list = []
        l_uniform_list = []
        NDCG_list = []

        # 划分训练集和测试集的pop与unpop
        '''pop_indices_train, unpop_indices_train = self.pop_and_unpop(self.data.training_set_u)
        # pop_indices_test, unpop_indices_test = self.pop_and_unpop(self.data.test_set)
        user_pop_items, user_unpop_items, user_indices = self.random_select(pop_indices_train, unpop_indices_train)
        selected_user = random.choice(user_indices)

        selected_user_pop_items = [item for item in self.training_set_u[selected_user] if item in user_pop_items]
        selected_user_unpop_items = [item for item in self.training_set_u[selected_user] if item in user_unpop_items]
        print(selected_user_pop_items)
        print(len(selected_user_pop_items),len(selected_user_unpop_items))'''

        for epoch in range(self.maxEpoch):

            l_align = 0
            l_uniform = 0
            NDCG = 0

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                align_loss, uni_loss, directAU_loss = self.calculate_loss(user_emb, pos_item_emb)

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss:', cl_loss.item())

                l_align += align_loss.item()
                l_uniform += uni_loss.item()

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            l_align = l_align / math.ceil(len(self.data.training_data) / self.batch_size)
            l_uniform = l_uniform / math.ceil(len(self.data.training_data) / self.batch_size)

            l_align_list.append(l_align)
            l_uniform_list.append(l_uniform)

            # if epoch==0:
                # self.draw(user_pop_items, user_unpop_items, user_indices, selected_user,selected_user_pop_items, selected_user_unpop_items)
                # self.draw_3(user_pop_items, user_unpop_items, user_indices)

            # 评估当前模型
            # self.train_evaluation(pop_indices_train, unpop_indices_train)
            # self.test_evaluation(pop_indices_train, unpop_indices_train)

            '''G1, G2 = self.split_group_items(self.data, self.popular)
            align_loss = self.alignment_user(self.item_emb[G1], self.item_emb[G2]) * 1000
            optimizer.zero_grad()
            align_loss.backward()
            optimizer.step()'''

            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            NDCG = float(v)
            NDCG_list.append(NDCG)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        print(l_align_list)
        print(l_uniform_list)
        print(NDCG_list)
        self.draw_trace(l_align_list, l_uniform_list, NDCG_list)

        '''NDCG_train_pop, NDCG_train_unpop = self.train_evaluation(pop_indices_train, unpop_indices_train)
        NDCG_test_pop, NDCG_test_unpop = self.test_evaluation(pop_indices_train, unpop_indices_train)

        # NDCG柱状图
        categories = ['Train Unpop', 'Train Pop', 'Test Unpop', 'Test Pop', 'Test All']
        values = [NDCG_train_unpop, NDCG_train_pop, NDCG_test_unpop, NDCG_test_pop, NDCG_all]
        colors = ['blue', 'green', 'lightblue', 'lightgreen', 'orange']
        # 创建柱状图
        fig, ax = plt.subplots()
        index = np.arange(len(categories))
        bar_width = 0.35
        opacity = 0.8
        for i in range(len(categories)):
            plt.bar(index[i], values[i], bar_width, alpha=opacity, color=colors[i], label=categories[i])
        plt.xlabel('Categories')
        plt.ylabel('NDCG@100')
        plt.title('NDCG@100 by Dataset Subset')
        plt.xticks(index, categories)
        plt.ylim(0, max(values) * 1.2)  # 设置y轴的上限稍高于最大值
        # 添加一个合适的布局和显示图表
        plt.tight_layout()
        plt.show()'''

        # self.draw(user_pop_items, user_unpop_items, user_indices, selected_user,selected_user_pop_items, selected_user_unpop_items)
        # self.draw_3(user_pop_items, user_unpop_items, user_indices)

    def draw_trace(self, l_align, l_uniform, NCDG):
        # 创建图表
        plt.figure(figsize=(10, 6))

        # 绘制轨迹图
        plt.scatter(l_uniform, l_align, marker='o', color='b', label='SimGCL')
        for i in range(len(l_align)):
            plt.annotate(NCDG[i], (l_uniform[i], l_align[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        # 标注起点和终点
        plt.scatter(l_uniform[0], l_align[0], color='blue', edgecolors='k', s=100, zorder=5, label='Start')
        plt.scatter(l_uniform[-1], l_align[-1], color='red', edgecolors='k', s=100, zorder=5, label='End')

        # 添加标注文本
        plt.text(l_uniform[0], l_align[0], 'Start', fontsize=12, verticalalignment='bottom',
                 horizontalalignment='right')
        plt.text(l_uniform[-1], l_align[-1], 'End', fontsize=12, verticalalignment='bottom',
                 horizontalalignment='right')

        # 设置图表标题和标签
        plt.title('Trajectory of $L_{align}$ and $L_{uniform}$ over Epochs')
        plt.xlabel('$L_{uniform}$')
        plt.ylabel('$L_{align}$')
        plt.legend()
        plt.grid(True)

        # 显示图表
        plt.show()

    def draw(self,user_pop_items, user_unpop_items, user_indices, selected_user, selected_user_pop_items, selected_user_unpop_items):
        # 可视化pop和unpop embedding

        user_pop_items = [int(item) for item in user_pop_items]
        user_unpop_items = [int(item) for item in user_unpop_items]
        user_indices = [int(item) for item in user_indices]

        selected_user = int(selected_user)
        selected_user_pop_items = [int(item) for item in selected_user_pop_items]
        selected_user_unpop_items = [int(item) for item in selected_user_unpop_items]

        selected_user_embeddings = self.user_emb[user_indices]
        selected_pop_embeddings = self.item_emb[user_pop_items]
        selected_unpop_embeddings = self.item_emb[user_unpop_items]


        selected_user_pop_indices = [user_pop_items.index(item) for item in selected_user_pop_items if item in user_pop_items]
        selected_user_unpop_indices = [user_unpop_items.index(item) for item in selected_user_unpop_items if item in user_unpop_items]
        selected_user_indices = user_indices.index(selected_user)
        print(selected_user_pop_indices)

        all_emb = torch.cat((selected_pop_embeddings, selected_unpop_embeddings, selected_user_embeddings), dim=0)
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        embeddings_cpu = all_emb.cpu()
        vis_dims = tsne.fit_transform(embeddings_cpu.detach().numpy())

        n_popular = len(user_pop_items)
        n_unpop = len(user_unpop_items)
        reduced_popular = vis_dims[:n_popular]
        reduced_unpopular = vis_dims[n_popular:n_unpop + n_popular]
        reduced_user = vis_dims[n_popular + n_unpop:]
        print(len(reduced_unpopular))

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_popular[:, 0], reduced_popular[:, 1], color='red', label='Popular', alpha=0.5)
        plt.scatter(reduced_unpopular[:, 0], reduced_unpopular[:, 1], color='blue', label='Unpopular', alpha=0.5)
        plt.scatter(reduced_user[:, 0], reduced_user[:, 1], color='green', label='user', alpha=0.5)

        for item in selected_user_pop_indices:
            item = int(item)

            plt.scatter(reduced_popular[item, 0], reduced_popular[item, 1], marker='x', color='darkred', edgecolor='black',label='User Pop Items' if item == selected_user_pop_indices[0] else "")
        for item in selected_user_unpop_indices:
            item = int(item)
            plt.scatter(reduced_unpopular[item, 0], reduced_unpopular[item, 1], marker='x', color='darkblue', edgecolor='black',label='User Unpop Items' if item == selected_user_unpop_indices[0] else "")

        plt.scatter(reduced_user[selected_user_indices, 0], reduced_user[selected_user_indices, 1], color='yellow', label='selected user')

        plt.title('t-SNE Visualization of Embeddings(500 users)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    def draw_3(self,user_pop_items, user_unpop_items, user_indices):
        user_pop_items = [int(item) for item in user_pop_items]
        user_unpop_items = [int(item) for item in user_unpop_items]
        user_indices = [int(item) for item in user_indices]

        selected_user_embeddings = self.user_emb[user_indices]
        selected_pop_embeddings = self.item_emb[user_pop_items]
        selected_unpop_embeddings = self.item_emb[user_unpop_items]

        all_emb = torch.cat((selected_pop_embeddings, selected_unpop_embeddings, selected_user_embeddings), dim=0)
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        embeddings_cpu = all_emb.cpu()
        vis_dims = tsne.fit_transform(embeddings_cpu.detach().numpy())

        n_popular = len(user_pop_items)
        n_unpop = len(user_unpop_items)
        reduced_popular = vis_dims[:n_popular]
        reduced_unpopular = vis_dims[n_popular:n_unpop + n_popular]
        reduced_user = vis_dims[n_popular + n_unpop:]

        # user
        plt.figure(figsize=(10, 10))  # 设置图形大小
        plt.scatter(reduced_user[:, 0], reduced_user[:, 1], color='green', label='user', alpha=0.5)
        plt.title('KDE Density Plot of user Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

        # pop
        plt.figure(figsize=(10, 10))  # 设置图形大小
        plt.scatter(reduced_popular[:, 0], reduced_popular[:, 1], color='red', label='Popular', alpha=0.5)
        plt.title('KDE Density Plot of pop item Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

        # unpop
        plt.figure(figsize=(10, 10))  # 设置图形大小
        plt.scatter(reduced_unpopular[:, 0], reduced_unpopular[:, 1], color='blue', label='Unpopular', alpha=0.5)
        plt.title('KDE Density Plot of unpop item Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()

    def random_select(self, pop_indices, unpop_indices):
        # 从用户中随机选择500个
        users = list(self.training_set_u.keys())  # 获取所有用户的列表
        selected_users = random.sample(users, 500)  # 随机选取100个用户

        # 将 pop_indices_train 和 unpop_indices_train 转换为集合以便快速查找
        pop_indices_set = set(pop_indices)
        unpop_indices_set = set(unpop_indices)

        # 创建列表来存储流行和不流行项目的索引
        popular_items = []
        unpopular_items = []

        # 遍历每个选中的用户
        for user in selected_users:
            # 获取该用户的项目评分字典
            user_items = self.training_set_u[user]
            # 遍历该用户的每个项目
            for item in user_items.keys():
                if item in pop_indices_set:
                    popular_items.append(item)
                elif item in unpop_indices_set:
                    unpopular_items.append(item)

        # 将列表转换为集合去除重复项
        popular_items_set = set(popular_items)
        unpopular_items_set = set(unpopular_items)
        popular_items = list(popular_items_set)
        unpopular_items = list(unpopular_items_set)

        print(popular_items)
        print(len(popular_items),len(unpopular_items))

        return popular_items, unpopular_items, selected_users
    def pop_and_unpop(self,set):
        # 初始化一个字典来存储每个项目的总评分
        item_popularity = defaultdict(int)

        # 遍历数据结构，累计每个项目的评分
        for user, items in set.items():
            for item, rating in items.items():
                item_popularity[item] += rating

        # 转换为列表并按流行度排序
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)

        # 获取项目总数
        total_items = len(sorted_items)

        # 计算流行和不流行的切分点
        cutoff_index = int(0.2 * total_items)

        # 划分流行和不流行的项目
        popular_items = [item for item, popularity in sorted_items[:cutoff_index]]
        unpopular_items = [item for item, popularity in sorted_items[cutoff_index:]]
        print(len(popular_items))

        return popular_items, unpopular_items

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = 2 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align, uniform, align + uniform

    def cal_cl_loss(self, idx, batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)


        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        '''user_cl_loss = InfoNCE(u1, u2,0.2)
        item_cl_loss = InfoNCE(i1, i2, 0.2)'''
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


class pop_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(pop_Encoder, self).__init__()
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



