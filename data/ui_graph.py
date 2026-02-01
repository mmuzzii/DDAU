import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle
import torch

class Interaction(Data,Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

        # self.ii_adj = self.normalize_graph_mat(self.__create_item_cooccurrence_matrix())
        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])


    def __generate_set(self):
        for entry in self.training_data:
            #user, item= entry
            user, item, rating = entry
            # 为每个用户分配一个内部ID，并在id2user保留原ID到内部ID的映射
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            # u-i矩阵
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
            '''self.training_set_u[user][item] = 1
            self.training_set_i[item][user] = 1'''
        for entry in self.test_data:
            #user, item = entry
            user, item, rating = entry
            # 如果用户或物品在训练数据中未出现过，则跳过(?)
            if user not in self.user or item not in self.item:
                continue

            self.test_set[user][item] = rating
            #self.test_set[user][item] = 1
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        # 从training_data中生成行索引,pair(u,i,r)
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        # 全1的评分向量
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        # (0<u>,R)(R^T,0<i>)
        adj_mat = tmp_adj + tmp_adj.T
        # 添加自连接
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m

    def get_user_item_pop_unpop(self):
        # 对邻接矩阵的每一列求和，得到一个数组，每个元素对应一个物品的出现次数
        item_popularity = self.interaction_mat.sum(axis=0)

        item_pop_cnt = np.array(item_popularity).squeeze()
        item_pop_id = np.arange(self.interaction_mat.shape[1])
        i_pop = torch.ones(self.item_num, dtype=torch.long)
        i_pop[item_pop_id] = torch.from_numpy(item_pop_cnt).long()

        # 排序，基于80%确定非流行的阈值
        item_pop_cnt = torch.from_numpy(item_pop_cnt)
        sorted_idx = torch.argsort(item_pop_cnt)
        unpop_threshold = int(len(sorted_idx) * 0.8) - 1
        unpop_threshold_value = item_pop_cnt[sorted_idx[unpop_threshold]]

        # 创建分组值，将物品分为“非流行”（0）和“流行”（1）
        min_value = unpop_threshold_value
        group_value = torch.zeros(2).long()
        for i in range(2):
            group_value[i] = min_value * (i + 1)
        group_value[-1] = max(i_pop)
        for i in range(2):
            if i == 0:
                i_pop = torch.where((-1 < i_pop) & (i_pop <= group_value[i]), i, i_pop)
            else:
                i_pop = torch.where((group_value[i - 1] < i_pop) & (i_pop <= group_value[i]), i, i_pop)


        user_popularity = self.interaction_mat.sum(axis=1)

        user_pop_cnt = np.array(user_popularity).squeeze()
        user_pop_id = np.arange(self.interaction_mat.shape[0])
        u_pop = torch.ones(self.user_num, dtype=torch.long)
        u_pop[user_pop_id] = torch.from_numpy(user_pop_cnt).long()

        user_pop_cnt = torch.from_numpy(user_pop_cnt)
        sorted_idx = torch.argsort(user_pop_cnt)
        unpop_threshold = int(len(sorted_idx) * 0.8) - 1
        unpop_threshold_value = user_pop_cnt[sorted_idx[unpop_threshold]]

        min_value = unpop_threshold_value
        group_value = torch.zeros(2).long()
        for i in range(2):
            group_value[i] = min_value * (i + 1)
        group_value[-1] = max(u_pop)
        for i in range(2):
            if i == 0:
                u_pop = torch.where((-1 < u_pop) & (u_pop <= group_value[i]), i, u_pop)
            else:
                u_pop = torch.where((group_value[i - 1] < u_pop) & (u_pop <= group_value[i]), i, u_pop)

        return u_pop, i_pop

    def sample_subgraph_user(self, user_id, num_samples=30):
        """
        为给定的 user_id 采样子图
        - user_id: 用户的内部ID (0 ~ user_num-1)
        - num_samples: 子图中采样的物品数量
        返回：子图的稀疏邻接矩阵
        """
        # 获取全局邻接矩阵
        ui_adj = self.ui_adj  # 二部图 (user_num + item_num) x (user_num + item_num)

        # 找到该用户的邻居（即与用户有交互的物品）
        user_neighbors = ui_adj[user_id].indices  # 获取非零索引
        item_neighbors = user_neighbors[user_neighbors >= self.user_num] - self.user_num  # 物品邻居ID

        # 如果邻居数多于采样数，随机采样 num_samples 个邻居
        if len(item_neighbors) > num_samples:
            sampled_items = np.random.choice(item_neighbors, num_samples, replace=False)
        else:
            sampled_items = item_neighbors

        # 构造子图的行、列索引
        '''subgraph_rows = [user_id] * len(sampled_items) + [item + self.user_num for item in sampled_items]
        subgraph_cols = [item + self.user_num for item in sampled_items] + [user_id] * len(sampled_items)

        # 子图权重（全1）
        subgraph_weights = [1] * len(subgraph_rows)

        # 构造子图的稀疏邻接矩阵
        subgraph_size = self.user_num + self.item_num
        subgraph_adj = sp.csr_matrix((subgraph_weights, (subgraph_rows, subgraph_cols)),
                                     shape=(subgraph_size, subgraph_size))

        subgraph_adj = self.normalize_graph_mat(subgraph_adj)'''

        return sampled_items

    def sample_subgraph_item(self, item_id, num_samples=30):
        """
        为给定的 item_id 采样子图
        - item_id: 物品的内部ID (0 ~ item_num-1)
        - num_samples: 子图中采样的用户数量
        返回：子图的稀疏邻接矩阵
        """
        # 获取全局邻接矩阵
        ui_adj = self.ui_adj  # 二部图 (user_num + item_num) x (user_num + item_num)

        # 将物品ID映射到全局邻接矩阵中的索引
        global_item_id = item_id + self.user_num  # 全局物品ID = 物品ID + 用户数量

        # 找到该物品的邻居（即与物品有交互的用户）
        item_neighbors = ui_adj[global_item_id].indices  # 获取非零索引
        user_neighbors = item_neighbors[item_neighbors < self.user_num]  # 用户邻居ID

        # 如果邻居数多于采样数，随机采样 num_samples 个邻居
        if len(user_neighbors) > num_samples:
            sampled_users = np.random.choice(user_neighbors, num_samples, replace=False)
        else:
            sampled_users = user_neighbors

        # 构造子图的行、列索引
        '''subgraph_rows = [global_item_id] * len(sampled_users) + list(sampled_users)
        subgraph_cols = list(sampled_users) + [global_item_id] * len(sampled_users)

        # 子图权重（全1）
        subgraph_weights = [1] * len(subgraph_rows)

        # 构造子图的稀疏邻接矩阵
        subgraph_size = self.user_num + self.item_num
        subgraph_adj = sp.csr_matrix((subgraph_weights, (subgraph_rows, subgraph_cols)),
                                     shape=(subgraph_size, subgraph_size))

        # 对子图进行归一化
        subgraph_adj = self.normalize_graph_mat(subgraph_adj)'''

        return sampled_users

    def __create_item_cooccurrence_matrix(self):
        """
        构建 item-item 共现矩阵
        """
        co_matrix = np.zeros((self.item_num, self.item_num), dtype=np.float32)

        # 遍历每个用户的交互记录
        for user, items in self.training_set_u.items():
            item_list = list(items.keys())
            item_list = [int(x) for x in item_list]
            for i in range(len(item_list)):
                for j in range(i + 1, len(item_list)):
                    co_matrix[item_list[i], item_list[j]] += 1
                    co_matrix[item_list[j], item_list[i]] += 1

        # 转换为稀疏矩阵
        co_matrix_sparse = sp.csr_matrix(co_matrix)
        return co_matrix_sparse




