import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    # 对称归一化邻接矩阵D^-1/2*A*D^-1/2
    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            # D^-1/2
            d_inv = np.power(rowsum, -0.5).flatten()
            # 处理度为0时产生的无穷大值
            d_inv[np.isinf(d_inv)] = 0.
            # 构造对角度矩阵
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass