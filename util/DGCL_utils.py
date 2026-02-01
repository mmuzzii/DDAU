import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp

def compute_ppr(at, alpha=0.2):
    # at为对称归一化后的邻接矩阵(D^-1/2)*A*(D^-1/2)；此函数使用ppr得到差异矩阵
    return alpha * inv((np.eye(at.shape[0]) - (1 - alpha) * at))