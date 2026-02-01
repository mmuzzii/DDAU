import torch
import matplotlib.pyplot as plt
import numpy as np
def calculate_effective_rank(embedding):
    # 1. 对嵌入矩阵执行奇异值分解（SVD）
    _, S, _ = torch.svd(embedding)

    # 2. 对奇异值进行 L1 归一化
    S_normalized = S / S.sum()

    # 3. 计算 Shannon entropy
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-8))  # 避免 log(0)

    # 4. 计算有效秩
    effective_rank = torch.exp(entropy)

    return effective_rank.item()

def calculate_singular_values(embedding_matrix):
    # 计算奇异值分解，获取奇异值
    _, S, _ = torch.linalg.svd(embedding_matrix)
    return S

def plot_singular_value_distribution(s1):
    # 绘制奇异值分布
    plt.figure(figsize=(8, 6))

    plt.hist(s1.cpu().detach().numpy(), bins=30, alpha=0.5, color='blue',label='directau1_dot')

    plt.xlabel('singular value')
    plt.ylabel('number')
    plt.legend(loc='upper right')
    plt.title('Singular Value Distribution')
    plt.show()

def plot_singular(s):
    s = s.cpu().detach().numpy()
    # 计算特征值
    cov_matrix = np.cov(s.T)  # 计算协方差矩阵
    eigenvalues, _ = np.linalg.eig(cov_matrix)  # 计算协方差矩阵的特征值

    # 对特征值进行排序（从大到小）
    eigenvalues = np.sort(eigenvalues)[::-1]

    # 取特征值的对数
    log_eigenvalues = np.log(eigenvalues)

    # 生成索引
    indices = np.arange(len(eigenvalues))

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(indices, log_eigenvalues, label='Embedding Space Spectrum', color='b')
    plt.xlabel('eigenvalues ranking index')
    plt.ylabel('log of eigenvalues')
    plt.title('Embedding Space Spectrum directau1clr')
    plt.grid(True)
    plt.show()

# 加载嵌入矩阵
loaded_embedding = torch.load('C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_swav.pt')

# 计算有效秩
'''erank = calculate_effective_rank(loaded_embedding)
print(f"Effective Rank: {erank}")'''

s = calculate_singular_values(loaded_embedding)
plot_singular_value_distribution(s)

# plot_singular(loaded_embedding)