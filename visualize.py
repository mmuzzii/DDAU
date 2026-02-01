import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

'''user_emb = torch.load('C:\\Users\\23386\\PycharmProjects\\rec\\emb\\user_embeddings_swavcl_book2.pt')
item_emb = torch.load('C:\\Users\\23386\\PycharmProjects\\rec\\emb\\item_embeddings_swavcl_book2.pt')

num_users = 5000
num_items = 5000

indices = np.random.choice(user_emb.shape[0], size=num_users, replace=False)
user_point = user_emb[indices]
indices = np.random.choice(item_emb.shape[0], size=num_items, replace=False)
item_point = item_emb[indices]
all_emb = torch.cat((item_point, user_point), dim=0)

tsne = TSNE(n_components=2, init='pca', random_state=42)
embeddings_cpu = all_emb.cpu()
vis_dims = tsne.fit_transform(embeddings_cpu.detach().numpy())


# 创建一个默认颜色的数组
item_colors = [(80/255, 151/255, 198/255)] * num_items
user_colors = [(234/255, 161/255, 154/255)] * num_users

plt.figure(figsize=(12, 8), dpi=400)  # 图像大小为12x8英寸，300dpi分辨率

# 绘制项目的点（先绘制项目点）
plt.scatter(vis_dims[:num_items, 0], vis_dims[:num_items, 1], c=item_colors, alpha=0.5, label='Item Points')

# 绘制用户的点（后绘制用户点）
plt.scatter(vis_dims[num_items:, 0], vis_dims[num_items:, 1], c=user_colors, alpha=0.5, label='User Points')

user_patch = mpatches.Patch(color=(234/255, 161/255, 154/255), label='user')
item_patch = mpatches.Patch(color=(80/255, 151/255, 198/255), label='item')
# item_patch = mpatches.Patch(color=(148/255, 212/255, 154/255), label='item')
plt.legend(handles=[user_patch, item_patch], fontsize=20)

plt.title('Cluster CL Visualization in Amazon-book',fontsize=20)
plt.show()'''

# Data
'''x_values = [1000, 2000, 4000, 6000, 8000]
recall = [0.0735, 0.0743, 0.0741, 0.0745, 0.0745]
ndcg = [0.0610, 0.0617, 0.0614, 0.0618, 0.0620]

# RGB colors normalized to [0, 1]
recall_color = (148/255, 211/255, 195/255)
ndcg_color = (114/255, 170/255, 204/255)

# Create the plot
fig, ax = plt.subplots(figsize=(3.5, 4))

# Plot Recall
ax.plot(x_values, recall, marker='o', color=recall_color, label='Recall', linestyle='-')

# Plot NDCG
ax.plot(x_values, ndcg, marker='s', color=ndcg_color, label='NDCG', linestyle='--')

# Set x-axis and y-axis labels
ax.set_xticks(x_values)
ax.set_xticklabels([str(x) for x in x_values])
ax.set_yticks(np.arange(0.060, 0.076, 0.002))
ax.set_yticklabels([f'{y*100:.1f}%' for y in np.arange(0.060, 0.076, 0.002)])


# Set title
ax.set_title('Yelp2018')

# Display legend
ax.legend(loc='lower left', fontsize=8)

# Adjust the y-axis to show percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

# Show grid lines
ax.grid(True)

# Display the plot
plt.tight_layout()
plt.show()'''

'''loss_cl_all= [23.49753053114426, 20.137443153543146, 19.289881984716917, 18.815263116534734, 18.500076941137017, 18.276906876929626, 18.094031662186442, 17.94953186321414, 17.829194257154356, 17.72610881239127, 17.63907083757161, 17.5697193021292, 17.508187482252012, 17.441156536681035, 17.391930967520732, 17.347271566095397, 17.30586920671323, 17.263122247443892, 17.233877580278655, 17.199344765692707, 17.17047862788787, 17.149418487050987, 17.11735709845339, 17.107053296212076, 17.085397589653972, 17.064616540905703, 17.046469624629808, 17.02207480052556, 17.016797194862054, 16.999981583039872]
loss_rec_all= [1.234982909813985, 1.1014824155687701, 1.081641135266321, 1.0732140438980613, 1.068710487487849, 1.0654599804668208, 1.0631571094130225, 1.0613404577552397, 1.059492980772675, 1.0584646568407243, 1.057311157223451, 1.0561057940781797, 1.0552943540436215, 1.053812742427745, 1.0533485399957194, 1.052096270425868, 1.0511396591465003, 1.0509571240543347, 1.0498336622609983, 1.0491539039580702, 1.0487281645102757, 1.0479587131767834, 1.047803170723783, 1.0469871480352144, 1.0468147851125646, 1.0461446136284809, 1.0461015251098993, 1.0455900447209157, 1.0454048201271604, 1.044863069135252]
NDCG= [0.0539, 0.05611, 0.05699, 0.0576, 0.05783, 0.05805, 0.05824, 0.05831, 0.05839, 0.05851, 0.05858, 0.05855, 0.05854, 0.05862, 0.05863, 0.05864, 0.05866, 0.05872, 0.05872, 0.05871, 0.05869, 0.05877, 0.05872, 0.05872, 0.05871, 0.05882, 0.05877, 0.0588, 0.05875, 0.05879]
recall = [0.04082, 0.0464, 0.04967, 0.05164, 0.05283, 0.05358, 0.05403, 0.05424, 0.05439, 0.05441, 0.05432, 0.05413, 0.05396, 0.05363, 0.05334, 0.053, 0.05268, 0.05232, 0.05188, 0.05155, 0.05112, 0.05077, 0.05045, 0.05011, 0.04978, 0.04941, 0.04906, 0.04875, 0.04837, 0.04812]

epochs = range(1, len(loss_cl_all) + 1)
# 创建图表
fig, ax1 = plt.subplots(figsize=(20, 18))

# 主 y 轴（左侧）
ax1.plot(epochs, loss_cl_all, label="CL Loss", marker='o', markersize=20, linestyle='-', color='blue')
# ax1.plot(epochs, loss_rec_all, label="Rec Loss", marker='o', markersize=20, linestyle='-', color='red')
ax1.set_xlabel("Epoch", fontsize=60)
ax1.set_ylabel("Loss", fontsize=60)
ax1.tick_params(axis='y', labelcolor='black', labelsize=50)
ax1.tick_params(axis='x', labelsize=50)
ax1.legend(loc="upper right", fontsize=50)

# 副 y 轴（右侧）
ax2 = ax1.twinx()  # 创建一个共享 x 轴的 y 轴
ax2.plot(epochs, NDCG, label="Recall", marker='o', markersize=20, linestyle='-', color='orange')
ax2.plot(epochs, recall, label="Recall SimGCL", marker='o', markersize=20, linestyle='-', color='red')
ax2.set_ylabel("recall", fontsize=60, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=50)
ax2.legend(loc="lower right", fontsize=50)

#plt.ylim(0.05, 0.06)

# 设置标题和网格
plt.title("Loss and Recall over Epochs of DDAU in iFashion", fontsize=50)
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

# Data for the three datasets: Douban-Book, Yelp2018, Amazon-Book
lambda_values = [0.3, 0.4, 0.5,0.8, 1, 2, 3]
yelp_recall = [0.0575, 0.0645, 0.0682, 0.0721, 0.0735, 0.0744,0.0737]
yelp_ndcg = [0.0477, 0.0535, 0.0568, 0.06, 0.061, 0.0616,0.0612]

douban_recall = [0.1772, 0.1829, 0.176, 0.1693, 0.1655, 0.1484,0.1316]
douban_ndcg = [0.1539, 0.1631, 0.1535, 0.1463, 0.141, 0.1156,0.0952]

iFashion_recall = [0.1193, 0.1213, 0.1216, 0.1212, 0.1203, 0.1145,0.1079]
iFashion_ndcg = [0.0573, 0.0584, 0.0589, 0.0587, 0.0586, 0.056,0.0532]

# Plotting the data
plt.figure(figsize=(24, 7))

# Create a new X tick positions to ensure equal spacing
equal_spacing_ticks = np.linspace(0, len(lambda_values) - 1, len(lambda_values))

# Yelp2018 plot
plt.subplot(1, 3, 1)
plt.plot(equal_spacing_ticks, yelp_recall, marker='o', label='Recall', color='orange')
plt.plot(equal_spacing_ticks, yelp_ndcg, marker='s', label='NDCG', color='darkblue')
plt.title("Yelp2018", fontsize=30)
plt.xlabel("λ", fontsize=25)
plt.ylabel("Metric Value", fontsize=25)
plt.legend()
plt.grid(True)
plt.xticks(equal_spacing_ticks, lambda_values, fontsize=20)  # Manually set the x-ticks to lambda_values
plt.tick_params(axis='both', labelsize=25)

# Douban-Book plot
plt.subplot(1, 3, 2)
plt.plot(equal_spacing_ticks, douban_recall, marker='o', label='Recall', color='orange')
plt.plot(equal_spacing_ticks, douban_ndcg, marker='s', label='NDCG', color='darkblue')
plt.title("Douban-Book", fontsize=30)
plt.xlabel("λ", fontsize=25)
plt.ylabel("Metric Value", fontsize=25)
plt.legend()
plt.grid(True)
plt.xticks(equal_spacing_ticks, lambda_values, fontsize=20)  # Manually set the x-ticks to lambda_values
plt.tick_params(axis='both', labelsize=25)

# iFashion plot
plt.subplot(1, 3, 3)
plt.plot(equal_spacing_ticks, iFashion_recall, marker='o', label='Recall', color='orange')
plt.plot(equal_spacing_ticks, iFashion_ndcg, marker='s', label='NDCG', color='darkblue')
plt.title("iFashion", fontsize=30)
plt.xlabel("λ", fontsize=25)
plt.ylabel("Metric Value", fontsize=25)
plt.legend()
plt.grid(True)
plt.xticks(equal_spacing_ticks, lambda_values, fontsize=20)  # Manually set the x-ticks to lambda_values
plt.tick_params(axis='both', labelsize=25)

# Adjust layout to add more space between subplots
plt.subplots_adjust(wspace=0.4)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data for each dataset
'''K_values = [1000,2000, 3000, 4000]

# Data for each dataset (for u=2, u=0.4, u=0.5)
data_1 = [0.0737,0.0743, 0.0744, 0.0742]  # u=2
data_2 = [0.1765,0.1786,0.1829,0.1801]  # u=0.4
data_3 = [0.1167,0.1203,0.1217,0.1216]  # u=0.5

# 设置图例和子图标题
titles = ["yelp2018", "douban-book", "iFashion"]
datasets = [data_1, data_2, data_3]
# colors = ['#B0C277', '#889F64', '#39593F', '#617C52']  # 每组颜色
colors = ['#B7E4FF', '#BBD0FF', '#7CAEF0', '#D1AEEC']

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)

y_limits = [(0.073, 0.075), (0.175, 0.185), (0.115, 0.123)]
# 遍历每个子图
for i, ax in enumerate(axes):
    data = datasets[i]
    ax.bar(range(len(K_values)), data, color=colors, alpha=0.8, width=0.6)
    ax.set_title(titles[i], fontsize=30)
    ax.set_xlabel("K", fontsize=30)
    ax.set_ylabel("NDCG@20", fontsize=30)
    ax.set_xticks(range(len(K_values)))
    ax.set_xticklabels(K_values, fontsize=25)

    ax.set_ylim(y_limits[i])
    ax.tick_params(axis='y', labelsize=25)

    # 在每个柱子上添加数值标签
    for j, v in enumerate(data):
        ax.text(j, v + 0.0001, f"{v:.4f}", ha='center', fontsize=20)
plt.subplots_adjust(wspace=15)
# 调整布局
plt.tight_layout()
plt.show()'''

