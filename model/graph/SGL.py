import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
# Paper: self-supervised graph learning for recommendation. SIGIR'21

user_0_e=[]
user_0_a=[]
class SGL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGL'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        loss_cl_all = []
        loss_rec_all = []
        NDCG = []
        count=0
        for epoch in range(self.maxEpoch):
            loss_cl = 0
            loss_rec = 0
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()

                '''for id in user_idx:
                    if id == 0:
                        user_0_e = rec_user_emb[id].data.cpu().numpy()'''

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2,n)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())

                loss_cl = loss_cl + cl_loss.item()
                loss_rec = loss_rec + rec_loss.item()

            loss_cl = loss_cl / n
            loss_cl_all.append(loss_cl)
            loss_rec = loss_rec / n
            loss_rec_all.append(loss_rec)

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            #print("count:", count)
            measure = self.fast_evaluation(epoch)
            k, v = measure[3].strip().split(':')
            recall = float(v)
            NDCG.append(recall)

        #user_0_a.insert(0, user_0_e)
        '''matrix = np.array(last_embedding_u)
        tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=30, n_iter=1000)
        # 将embedding降维为2维
        vis_dims = tsne.fit_transform(matrix)
        x = vis_dims[:, 0]
        y = vis_dims[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        plt.scatter(x, y, alpha=0.5)
        #plt.scatter(x_mean, y_mean, c='red', marker='x', s=100)
        #plt.scatter(x[0], y[0], c='green', marker='x', s=100)
        plt.title('t-SNE Visualization')
        plt.show()'''
        print('loss_cl_all', loss_cl_all)
        print('loss_rec_all', loss_rec_all)
        print('NDCG', NDCG)
        epochs = range(1, len(loss_cl_all) + 1)
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 主 y 轴（左侧）
        ax1.plot(epochs, loss_cl_all, label="CL Loss", marker='o', markersize=6, linestyle='-', color='blue')
        ax1.plot(epochs, loss_rec_all, label="Rec Loss", marker='o', markersize=6, linestyle='-', color='red')
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc="upper right", fontsize=10)

        # 副 y 轴（右侧）
        ax2 = ax1.twinx()  # 创建一个共享 x 轴的 y 轴
        ax2.plot(epochs, NDCG, label="NDCG", marker='o', markersize=6, linestyle='-', color='orange')
        ax2.set_ylabel("NDCG", fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc="lower right", fontsize=10)

        # 设置标题和网格
        plt.title("Loss and NDCG over Epochs of SGL in douban", fontsize=14)
        plt.grid(alpha=0.3)

        # 显示图表
        plt.tight_layout()
        plt.show()

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def draw(self,loss,epoch,label):
        x1 = range(0, epoch+1)
        y1 = loss
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, '.-',label=label, markevery=20)
        plt.xlabel('epoch')
        #plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    # 0or1进行一次增强，random walk 每层的增强视图不同
    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        # 将其转换为拉普拉斯矩阵
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            # 有扰动(增强)的邻接矩阵则用扰动的来更新embedding
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    # 扰动邻接矩阵为列表，即不同层有不同的邻接矩阵
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2,batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 获得两个增强视图的embedding
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        # print(user_view_1.shape)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)

        '''num=view1.shape[0]
        k = torch.tensor(int(num * 0.5))
        p = (1 / torch.sqrt(k)) * torch.randn(k,num).cuda()

        view1 = p @ view1
        view2 = p @ view2

        for id in u_idx:
            if id == 0:
                user_0_a.append(user_view_1[id].data.cpu().numpy())'''

        return InfoNCE(view1,view2,self.temp)

