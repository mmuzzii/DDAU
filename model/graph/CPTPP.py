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
from model.graph.SimGCL import SimGCL_Encoder
from sklearn.decomposition import NMF
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize

class CPTPP(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CPTPP, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CPTPP'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.inputs_type = int(args['-inputs_type'])
        prompt_size = int(args['-prompt_size'])

        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.prompts_generator = Prompts_Generator(self.emb_size, prompt_size).cuda()
        self.fusion_mlp = Fusion_MLP(self.emb_size, prompt_size).cuda()

        # 生成计算prompt generator的input所需要的邻接矩阵A的各种形式
        if self.inputs_type == 0:
            # historical interaction records(X=A*I_pretrain)^T
            self.interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda()
        if self.inputs_type == 2:
            # high_order
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()

    def _pre_train(self):
        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(50):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = pre_trained_model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], n)
                batch_loss = cl_loss

                # Backward and optimize
                optimizer.zero_grad()
                # 最后一个周期保留训练图
                if epoch == self.maxEpoch - 1:
                    batch_loss.backward(retain_graph=True)
                else:
                    batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

    def train(self):
        self._pre_train()

        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        if self.inputs_type == 1:
            # 邻接矩阵分解
            nmf = NMF(n_components=self.emb_size, max_iter=1000)
            self.user_profiles = torch.Tensor(nmf.fit_transform(self.data.interaction_mat.toarray())).cuda()

        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_emb, item_emb = model()
                if self.inputs_type == 0 or self.inputs_type == 2:
                    prompts = self._prompts_generation(item_emb, user_emb)
                else:
                    prompts = self.prompts_generator(self.user_profiles)

                prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, user_emb)
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = prompted_user_emb, item_emb
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                #align_loss, uni_loss, directAU_loss = self.calculate_loss(user_emb, pos_item_emb)
                #batch_loss = directAU_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) / self.batch_size

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100 == 0 and n > 0 :
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:',directAU_loss.item(), 'align_loss:', align_loss.item(), 'uniform_loss:', uni_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                if self.inputs_type == 0 or self.inputs_type == 2:
                    prompts = self._prompts_generation(self.item_emb, self.user_emb)
                else:
                    prompts = self.prompts_generator(self.user_profiles)
                prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, self.user_emb)
                self.user_emb = prompted_user_emb
            if epoch >= 5:
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        '''tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        emb = self.item_emb.cpu()
        item_emb_2d = tsne.fit_transform(emb)
        item_emb_2d = normalize(item_emb_2d, axis=1, norm='l2')

        f, axs = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        sns.kdeplot(x=[p[0] for p in item_emb_2d], y=[p[1] for p in item_emb_2d], bw_adjust=0.5, fill=True, cmap="GnBu",
                    ax=axs[0])
        axs[0].set_title(f'2D KDE of SimGCL-prompt', fontsize=12, fontweight="bold")
        # 提取数据点的坐标，用于计算角度
        x = [p[0] for p in item_emb_2d]
        y = [p[1] for p in item_emb_2d]

        # 计算角度并绘制一维KDE
        angles = np.arctan2(y, x)
        sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axs[1], color='green')
        axs[1].set_title(f'Angle KDE of SimGCL-prompt', fontsize=12)

        # 显示图形
        plt.tight_layout()
        plt.show()'''

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            if self.inputs_type == 0 or self.inputs_type == 2:
                prompts = self._prompts_generation(self.best_item_emb, self.best_user_emb)
            else:
                prompts = self.prompts_generator(self.user_profiles)
            prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, self.best_user_emb)
            self.best_user_emb = prompted_user_emb

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


    def cal_cl_loss(self, idx, batch_num):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = 1 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align,uniform,align + uniform

    def _prompts_generation(self, item_emb, user_emb):
        if self.inputs_type == 0:
            inputs = self._historical_records(item_emb)
        # elif self.inputs_type == 1:
        #     inputs = self._adjacency_matrix_factorization()
        elif self.inputs_type == 2:
            inputs = self._high_order_u_relations(item_emb, user_emb)
        prompts = self.prompts_generator(inputs)
        return prompts

    def _historical_records(self, item_emb):
        # A*I_pretrain
        inputs = torch.mm(self.interaction_mat, item_emb)
        return inputs

    def _high_order_u_relations(self, item_emb, user_emb):
        # A((0,A),(A^T,0))*(U_pretrain,I_pretrain)
        ego_embeddings = torch.cat((user_emb, item_emb), 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        inputs, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return inputs

    def _prompts_u_embeddings_fusion(self, prompts, user_emb):
        prompts_user_emb = torch.cat((prompts, user_emb), 1)
        prompted_user_emb = self.fusion_mlp(prompts_user_emb)
        return prompted_user_emb

# 输入user profile matrix(3种构建方法)，输出personalized prompts (2-layers MLP)
class Prompts_Generator(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Prompts_Generator, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(emb_size, prompt_size), nn.Linear(prompt_size, prompt_size)])
        self.activation = nn.Tanh()
        # self.activation = nn.Sigmoid()

    def forward(self, inputs):
        prompts = inputs
        for i in range(len(self.layers)):
            prompts = self.layers[i](prompts)
            prompts = self.activation(prompts)

        return prompts

# 将[P(prompt),U_pretrain]^T 维度压缩为personalized prompt enhanced user representation (MLP:n*(p+d)->n*d)
class Fusion_MLP(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Fusion_MLP, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(emb_size + prompt_size, emb_size), nn.Linear(emb_size, emb_size)])
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        return x