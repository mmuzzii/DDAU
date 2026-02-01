import matplotlib
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
import faiss
from tqdm import tqdm
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import seaborn as sns




class ICL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ICL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['ICL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = ICL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.device = torch.device("cuda")
        self.clusters = []
        self.clusters_i = []
        self.criterion = nn.CrossEntropyLoss().cuda()
        cluster = KMeans(
            num_cluster=256,
            seed=1,
            hidden_size=64,
            gpu_id=0,
            device=self.device,
        )

        self.clusters.append(cluster)

        cluster_i = KMeans(
            num_cluster=32,
            seed=1,
            hidden_size=64,
            gpu_id=0,
            device=self.device,
        )

        self.clusters_i.append(cluster_i)

        self.cf_criterion = NCELoss(1, self.device)
        self.pcl_criterion = PCLoss(1, self.device)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            kmeans_training_data = []
            kmeans_training_data_i = []
            print("Training Clusters:")
            self.model.eval()
            rec_user_emb, rec_item_emb = model()
            rec_user_emb_out = rec_user_emb.detach().cpu().numpy()
            kmeans_training_data.append(rec_user_emb_out)
            '''for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                rec_user_emb, rec_item_emb = model()

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb = rec_user_emb[user_idx]
                rec_item_emb = rec_item_emb[pos_idx]

                # ------ intentions clustering ----- #
                self.model.eval()

                # print(rec_user_emb.shape)
                rec_user_emb_out = rec_user_emb.view(rec_user_emb.shape[0], -1)
                rec_user_emb_out = rec_user_emb_out.detach().cpu().numpy()
                rec_item_emb_out = rec_item_emb.view(rec_item_emb.shape[0], -1)
                rec_item_emb_out = rec_item_emb_out.detach().cpu().numpy()
                kmeans_training_data = [rec_user_emb_out, rec_item_emb_out]
                rec_user_emb_out = rec_user_emb.detach().cpu().numpy()
                kmeans_training_data.append(rec_user_emb_out)
                print("1",rec_user_emb_out.shape)

                rec_item_emb_out = rec_item_emb.detach().cpu().numpy()
                kmeans_training_data_i.append(rec_item_emb_out)'''

                # 将列表 kmeans_training_data 中的所有 NumPy 数组沿着第一个轴（axis=0，即行）合并成一个大的数组。这个数组将作为 K-Means 聚类算法的训练数据
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
            # print("2",kmeans_training_data.shape)
            # kmeans_training_data_i = np.concatenate(kmeans_training_data_i, axis=0)

            for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                cluster.train(kmeans_training_data)
                self.clusters[i] = cluster

            '''for i, cluster in tqdm(enumerate(self.clusters_i), total=len(self.clusters_i)):
                cluster.train(kmeans_training_data_i)
                self.clusters_i[i] = cluster'''

            # clean memory
            del kmeans_training_data, kmeans_training_data_i
            import gc
            gc.collect()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                self.model.train()
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # rec_loss, align_loss, uni_loss= self.calculate_loss(user_emb, pos_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])

                # ---------- contrastive learning task -------------#
                # instance cl loss

                # rec_user_emb_out = user_emb.view(user_emb.shape[0], -1)
                # rec_item_emb_out = pos_item_emb.view(pos_item_emb.shape[0], -1)
                rec_user_emb_out = user_emb.detach().cpu().numpy()
                rec_item_emb_out = pos_item_emb.detach().cpu().numpy()
                '''out = [rec_user_emb_out,rec_item_emb_out]
                out = np.concatenate(out, axis=0)'''


                # query 这个batch的 clusters
                for cluster in self.clusters:
                    seq2intents_user = []
                    intent_ids_user = []
                    intent_id, seq2intent = cluster.query(rec_user_emb_out)
                    seq2intents_user.append(seq2intent)
                    intent_ids_user.append(intent_id)

                '''for cluster in self.clusters_i:
                    seq2intents_item = []
                    intent_ids_item = []
                    intent_id, seq2intent = cluster.query(rec_item_emb_out)
                    seq2intents_item.append(seq2intent)
                    intent_ids_item.append(intent_id)'''

                '''for cluster in self.clusters:
                    seq2intents_item = []
                    intent_ids_item = []
                    intent_id, seq2intent = cluster.query(rec_item_emb_out)
                    seq2intents_item.append(seq2intent)
                    intent_ids_item.append(intent_id)'''

                '''cl_loss, icl_loss = self.cal_cl_loss([user_idx, pos_idx], seq2intents_user, seq2intents_item, intent_ids_user, intent_ids_item)
                cl_loss = 0.3 * cl_loss
                icl_loss = 0.2 * icl_loss

                joint_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss + icl_loss'''

                # u_cl_align, i_cl_align, bpr_align, uniform, icl_loss = self.align_uni_loss([user_idx, pos_idx], user_emb, pos_item_emb, seq2intents_user, seq2intents_item, intent_ids_user, intent_ids_item)

                u_cl_align, i_cl_align, bpr_align, uniform, icl_loss= self.align_uni_loss([user_idx, pos_idx],user_emb, pos_item_emb, seq2intents_user, intent_ids_user)
                batch_loss = (u_cl_align + i_cl_align) * 0.3 + uniform + bpr_align + 0.1 * icl_loss + l2_reg_loss(self.reg, user_emb,
                                                                                                 pos_item_emb) / self.batch_size

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'u_cl_align:', u_cl_align.item(),  'i_cl_align:', i_cl_align.item(), 'bpr_align:', bpr_align.item(), 'uniform:', uniform.item(), 'batch loss:', batch_loss.item(), 'icl_loss:', icl_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            # 评估当前模型
            measure = self.fast_evaluation(epoch)
            k, v = measure[2].strip().split(':')
            recall = float(v)



        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        # user_e, item_e = self.encoder(user, item)  # [bsz, dim]

        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + 1 * uniform
        return loss, align, uniform

    def cal_cl_loss(self, idx, intents_user, intents_item, intent_ids_user, intent_ids_item):
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()

        u_idx1 = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx1 = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        # 两个计算增强representation的encoder
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        # user_view_1, item_view_1, user_view_2, item_view_2 = [self.model.project(x) for x in [user_view_1, item_view_1, user_view_2, item_view_2]]

        # 往user_embedding和item_embedding中，都加入了噪声。所以在计算CL的时候两者都要
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        icl_loss_1 = 0
        icl_loss_2 = 0
        for intent, intent_id in zip(intents_user, intent_ids_user):
            # print("intent size:",intent.shape,"intent_id len",intent_id.shape)
            user_icl_loss_1 = self.nce_loss(user_view_1[u_idx], intent, intent_id)
            user_icl_loss_2 = self.nce_loss(user_view_2[u_idx], intent, intent_id)
            icl_loss_1 += user_icl_loss_1 + user_icl_loss_2

        for intent, intent_id in zip(intents_item, intent_ids_item):
            item_icl_loss_1 = self.nce_loss(item_view_1[i_idx], intent, intent_id)
            item_icl_loss_2 = self.nce_loss(item_view_2[i_idx], intent, intent_id)
            icl_loss_2 += item_icl_loss_1 + item_icl_loss_2

        icl_loss = (icl_loss_1 + icl_loss_2) / (2 * (len(intents_user) + len(intents_item)))
        return user_cl_loss + item_cl_loss, icl_loss


    def nce_loss(self, batch_sample_one, intents, intent_ids):
        # print("batch_sample_size:",batch_sample_one.shape,"intent_size:",intents.shape)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / 1
        sim22 = torch.matmul(intents, intents.T) / 1
        sim12 = torch.matmul(batch_sample_one, intents.T) / 1
        d = sim12.shape[-1]

        # 每行对应一个intent_id
        # intent_ids = intent_ids.contiguous().view(-1, 1)
        # 相同intent处结果=1，从而可mask掉相同intent
        mask = torch.eye(d, dtype=torch.long).to(self.device)
        sim11[mask == 1] = float("-inf")
        sim22[mask == 1] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)

        return nce_loss

    # def align_uni_loss(self, idx, user_emb, item_emb, intents_user, intents_item, intent_ids_user, intent_ids_item):
    def align_uni_loss(self, idx, user_emb, item_emb, intents_user, intent_ids_user):
        '''u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()'''
        u_idx = torch.Tensor(idx[0]).type(torch.long).cuda()
        i_idx = torch.Tensor(idx[1]).type(torch.long).cuda()

        u_idx1 = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx1 = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        # cl角度对齐
        '''u_cl_align = self.alignment(user_view_1[u_idx], user_view_2[u_idx])
        i_cl_align = self.alignment(item_view_1[i_idx], item_view_2[i_idx])'''
        u_cl_align = self.alignment(user_view_1[u_idx1], user_view_2[u_idx1])
        i_cl_align = self.alignment(item_view_1[i_idx1], item_view_2[i_idx1])

        # bpr角度对齐
        bpr_align = self.alignment(user_emb, item_emb)

        # cl角度均匀
        uniform = 2 * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2

        icl_loss_1 = 0
        icl_loss_2 = 0
        # print(len(intents_user), len(intent_ids_user))
        '''for intent, intent_id in zip(intents_user, intent_ids_user):
            # print("intent size:",intent.shape,"intent_id len",intent_id.shape)
            # print(intent," ",intent_id)
            # print("uid",u_idx)
            user_icl_loss_1 = self.nce_loss(user_view_1[u_idx], intent, intent_id)
            user_icl_loss_2 = self.nce_loss(user_view_2[u_idx], intent, intent_id)
            icl_loss_1 += (user_icl_loss_2 + user_icl_loss_1)/2'''

        # for intent in intents_user:
            # print("intent size:",intent.shape,"intent_id len",intent_id.shape)
            # print(intent," ",intent_id)
            # print("uid",u_idx)
        icl_loss_u = self.pcl_criterion(user_view_1[u_idx], user_view_2[u_idx], intents=intents_user, intent_ids=None)
        '''user_icl_loss_1 = self.nce_loss(user_view_1[u_idx], intent, None)
        user_icl_loss_2 = self.nce_loss(user_view_2[u_idx], intent, None)
        icl_loss_1 += user_icl_loss_2 + user_icl_loss_1'''
        # icl_loss_1 += 0.1 * icl_loss_u

        # icl_loss_i = self.pcl_criterion(item_view_1[i_idx], item_view_2[i_idx], intents=intents_item, intent_ids=intent_ids_item)
        # icl_loss_2 += 0.1 * icl_loss_i
        '''icl_u1 = 0
        icl_u2 = 0
        icl_i1 = 0
        icl_i2 = 0
        for intent in intents_user:
            icl_u1 = InfoNCE(user_view_1[u_idx], intent, temperature=1)
            icl_u2 = InfoNCE(user_view_2[u_idx], intent, temperature=1)
        for intent in intents_item:
            icl_i1 = InfoNCE(item_view_1[i_idx], intent, temperature=1)
            icl_i2 = InfoNCE(item_view_2[i_idx], intent, temperature=1)

        icl_loss_1 = (icl_u1 + icl_u2)/2
        icl_loss_2 = (icl_i1 + icl_i2)/2'''

        '''for intent in intents_user:
            icl_loss_1 = InfoNCE(user_view_1[u_idx], intent, temperature=1)
            icl_loss_2 = InfoNCE(user_view_2[u_idx], intent, temperature=1)'''

        # icl_loss = (icl_loss_1 + icl_loss_2) / (2 * (len(intents_user) + len(intents_item)))

        '''for intent, intent_id in zip(intents_item, intent_ids_item):
            item_icl_loss_1 = self.nce_loss(item_emb, intent, intent_id)
            # item_icl_loss_2 = self.nce_loss(item_view_2[i_idx], intent, intent_id)
            icl_loss_1 += item_icl_loss_1'''

        # icl_loss = (icl_loss_1 + icl_loss_2) / (2 * (len(intents_user) + len(intents_item)))
        # icl_loss = icl_loss_1 / 2 * (len(intents_user))

        return u_cl_align, i_cl_align, bpr_align, uniform, icl_loss_u

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        # 获得内部id
        u = self.data.get_user_id(u)
        # 内积计算得分
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class ICL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(ICL_Encoder, self).__init__()
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


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device='cpu'):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        # cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            # print("yes")
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        # self.centroids = centroids

    def query(self, x):
        # 对给定的数据 x 进行聚类查询，D:每个查询点到聚类中心的距离；I:每个查询点聚类中心索引
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        # 将索引转换为整数列表
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).cuda()
        # 返回查询数据的聚类中心索引和聚类中心的embedding
        return seq2cluster, self.centroids[seq2cluster]

class PCLoss(nn.Module):

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features:
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_id)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_id)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        # don't do de-noise
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss

class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = 1
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        '''
            两个batch通常是两个view
        '''
        # 计算同一view和不同view的相似度
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            # 创建对角阵，对角线上元素为1
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss
