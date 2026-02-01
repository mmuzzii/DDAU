from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []

        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)


    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict 为测试集中的每个用户生成推荐列表
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            # 剔除已经在train_set里交互过的物品
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            # 为每个用户创建一个包含物品名和对应得分的列表，并存储在 rec_list 字典中
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.output['-dir']
        file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        # return measure

        return measure


    def train_pre(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict 为测试集中的每个用户生成推荐列表
        rec_list = {}
        user_count = len(self.data.training_set_u)
        for i, user in enumerate(self.data.training_set_u):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            # 剔除已经交互过的物品
            '''rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8'''

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            # 为每个用户创建一个包含物品名和对应得分的列表，并存储在 rec_list 字典中
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def train_evaluation(self, pop_indices, unpop_indices):
        print('Evaluating the model on train_set...')
        rec_list = self.train_pre()
        pop_rec_list, unpop_rec_list = self.separate_recommendations(rec_list, pop_indices, unpop_indices)
        measure_pop = ranking_evaluation(self.data.training_set_u, pop_rec_list, [self.max_N])
        measure_unpop = ranking_evaluation(self.data.training_set_u, unpop_rec_list, [self.max_N])

        print('-' * 120)
        print(' (Top-' + str(self.max_N) + ' Item Recommendation in train set)')
        measure1 = [m.strip() for m in measure_pop[1:]]
        print('*popular*')
        print(measure1)

        measure2 = [m.strip() for m in measure_unpop[1:]]
        print('*unpopular*')
        print(measure2)

        k, v = measure1[3].strip().split(':')
        NDGC_pop = float(v)
        k, v = measure2[3].strip().split(':')
        NDGC_unpop = float(v)

        return NDGC_pop,NDGC_unpop

    def separate_recommendations(self, rec_list, popular_items, unpopular_items):
        # 将项目名称转换为集合以便快速查找
        popular_items_set = set(popular_items)
        unpopular_items_set = set(unpopular_items)

        # 创建存储流行和不流行推荐的字典
        popular_rec_list = {}
        unpopular_rec_list = {}

        # 遍历所有用户及其推荐列表
        for user, items in rec_list.items():
            # 初始化用户的推荐列表
            popular_rec_list[user] = []
            unpopular_rec_list[user] = []

            # 检查每个推荐项是否属于流行或不流行
            for item_name, score in items:
                if item_name in popular_items_set:
                    popular_rec_list[user].append((item_name, score))
                elif item_name in unpopular_items_set:
                    unpopular_rec_list[user].append((item_name, score))

        return popular_rec_list, unpopular_rec_list

    def test_evaluation(self, pop_indices, unpop_indices):
        print('Evaluating the model on test set...')
        rec_list = self.test()
        pop_rec_list, unpop_rec_list = self.separate_recommendations(rec_list, pop_indices, unpop_indices)
        measure_pop = ranking_evaluation(self.data.test_set, pop_rec_list, [self.max_N])
        measure_unpop = ranking_evaluation(self.data.test_set, unpop_rec_list, [self.max_N])

        print('-' * 120)
        print(' (Top-' + str(self.max_N) + ' Item Recommendation in test set)')
        measure1 = [m.strip() for m in measure_pop[1:]]
        print('*popular*')
        print(measure1)

        measure2 = [m.strip() for m in measure_unpop[1:]]
        print('*unpopular*')
        print(measure2)

        k, v = measure1[3].strip().split(':')
        NDGC_pop = float(v)
        k, v = measure2[3].strip().split(':')
        NDGC_unpop = float(v)

        return NDGC_pop, NDGC_unpop