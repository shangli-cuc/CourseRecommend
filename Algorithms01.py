#!/usr/bin/env python
# coding: utf-8

# # 基于物品的协同过滤算法
# 导入包
import random
import math
import time
from tqdm import tqdm


# ## 一. 通用函数定义
# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        # print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper


# ### 1. 数据处理相关
# 1. load data
# 2. split data
class Dataset():
    
    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)
    
    @timmer
    def loadData(self, fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(str, l.strip().split(',')[:2])))
        return data
    
    @timmer
    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M-1) == k:  
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test)


# ### 2. 评价指标
# 1. Precision
# 2. Recall
# 3. F1
# 4. Coverage
# 5. Popularity(Novelty)
class Metric():
    
    def __init__(self, train, test, GetRecommendation):
        '''
        :params: train, 训练数据
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()
        
    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs
        
    # 定义精确率指标计算方式
    def precision(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)
    
    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(test_items)
        return round(hit / all * 100, 2)
   
    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)
    
    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1

        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop / num, 6)
    
    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'F1': 0.0,
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        return metric


# ## 二. 算法实现
# 1. ItemCF
# 2. ItemIUF
# 3. ItemCF_Norm

# 1. 基于物品余弦相似度的推荐
def ItemCF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
    
    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True))for k, v in sim.items()}
    
    # 获取接口函数
    '''
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for u, _ in sorted_item_sim[item][:K]:
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += sim[item][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs
    '''
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for sim_item, sim_value in sorted_item_sim[item][:K]:
                if sim_item not in seen_items:
                    if sim_item not in items:
                        items[sim_item] = 0
                    items[sim_item] += sim[item][sim_item]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


# 2. 基于改进的物品余弦相似度的推荐
def ItemIUF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    ''' 
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                # 相比ItemCF，主要是改进了这里
                sim[u][v] += 1 / math.log(1 + len(items))
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
    
    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True)) for k, v in sim.items()}
    
    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for sim_item, sim_value in sorted_item_sim[item][:K]:
                # 要去掉用户见过的
                if sim_item not in seen_items:
                    if sim_item not in items:
                        items[sim_item] = 0
                    items[sim_item] += sim[item][sim_item]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs
    
    return GetRecommendation


# 3. 基于归一化的物品余弦相似度的推荐
def ItemCF_Norm(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])
            
    # 对相似度矩阵进行按行归一化
    for u in sim:
        s = 0
        for v in sim[u]:
            s += sim[u][v]
        if s > 0:
            for v in sim[u]:
                sim[u][v] /= s
    
    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(),                                key=lambda x: x[1], reverse=True))                        for k, v in sim.items()}
    
    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for sim_item, sim_value in sorted_item_sim[item][:K]:
                if sim_item not in seen_items:
                    if sim_item not in items:
                        items[sim_item] = 0
                    items[sim_item] += sim[item][sim_item]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs
    
    return GetRecommendation


# ## 三. 实验
# 1. ItemCF实验，K=[5, 10, 20, 40, 80, 160]
# 2. ItemIUF实验, K=10
# 3. ItemCF-Norm实验，K=10

# M=8, N=10
class Experiment():
    
    def __init__(self, M, K, N, fp='processedData/ratingInfo.txt', rt='ItemCF'):
        '''
        :params: M, 进行多少次实验
        :params: K, TopK相似物品的个数，即邻域
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'ItemCF': ItemCF, 'ItemIUF': ItemIUF, 'ItemCF-Norm': ItemCF_Norm}
    
    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()
    
    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0, 'F1':0,
                   'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            # print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metric['F1']=round((2*metric['Precision']*metric['Recall'])/(1.0*(metric['Precision']+metric['Recall'])),2)
            # print('Metric:', metric)
            metrics = {k: metrics[k]+metric[k] for k in metrics}
        # 获得一次完整实验的结果，保存在metrics里，并根据折数M取平均，该次实验参数分别为第M折，邻域数K，推荐列表长度N
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, K={}, N={}): {}'.format(self.M, self.K, self.N, metrics))
        return metrics



# 1. ItemCF实验
# 保存不同参数的实验结果，用于作图
'''
result=[]
M, N = 8, 10
for K in [5, 10, 20, 40, 80, 160]:
    cf_exp = Experiment(M, K, N, rt='ItemCF')
    metrics=cf_exp.run()
    result.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity'],])
precision,recall,f1,coverage,popularity=zip(*result)

# K是邻域，N是最终推荐列表长度
M, N = 8, 10
K = 10
for N in [5, 10, 20, 40, 80, 160]:
    cf_exp = Experiment(M, K, N, rt='ItemCF')
    cf_exp.run()
'''


'''
# 2. ItemIUF实验
M, N = 8, 10
K = 10 # 与书中保持一致
iuf_exp = Experiment(M, K, N, rt='ItemIUF')
iuf_exp.run()


# In[11]:


# 3. ItemCF-Norm实验
M, N = 8, 10
K = 10 # 与书中保持一致
norm_exp = Experiment(M, K, N, rt='ItemCF-Norm')
norm_exp.run()
'''

# ## 四. 实验结果
# 
# 1. ItemCF实验
# 
#     Running time: 835.2748167514801
#     
#     Average Result (M=8, K=5, N=10): 
#     {'Precision': 21.28, 'Recall': 10.22, 
#      'Coverage': 21.67, 'Popularity': 7.16666}
#      
#     Running time: 835.2476677894592
#     
#     Average Result (M=8, K=10, N=10): 
#     {'Precision': 22.17, 'Recall': 10.65, 
#      'Coverage': 19.11, 'Popularity': 7.2495425}
#      
#     Running time: 867.068473815918
#     
#     Average Result (M=8, K=20, N=10): 
#     {'Precision': 22.13, 'Recall': 10.62, 
#      'Coverage': 16.92, 'Popularity': 7.33466}
#      
#     Running time: 970.096118927002
#     
#     Average Result (M=8, K=40, N=10): 
#     {'Precision': 21.53, 'Recall': 10.34, 
#      'Coverage': 15.49, 'Popularity': 7.3892265}
#      
#     Running time: 1093.511596918106
#     
#     Average Result (M=8, K=80, N=10): 
#     {'Precision': 20.66, 'Recall': 9.92, 
#      'Coverage': 13.66, 'Popularity': 7.41055}
#      
#     Running time: 1299.2117609977722
#     
#     Average Result (M=8, K=160, N=10): 
#     {'Precision': 19.42, 'Recall': 9.32, 
#      'Coverage': 12.09, 'Popularity': 7.38311}
#      
# 2. ItemIUF实验
#     
#     Running time: 1606.6134660243988
#     
#     Average Result (M=8, K=10, N=10): 
#     {'Precision': 22.64, 'Recall': 10.87, 
#      'Coverage': 17.55, 'Popularity': 7.35}
#      
# 3. ItemCF-Norm实验
#     
#     Running time: 875.6982419490814
#     
#     Average Result (M=8, K=10, N=10): 
#     {'Precision': 22.66, 'Recall': 10.88, 
#      'Coverage': 37.68, 'Popularity': 6.999544}

# ## 五. 总结
# 1. 数据集分割的小技巧，用同样的seed
# 2. 各个指标的实现，要注意
# 3. 为每个用户推荐的时候是推荐他们**没有见过**的，因为测试集里面是这样的
# 4. 推荐的时候K和N各代表邻域和推荐长度，要分开设置，先取TopK，然后取TopN
# 5. ItemIUF的结果与书中的正好相反，书里面是PR都有些许降低，但CP有提升。但本人做的实验则是PR提升明显，CP反而降低，十分玄学。而且，在ItemCF-Norm的实验中，Coverage的提升显著，也与书中的结果有些出入

# ## 附：运行日志
# 
# 1. ItemCF实验
# Func loadData, run time: 1.357867956161499
# Func splitData, run time: 1.9750580787658691
# Experiment 0:
# Metric: {'Precision': 21.29, 'Recall': 10.22, 'Coverage': 21.3, 'Popularity': 7.167103}
# Func worker, run time: 105.31731390953064
# Func splitData, run time: 1.8287019729614258
# Experiment 1:
# Metric: {'Precision': 21.45, 'Recall': 10.27, 'Coverage': 21.85, 'Popularity': 7.151314}
# Func worker, run time: 103.2586419582367
# Func splitData, run time: 1.8108947277069092
# Experiment 2:
# Metric: {'Precision': 21.3, 'Recall': 10.18, 'Coverage': 22.03, 'Popularity': 7.165002}
# Func worker, run time: 101.99979496002197
# Func splitData, run time: 1.7960660457611084
# Experiment 3:
# Metric: {'Precision': 21.17, 'Recall': 10.18, 'Coverage': 21.34, 'Popularity': 7.178365}
# Func worker, run time: 102.11498403549194
# Func splitData, run time: 1.7130441665649414
# Experiment 4:
# Metric: {'Precision': 21.21, 'Recall': 10.2, 'Coverage': 21.8, 'Popularity': 7.170794}
# Func worker, run time: 101.90551114082336
# Func splitData, run time: 1.8183128833770752
# Experiment 5:
# Metric: {'Precision': 21.39, 'Recall': 10.32, 'Coverage': 21.76, 'Popularity': 7.163104}
# Func worker, run time: 101.97199416160583
# Func splitData, run time: 1.7958929538726807
# Experiment 6:
# Metric: {'Precision': 21.31, 'Recall': 10.25, 'Coverage': 21.9, 'Popularity': 7.161708}
# Func worker, run time: 101.57879590988159
# Func splitData, run time: 1.817734956741333
# Experiment 7:
# Metric: {'Precision': 21.16, 'Recall': 10.15, 'Coverage': 21.38, 'Popularity': 7.175929}
# Func worker, run time: 101.03642511367798
# Average Result (M=8, K=5, N=10): {'Precision': 21.284999999999997, 'Recall': 10.221250000000001, 'Coverage': 21.67, 'Popularity': 7.166664874999999}
# Func run, run time: 835.2748167514801
# Func loadData, run time: 1.2348299026489258
# Func splitData, run time: 1.8201029300689697
# Experiment 0:
# Metric: {'Precision': 22.01, 'Recall': 10.57, 'Coverage': 19.35, 'Popularity': 7.248504}
# Func worker, run time: 104.0655460357666
# Func splitData, run time: 1.8287677764892578
# Experiment 1:
# Metric: {'Precision': 22.12, 'Recall': 10.59, 'Coverage': 18.95, 'Popularity': 7.244242}
# Func worker, run time: 103.43892693519592
# Func splitData, run time: 1.804075002670288
# Experiment 2:
# Metric: {'Precision': 22.59, 'Recall': 10.8, 'Coverage': 19.19, 'Popularity': 7.245515}
# Func worker, run time: 103.44988584518433
# Func splitData, run time: 1.7733349800109863
# Experiment 3:
# Metric: {'Precision': 22.02, 'Recall': 10.58, 'Coverage': 19.37, 'Popularity': 7.245227}
# Func worker, run time: 104.05003190040588
# Func splitData, run time: 1.8094689846038818
# Experiment 4:
# Metric: {'Precision': 22.11, 'Recall': 10.63, 'Coverage': 19.33, 'Popularity': 7.260709}
# Func worker, run time: 100.68873810768127
# Func splitData, run time: 1.7294957637786865
# Experiment 5:
# Metric: {'Precision': 22.17, 'Recall': 10.69, 'Coverage': 19.02, 'Popularity': 7.251251}
# Func worker, run time: 101.01811790466309
# Func splitData, run time: 1.73459792137146
# Experiment 6:
# Metric: {'Precision': 22.4, 'Recall': 10.77, 'Coverage': 18.48, 'Popularity': 7.24112}
# Func worker, run time: 101.37971901893616
# Func splitData, run time: 1.7321960926055908
# Experiment 7:
# Metric: {'Precision': 21.98, 'Recall': 10.54, 'Coverage': 19.18, 'Popularity': 7.259772}
# Func worker, run time: 101.52781391143799
# Average Result (M=8, K=10, N=10): {'Precision': 22.174999999999997, 'Recall': 10.646249999999998, 'Coverage': 19.10875, 'Popularity': 7.2495425}
# Func run, run time: 835.2476677894592
# Func loadData, run time: 1.2376840114593506
# Func splitData, run time: 1.7310190200805664
# Experiment 0:
# Metric: {'Precision': 22.09, 'Recall': 10.61, 'Coverage': 16.78, 'Popularity': 7.331556}
# Func worker, run time: 104.41120624542236
# Func splitData, run time: 1.7812540531158447
# Experiment 1:
# Metric: {'Precision': 22.41, 'Recall': 10.73, 'Coverage': 16.87, 'Popularity': 7.327797}
# Func worker, run time: 104.20949697494507
# Func splitData, run time: 1.7324512004852295
# Experiment 2:
# Metric: {'Precision': 22.5, 'Recall': 10.76, 'Coverage': 16.83, 'Popularity': 7.330741}
# Func worker, run time: 104.43356919288635
# Func splitData, run time: 1.7455089092254639
# Experiment 3:
# Metric: {'Precision': 21.99, 'Recall': 10.57, 'Coverage': 17.12, 'Popularity': 7.339063}
# Func worker, run time: 103.83610510826111
# Func splitData, run time: 1.7279980182647705
# Experiment 4:
# Metric: {'Precision': 21.84, 'Recall': 10.5, 'Coverage': 16.99, 'Popularity': 7.340118}
# Func worker, run time: 104.33192300796509
# Func splitData, run time: 1.6494619846343994
# Experiment 5:
# Metric: {'Precision': 21.86, 'Recall': 10.54, 'Coverage': 16.85, 'Popularity': 7.3356}
# Func worker, run time: 109.94820308685303
# Func splitData, run time: 1.9003541469573975
# Experiment 6:
# Metric: {'Precision': 22.37, 'Recall': 10.75, 'Coverage': 16.8, 'Popularity': 7.321315}
# Func worker, run time: 111.54657578468323
# Func splitData, run time: 1.8000450134277344
# Experiment 7:
# Metric: {'Precision': 21.94, 'Recall': 10.52, 'Coverage': 17.12, 'Popularity': 7.351119}
# Func worker, run time: 108.88186001777649
# Average Result (M=8, K=20, N=10): {'Precision': 22.125, 'Recall': 10.6225, 'Coverage': 16.919999999999998, 'Popularity': 7.334663624999999}
# Func run, run time: 867.068473815918
# Func loadData, run time: 1.2636096477508545
# Func splitData, run time: 1.806412935256958
# Experiment 0:
# Metric: {'Precision': 21.54, 'Recall': 10.34, 'Coverage': 15.47, 'Popularity': 7.389295}
# Func worker, run time: 114.59086179733276
# Func splitData, run time: 1.8383910655975342
# Experiment 1:
# Metric: {'Precision': 22.08, 'Recall': 10.57, 'Coverage': 15.48, 'Popularity': 7.382177}
# Func worker, run time: 113.1404218673706
# Func splitData, run time: 1.806563138961792
# Experiment 2:
# Metric: {'Precision': 21.78, 'Recall': 10.41, 'Coverage': 15.07, 'Popularity': 7.382617}
# Func worker, run time: 113.9739158153534
# Func splitData, run time: 1.768733263015747
# Experiment 3:
# Metric: {'Precision': 21.47, 'Recall': 10.32, 'Coverage': 15.71, 'Popularity': 7.393157}
# Func worker, run time: 117.33210301399231
# Func splitData, run time: 1.9012501239776611
# Experiment 4:
# Metric: {'Precision': 21.24, 'Recall': 10.21, 'Coverage': 15.74, 'Popularity': 7.397843}
# Func worker, run time: 120.33364200592041
# Func splitData, run time: 1.9740369319915771
# Experiment 5:
# Metric: {'Precision': 21.12, 'Recall': 10.19, 'Coverage': 15.63, 'Popularity': 7.385106}
# Func worker, run time: 124.77882289886475
# Func splitData, run time: 1.9539570808410645
# Experiment 6:
# Metric: {'Precision': 21.7, 'Recall': 10.43, 'Coverage': 15.56, 'Popularity': 7.378948}
# Func worker, run time: 124.54463386535645
# Func splitData, run time: 1.9221651554107666
# Experiment 7:
# Metric: {'Precision': 21.28, 'Recall': 10.21, 'Coverage': 15.23, 'Popularity': 7.404669}
# Func worker, run time: 124.96897101402283
# Average Result (M=8, K=40, N=10): {'Precision': 21.526249999999997, 'Recall': 10.335, 'Coverage': 15.48625, 'Popularity': 7.3892265}
# Func run, run time: 970.096118927002
# Func loadData, run time: 1.2885360717773438
# Func splitData, run time: 1.9453051090240479
# Experiment 0:
# Metric: {'Precision': 20.71, 'Recall': 9.95, 'Coverage': 13.71, 'Popularity': 7.41184}
# Func worker, run time: 136.20149898529053
# Func splitData, run time: 1.9885108470916748
# Experiment 1:
# Metric: {'Precision': 21.06, 'Recall': 10.08, 'Coverage': 13.64, 'Popularity': 7.399879}
# Func worker, run time: 138.96288514137268
# Func splitData, run time: 1.9737217426300049
# Experiment 2:
# Metric: {'Precision': 20.69, 'Recall': 9.89, 'Coverage': 13.18, 'Popularity': 7.405965}
# Func worker, run time: 140.04792022705078
# Func splitData, run time: 1.9420268535614014
# Experiment 3:
# Metric: {'Precision': 20.69, 'Recall': 9.94, 'Coverage': 13.81, 'Popularity': 7.414836}
# Func worker, run time: 134.16970086097717
# Func splitData, run time: 1.8937797546386719
# Experiment 4:
# Metric: {'Precision': 20.53, 'Recall': 9.87, 'Coverage': 13.84, 'Popularity': 7.416375}
# Func worker, run time: 133.87962293624878
# Func splitData, run time: 1.9022479057312012
# Experiment 5:
# Metric: {'Precision': 20.36, 'Recall': 9.82, 'Coverage': 13.57, 'Popularity': 7.411035}
# Func worker, run time: 134.15448117256165
# Func splitData, run time: 1.8966238498687744
# Experiment 6:
# Metric: {'Precision': 20.79, 'Recall': 9.99, 'Coverage': 13.5, 'Popularity': 7.40158}
# Func worker, run time: 130.25284504890442
# Func splitData, run time: 1.8891630172729492
# Experiment 7:
# Metric: {'Precision': 20.46, 'Recall': 9.81, 'Coverage': 14.06, 'Popularity': 7.422915}
# Func worker, run time: 128.92854809761047
# Average Result (M=8, K=80, N=10): {'Precision': 20.66125, 'Recall': 9.91875, 'Coverage': 13.66375, 'Popularity': 7.410553125}
# Func run, run time: 1093.511596918106
# Func loadData, run time: 1.2567250728607178
# Func splitData, run time: 1.8248507976531982
# Experiment 0:
# Metric: {'Precision': 19.56, 'Recall': 9.4, 'Coverage': 11.92, 'Popularity': 7.386144}
# Func worker, run time: 151.13417983055115
# Func splitData, run time: 1.8845288753509521
# Experiment 1:
# Metric: {'Precision': 19.76, 'Recall': 9.46, 'Coverage': 12.5, 'Popularity': 7.368402}
# Func worker, run time: 153.7669289112091
# Func splitData, run time: 2.0910489559173584
# Experiment 2:
# Metric: {'Precision': 19.68, 'Recall': 9.41, 'Coverage': 11.96, 'Popularity': 7.379513}
# Func worker, run time: 164.56706523895264
# Func splitData, run time: 1.9786031246185303
# Experiment 3:
# Metric: {'Precision': 19.4, 'Recall': 9.32, 'Coverage': 12.08, 'Popularity': 7.389774}
# Func worker, run time: 164.01787090301514
# Func splitData, run time: 2.0074920654296875
# Experiment 4:
# Metric: {'Precision': 19.26, 'Recall': 9.26, 'Coverage': 12.21, 'Popularity': 7.385536}
# Func worker, run time: 163.91729307174683
# Func splitData, run time: 1.9719181060791016
# Experiment 5:
# Metric: {'Precision': 19.06, 'Recall': 9.19, 'Coverage': 12.22, 'Popularity': 7.379692}
# Func worker, run time: 165.07782125473022
# Func splitData, run time: 1.930976152420044
# Experiment 6:
# Metric: {'Precision': 19.25, 'Recall': 9.25, 'Coverage': 11.95, 'Popularity': 7.374345}
# Func worker, run time: 163.38418984413147
# Func splitData, run time: 1.8265297412872314
# Experiment 7:
# Metric: {'Precision': 19.35, 'Recall': 9.28, 'Coverage': 11.87, 'Popularity': 7.401496}
# Func worker, run time: 156.3625569343567
# Average Result (M=8, K=160, N=10): {'Precision': 19.415000000000003, 'Recall': 9.32125, 'Coverage': 12.088750000000001, 'Popularity': 7.38311275}
# Func run, run time: 1299.211760997772
# 
# 2. ItemIUF实验
# Func loadData, run time: 1.3564789295196533
# Func splitData, run time: 1.8902332782745361
# Experiment 0:
# Metric: {'Precision': 22.51, 'Recall': 10.81, 'Coverage': 17.53, 'Popularity': 7.346247}
# Func worker, run time: 202.80446100234985
# Func splitData, run time: 1.8700988292694092
# Experiment 1:
# Metric: {'Precision': 22.87, 'Recall': 10.95, 'Coverage': 17.43, 'Popularity': 7.346612}
# Func worker, run time: 202.73692202568054
# Func splitData, run time: 1.9114680290222168
# Experiment 2:
# Metric: {'Precision': 22.93, 'Recall': 10.96, 'Coverage': 17.86, 'Popularity': 7.353326}
# Func worker, run time: 202.54596090316772
# Func splitData, run time: 1.898630142211914
# Experiment 3:
# Metric: {'Precision': 22.5, 'Recall': 10.82, 'Coverage': 17.55, 'Popularity': 7.347087}
# Func worker, run time: 210.07687187194824
# Func splitData, run time: 2.0206642150878906
# Experiment 4:
# Metric: {'Precision': 22.23, 'Recall': 10.69, 'Coverage': 17.62, 'Popularity': 7.355618}
# Func worker, run time: 194.73034501075745
# Func splitData, run time: 1.8169701099395752
# Experiment 5:
# Metric: {'Precision': 22.73, 'Recall': 10.96, 'Coverage': 17.45, 'Popularity': 7.351502}
# Func worker, run time: 191.2959520816803
# Func splitData, run time: 1.7530040740966797
# Experiment 6:
# Metric: {'Precision': 22.92, 'Recall': 11.02, 'Coverage': 17.21, 'Popularity': 7.341635}
# Func worker, run time: 189.27422094345093
# Func splitData, run time: 1.7417218685150146
# Experiment 7:
# Metric: {'Precision': 22.42, 'Recall': 10.75, 'Coverage': 17.72, 'Popularity': 7.360763}
# Func worker, run time: 196.7241768836975
# Average Result (M=8, K=10, N=10): {'Precision': 22.63875, 'Recall': 10.87, 'Coverage': 17.54625, 'Popularity': 7.350348749999999}
# Func run, run time: 1606.6134660243988
# 
# 3. ItemCF-Norm实验
# Func loadData, run time: 1.2446231842041016
# Func splitData, run time: 1.8000209331512451
# Experiment 0:
# Metric: {'Precision': 22.43, 'Recall': 10.77, 'Coverage': 37.4, 'Popularity': 6.997795}
# Func worker, run time: 106.30902194976807
# Func splitData, run time: 1.8364269733428955
# Experiment 1:
# Metric: {'Precision': 22.64, 'Recall': 10.84, 'Coverage': 37.74, 'Popularity': 6.993843}
# Func worker, run time: 105.63388681411743
# Func splitData, run time: 1.820976972579956
# Experiment 2:
# Metric: {'Precision': 23.06, 'Recall': 11.03, 'Coverage': 37.05, 'Popularity': 6.994036}
# Func worker, run time: 108.06170320510864
# Func splitData, run time: 1.9166691303253174
# Experiment 3:
# Metric: {'Precision': 22.52, 'Recall': 10.82, 'Coverage': 38.16, 'Popularity': 7.003088}
# Func worker, run time: 112.67198371887207
# Func splitData, run time: 1.895146131515503
# Experiment 4:
# Metric: {'Precision': 22.56, 'Recall': 10.85, 'Coverage': 38.82, 'Popularity': 7.001239}
# Func worker, run time: 107.44453597068787
# Func splitData, run time: 1.7972638607025146
# Experiment 5:
# Metric: {'Precision': 22.74, 'Recall': 10.97, 'Coverage': 37.12, 'Popularity': 7.007333}
# Func worker, run time: 106.20418381690979
# Func splitData, run time: 1.7266900539398193
# Experiment 6:
# Metric: {'Precision': 22.82, 'Recall': 10.97, 'Coverage': 37.7, 'Popularity': 6.989135}
# Func worker, run time: 106.62335729598999
# Func splitData, run time: 1.8284211158752441
# Experiment 7:
# Metric: {'Precision': 22.49, 'Recall': 10.79, 'Coverage': 37.47, 'Popularity': 7.009879}
# Func worker, run time: 106.70008587837219
# Average Result (M=8, K=10, N=10): {'Precision': 22.6575, 'Recall': 10.879999999999999, 'Coverage': 37.682500000000005, 'Popularity': 6.9995435}
# Func run, run time: 875.6982419490814
