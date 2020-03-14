from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import datetime
import random

import numpy as np
import six
from tabulate import tabulate
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import train_test_split

from pyecharts.charts import Bar, Pie, Line, Page
from pyecharts import options as opts


def precision_recall_at_k(predictions, k=10, threshold=3):
    '''Return precision and recall at k metrics for each user.这里的k是topK'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

# 非冷启动实验结果
# 将原数据集随机分割n_splits次
# 每次均会产生一个训练集和测试集，测试集占比0.2，因为是随机分割，必然会存在3种情况：
# 一个用户的评分全部在训练集、分别在训练集和测试集、全部在测试集
# 测试推荐指标时，用训练集结果来预测测试集中的用户评分，得到测试集中对应用户对物品的预测评分
# 这时，如果真实评分高于阈值，认为是相关物品，如果预测评分高于阈值，命中推荐
# 这里没有产生topN的推荐，是直接对有过用户行为的物品进行评分预测，判定准确度
# precision分子是命中数，分母是测试集长度
# recall分子是命中数，分母是真实

# data = Dataset.load_builtin('ml-100k')
reader = Reader(line_format='user item rating', sep=',')
# data = Dataset.load_from_file("data/csv/ratingInfo2013.csv",reader=reader)
data = Dataset.load_from_file("processedData/ratingInfo.txt", reader=reader)
kf = KFold(n_splits=5)
algo = SVD()
algo = NormalPredictor()
algo = KNNBaseline(k=10,sim_options={'name': 'cosine','user_based': False})
algo = KNNBasic(k=10,sim_options={'name': 'cosine','user_based': False})


for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    # sumPrecision=0.0
    # for _,precision in precisions.items():
    #     sumPrecision+=precision
    # print(sumPrecision/len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))
    print()
