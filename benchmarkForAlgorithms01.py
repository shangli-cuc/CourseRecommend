"""
This module illustrates how to compute Precision at k and Recall at k metrics.
"""

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


# Ks = range(5 , 30 , 5)

# ugly dict to map algo names and datasets to their markdown links in the table
stable = 'http://surprise.readthedocs.io/en/stable/'
LINK = {'SVD': '[{}]({})'.format('SVD',
                                 stable +
                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD'),
        'SVDpp': '[{}]({})'.format('SVD++',
                                   stable +
                                   'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp'),
        'NMF': '[{}]({})'.format('NMF',
                                 stable +
                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF'),
        'SlopeOne': '[{}]({})'.format('Slope One',
                                      stable +
                                      'slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne'),
        'KNNBasic': '[{}]({})'.format('k-NN',
                                      stable +
                                      'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic'),
        'KNNWithMeans': '[{}]({})'.format('Centered k-NN',
                                          stable +
                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans'),
        'KNNBaseline': '[{}]({})'.format('k-NN Baseline',
                                         stable +
                                         'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline'),
        'CoClustering': '[{}]({})'.format('Co-Clustering',
                                          stable +
                                          'co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering'),
        'BaselineOnly': '[{}]({})'.format('Baseline',
                                          stable +
                                          'basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly'),
        'NormalPredictor': '[{}]({})'.format('Random',
                                             stable +
                                             'basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor'),
        'ml-100k': '[{}]({})'.format('Movielens 100k',
                                     'http://grouplens.org/datasets/movielens/100k'),
        'ml-1m': '[{}]({})'.format('Movielens 1M',
                                   'http://grouplens.org/datasets/movielens/1m'),
        }


def precision_recall_at_k(predictions, k=10, threshold=3):
    '''Return precision and recall at k metrics for each user.'''

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

# dataset = 'ml-100k'
# data = Dataset.load_builtin('ml-100k')

dataset = 'courseSet'
# df=pd.read_excel("data/xlsx/ratingInfo2013.xlsx")
df=pd.read_csv("toySet.csv")

data=Dataset.load_from_df(df,reader=Reader(line_format='user item rating timestamp', sep=','))

# data=Dataset.load_from_file("2013_2.csv",reader=Reader(line_format='user item rating timestamp', sep=','))
kf = KFold(n_splits=5)
trainset,testset = train_test_split(data,test_size=0.25)
'''
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
    # Precision and recall can then be averaged over all users
    prec = sum(p for p in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    f1 = 2 * prec * recall / (prec + recall)
    print(prec)
    print(recall)
    print(f1)
'''
table = []
classes = (SVD, SVDpp, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
           BaselineOnly, NormalPredictor)
# classes=[SVD,SVDpp]
Ks = range(1,101,1)
page = Page()
name=str()

linePrecision=Line()
lineRecall=Line()
lineF1=Line()
for klass in classes:
    # start = time.time()
    if klass.__name__ == 'SVD':
        algo = SVD()
        name="SVD"
    elif klass.__name__ == 'SVDpp':
        algo = SVDpp()
        name="SVD++"
    elif klass.__name__ == 'PMF':
        algo = SVD(biased=False)
        name="PMF"
    # elif klass.__name__ == 'NMF':
    #     algo = NMF()
    #     name="NMF"
    elif klass.__name__ == 'SlopeOne':
        algo = SlopeOne()
        name="SlopeOne"
    elif klass.__name__ == 'KNNBasic':
        # 基于用户的协同过滤
        algo = KNNBasic(k=40,sim_options = {'name': 'cosine','user_based': True})
        name="CFBaseUser"
    elif klass.__name__ == 'KNNBasic':
        # 基于物品的协同过滤
        algo = KNNBasic(k=40,sim_options = {'name': 'cosine','user_based': False})
        name="CFBaseItem"
    elif klass == 'KNNWithMeans':
        algo = KNNWithMeans()
        name="KNNWithMeans"
    elif klass.__name__ == 'KNNBaseline':
        algo = KNNBaseline()
        name = "KNNBaseline"
    # elif klass.__name__ == 'CoClustering':
    #     algo = CoClustering()
    #     name="CoClustering"
    elif klass.__name__ == 'BaselineOnly':
        algo = BaselineOnly()
        name="BaselineOnly"
    else :
        algo = NormalPredictor()
        name="NormalPredictor"
    #cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    # precisions
    result=[]
    for k in Ks:
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, threshold=3)
        prec = sum(p for p in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        f1 = 2 * prec * recall / (prec + recall)
        link = LINK[klass.__name__]
        link = '{}@{}'.format(klass.__name__,k)
        new_line = [link, prec, recall, f1]
        # print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
        table.append(new_line)
        temp=[]
        temp.append(prec)
        temp.append(recall)
        temp.append(f1)
        result.append(temp)

    precisionTemp,recallTemp,f1Temp=zip(*result)

    linePrecision.add_xaxis(Ks)
    linePrecision.add_yaxis(klass.__name__,precisionTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    linePrecision.set_global_opts(
            title_opts=opts.TitleOpts(title="算法precision对比"),
            xaxis_opts=opts.AxisOpts(name="邻域K"),
            yaxis_opts=opts.AxisOpts(name="precision"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )

    lineRecall.add_xaxis(Ks)
    lineRecall.add_yaxis(klass.__name__,recallTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    lineRecall.set_global_opts(
            title_opts=opts.TitleOpts(title="算法recall对比"),
            xaxis_opts=opts.AxisOpts(name="邻域K"),
            yaxis_opts=opts.AxisOpts(name="recall"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )

    lineF1.add_xaxis(Ks)
    lineF1.add_yaxis(klass.__name__,f1Temp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    lineF1.set_global_opts(
            title_opts=opts.TitleOpts(title="算法F1对比"),
            xaxis_opts=opts.AxisOpts(name="邻域K"),
            yaxis_opts=opts.AxisOpts(name="F1"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )
page.add(linePrecision,lineRecall,lineF1)
# page.add(linePrecision)

page.render("对比.html")

    # Precision and recall can then be averaged over all users
    
# header = [LINK[dataset],
#           'Precision',
#           'Recall',
#           'F1'
#           ]
# print(tabulate(table, header, tablefmt="pipe"))
# print(tabulate(table, tablefmt="pipe"))

