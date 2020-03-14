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


class CF:
    def __init__(self, courses, ratings, k=5, n=10):
        self.courses = courses
        self.ratings = ratings

        # 邻居个数
        self.k = k

        # 推荐个数
        self.n = n

        # 用户对课程的评分
        # 数据格式{'UserID：用户ID':[(courseID：课程ID,Rating：用户对课程的评分)]}
        self.userDict = {}

        # 对某课程评分的用户
        # 数据格式：{'courseID：课程ID',[UserID：用户ID]}
        # {'1',[1,2,3..],...}
        self.ItemUser = {}

        # 邻居的信息
        self.neighbors = []

        # 推荐列表
        self.recommandList = []
        self.cost = 0.0

    # 基于用户的推荐
    # 根据对课程的评分计算用户之间的相似度
    def recommendByUser(self, userId):
        self.formatRate()
        # 推荐个数等于本身评分课程个数，用户计算准确率
        self.n = len(self.userDict[userId])
        self.getNearestNeighbor(userId)
        self.getrecommandList(userId)
        self.getPrecision(userId)

    # 获取推荐列表
    def getrecommandList(self, userId):
        self.recommandList = []
        # 建立推荐字典
        recommandDict = {}
        for neighbor in self.neighbors:
            courses = self.userDict[neighbor[1]]
            for course in courses:
                if(movie[0] in recommandDict):
                    recommandDict[course[0]] += neighbor[0]
                else:
                    recommandDict[course[0]] = neighbor[0]

        # 建立推荐列表
        for key in recommandDict:
            self.recommandList.append([recommandDict[key], key])
        self.recommandList.sort(reverse=True)
        self.recommandList = self.recommandList[:self.n]
        
    # 将ratings转换为userDict和ItemUser
    def formatRate(self):
        self.userDict = {}
        self.ItemUser = {}
        for i in self.ratings:
            # 评分最高为5 除以5 进行数据归一化
            temp = (i[1], float(i[2]))
            # 计算userDict {'1':[(1,5),(2,5)...],'2':[...]...}
            if(i[0] in self.userDict):
                self.userDict[i[0]].append(temp)
            else:
                self.userDict[i[0]] = [temp]
            # 计算ItemUser {'1',[1,2,3..],...}
            if(i[1] in self.ItemUser):
                self.ItemUser[i[1]].append(i[0])
            else:
                self.ItemUser[i[1]] = [i[0]]

    # 找到某用户的相邻用户
    def getNearestNeighbor(self, userId):
        neighbors = []
        self.neighbors = []
        # 获取userId评分的课程都有那些用户也评过分
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if(j != userId and j not in neighbors):
                    neighbors.append(j)
        # 计算这些用户与userId的相似度并排序
        for i in neighbors:
            dist = self.getCost(userId, i)
            self.neighbors.append([dist, i])
        # 排序默认是升序，reverse=True表示降序
        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.k]

    # 格式化userDict数据
    def formatuserDict(self, userId, l):
        user = {}
        for i in self.userDict[userId]:
            user[i[0]] = [i[1], 0]
        for j in self.userDict[l]:
            if(j[0] not in user):
                user[j[0]] = [0, j[1]]
            else:
                user[j[0]][1] = j[1]
        return user
        
    # 计算余弦距离
    def getCost(self, userId, l):
        # 获取用户userId和l评分课程的并集
        # {'课程ID'：[userId的评分，l的评分]} 没有评分为0
        user = self.formatuserDict(userId, l)
        x = 0.0
        y = 0.0
        z = 0.0
        for k, v in user.items():
            x += float(v[0]) * float(v[0])
            y += float(v[1]) * float(v[1])
            z += float(v[0]) * float(v[1])
        if(z == 0.0):
            return 0
        return z / sqrt(x * y)

    # 推荐的准确率
    def getPrecision(self, userId):
        user = [i[0] for i in self.userDict[userId]]
        recommand = [i[1] for i in self.recommandList]
        count = 0.0
        if(len(user) >= len(recommand)):
            for i in recommand:
                if(i in user):
                    count += 1.0
            self.cost = count / len(recommand)
        else:
            for i in user:
                if(i in recommand):
                    count += 1.0
            self.cost = count / len(user)

    # 显示推荐列表
    def showTable(self):
        neighbors_id = [i[1] for i in self.neighbors]
        # table = Texttable()
        # table.set_deco(Texttable.HEADER)
        # table.set_cols_dtype(["t", "t", "t", "t"])
        # table.set_cols_align(["l", "l", "l", "l"])
        table=[]
        rows = []
        # rows.append([u"movie ID", u"Name", u"release", u"from userID"])
        rows.append(['courseId','courseName'])
        for item in self.recommandList:
            fromID = []
            for i in self.movies:
                if i[0] == item[1]:
                    movie = i
                    break
            for i in self.ItemUser[item[1]]:
                if i in neighbors_id:
                    fromID.append(i)
            movie.append(fromID)
            rows.append(movie)
            # print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
            # table.append(new_line)
        table.append(rows)
        print(tabulate(table,headers='firstrow'))
        # table.add_rows(rows)
        # print(table.draw())


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def precision_recall(top_n, userDict, threshold=3):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid1, values1 in top_n.items():
        for uid2, values2 in userDict.items():
            if(uid1==uid2):
                recommandList=[cid1 for cid1,est_rating in values1]
                n_relevant = sum((true_r >= threshold) for (cid2, true_r) in values2)
                n_recommend = sum((est >= threshold) for (cid1, est) in values1)
                n_relevant_and_recommend = sum(((true_r >= threshold) and (est >= threshold))
                    for (est, true_r) in values1)
                print()
                # k=len()
        # user_est_true[uid].append((est, true_r))
        print()

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        
        k=len(top_n)
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

# 获取数据
def readFile(filename):
    files = open(filename, "r", encoding="utf-8")
    # 如果读取不成功试一下
    # files = open(filename, "r", encoding="iso-8859-15")
    data = []
    for line in files.readlines()[1:]:
        item = line.strip().split(",")
        data.append(item)
    return data

# 将ratings转换为userDict和ItemUser
def formatRate(ratings):
    userDict = {}
    ItemUser = {}
    for i in ratings:
        # 评分最高为5 除以5 进行数据归一化
        temp = (i[1], float(i[2]))
        # 计算userDict {'1':[(1,5),(2,5)...],'2':[...]...}
        if(i[0] in userDict):
            userDict[i[0].strip()].append(temp)
        else:
            userDict[i[0].strip()] = [temp]
        # 计算ItemUser {'1',[1,2,3..],...}
        if(i[1] in ItemUser):
            ItemUser[i[1].strip()].append(i[0].strip())
        else:
            ItemUser[i[1].strip()] = [i[0].strip()]
    return userDict,ItemUser

# First train an SVD algorithm on the movielens dataset.
# data = Dataset.load_builtin('ml-100k')
reader = Reader(line_format='user item rating', sep=',')
data=Dataset.load_from_file("data/csv/ratingInfo2013.csv",reader=reader)
# data = Dataset.load_from_file("toySet.csv", reader=reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
testset = trainset.build_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
ratings=readFile("data/csv/ratingInfo2013.csv")
userDict,ItemUser = formatRate(ratings)

# for i in self.ratings:
#     if(i[0] in result):
#         result[uid].append(temp)
#     else:
#         result[]=[temp]

precision_recall(top_n,userDict,threshold=3)