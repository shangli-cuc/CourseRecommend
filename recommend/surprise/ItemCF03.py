import os
import io
import sys
import math
import time
import random
import numpy as np
import pandas as pd

from operator import itemgetter
from collections import defaultdict

from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split

from surprise import Dataset
from surprise import dataset
from surprise import Reader
from surprise import evaluate, print_perf
from surprise import accuracy

from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import BaselineOnly
from surprise import BaselineOnly
from surprise import NormalPredictor

random.seed(0)

# Also, a dummy Dataset class
class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r,None) for (uid, iid, r,_) in
                            zip(df['XH'], df['KCH'], df['KCCJ'], df['rating'])]
        # self.raw_ratings = [(sid, cid, r, cname, cdescp) for (sid, cid, r, cname, cdescp) in
                    # zip(df['XH'], df['KCH'], df['KCCJ'], df['KCMC'], df['KCJJ'])]
        self.reader=reader


class ItemBasedCF:
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self,data_file):
        self.data_file = data_file 
        self.train_file = []
        self.train_len=0
        self.test_file = []
        self.test_len=0
        self.trainset = {}
        self.testset = {}


        self.n_sim_course = 20
        self.n_rec_course = 10

        self.course_sim_mat = {}
        self.course_popular = {}
        self.course_count = 0

        df=pd.read_csv(self.data_file,header=None)
        n_students = df[0].unique().shape[0]
        n_courses = df[1].unique().shape[0]
        print('Number of students = ' + str(n_students) + ' | Number of courses = ' + str(n_courses) + ' | Number of data = ' + str(len(df)))

        # n_students = df['XH'].unique().shape[0]
        # n_courses = df['KCH'].unique().shape[0]
        # print('Number of students = ' + str(n_students) + ' | Number of courses = ' + str(n_courses) + ' | Number of data = ' + str(len(df)))

        # self.precision=[]
        # self.recall=[]
        # self.F1=[]

    def splitData(self,k,M=5,seed=1):
        random.seed(seed)
        n=0
        for line in open(self.data_file):
            if(n==0): 
                n+=1 
                continue
            randomNum=random.randint(0,M)
            # print(randomNum)
            if randomNum==k:
                self.test_file.append(line)
                self.test_len+=1
            else:
                self.train_file.append(line)
                self.train_len+=1

    def readData(self,file):  
        #读取文件，并生成用户-物品的评分表 
        self.data_dict = dict()     #用户-物品的评分表  
        for line in file:
            tmp = line.strip().split(",")
            if len(tmp)<4: continue
            # user,item,_,score= tmp[:4] # recommend方法中以相似度*1排序，即直接以相似度排序
            user,item,score,_= tmp[:4] # recommend方法中以相似度*grade排序
                                       # recommend方法中参考grade，以一定权重优化相似度排序
            self.data_dict.setdefault(user,{})  
            self.data_dict[user][item] = int(float(score)) 
        return self.data_dict    

    def calc_course_sim(self):
        for student, courses in self.trainset.items():
            for course in courses:
                # count item popularity
                if course not in self.course_popular:
                    self.course_popular[course] = 0
                self.course_popular[course] += 1

        self.course_count = len(self.course_popular)
        print('total course number = %d' % self.course_count)
        itemsim_mat = self.course_sim_mat

        for student, courses in self.trainset.items():
            for c1 in courses:
                itemsim_mat.setdefault(c1, defaultdict(int))
                for c2 in courses:
                    if c1 == c2:
                        continue
                    # 基础算法，物品i和j共现一次就计数加一，活跃度置为1
                    # itemsim_mat[c1][c2] += 1
                    # 修正后的活跃度
                    # 改进算法IUF（Inverse User Frequence）即用户活跃度对数的倒数的参数，来修正物品相似度的计算公式。认为活跃用户对物品相似度的贡献应该小于不活跃的用户
                    itemsim_mat[c1][c2] += 1/math.log(1+len(courses)*1.0) 

        simfactor_count = 0
        PRINT_STEP = 2000000

        for c1, related_courses in itemsim_mat.items():
            for c2, count in related_courses.items():
                itemsim_mat[c1][c2]=count/math.sqrt(self.course_popular[c1]*self.course_popular[c2])
                simfactor_count += 1

    def recommend(self, student):
        ''' Find K similar movies and recommend N movies. '''
        K = self.n_sim_course
        N = self.n_rec_course
        rank = {}
        selected_courses = self.trainset[student]

        for course, rating in selected_courses.items():
            for related_course, similarity_factor in sorted(self.course_sim_mat[course].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_course in selected_courses:
                    continue
                rank.setdefault(related_course, 0)
                rank[related_course] += similarity_factor * rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        N = self.n_rec_course
        # varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_courses = set()
        # varables for popularity
        popular_sum = 0

        for i, student in enumerate(self.trainset):
            rec_courses = self.recommend(student)
            test_courses = self.testset.get(student, {})
            for course, _ in rec_courses:
                if course in test_courses:
                    hit += 1
                all_rec_courses.add(course)
                popular_sum += math.log(1 + self.course_popular[course])
            rec_count += N
            test_count += len(test_courses)

            # 输出推荐结果
            # if i % 500 == 0:
            #     print('recommended for %d students' % i)
            #     print("recommend student: %s for %d courses: " % (student,self.n_rec_course))
            #     for rec_course in rec_courses:
            #         print(rec_course[0]+"   ",end="")
            #     print()
        # if(hit==0)
        self.precision = hit / (1.0 * rec_count)
        self.recall = hit / (1.0 * test_count)
        self.F1=(2*self.precision*self.recall)/(self.precision+self.recall)

        coverage = len(all_rec_courses) / (1.0 * self.course_count)
        popularity = popular_sum / (1.0 * rec_count)

        # print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
        #        (precision, recall, coverage, popularity), file=sys.stderr)
        # print('precision=%.4f\trecall=%.4f\tF1=%.4f' % (precision, recall, F1))
        # print('precision=%.4f\trecall=%.4f\tF1=%.4f' % (precision, recall, F1))
        # print('precision=%.4f\trecall=%.4f\t' % (precision, recall))

        # self.precision.append(precision)
        # self.recall.append(rec_count)
        # self.F1.append(F1)

if __name__ == '__main__':
    # fileName='2013.csv'
    fileName='processedData/ratingInfo.txt'
    M=5
    pre_lst=[]
    rec_lst=[]
    F1_lst=[]
    for k in range(M): #进行5次交叉验证
        Item = ItemBasedCF(fileName)
        Item.train_len=0
        Item.test_len=0
        Item.splitData(k,M,seed=1)
        Item.trainset=Item.readData(Item.train_file)
        Item.testset=Item.readData(Item.test_file) 
        print('训练集数量',Item.train_len)
        print('测试集数量',Item.test_len)
        Item.calc_course_sim() # 计算物品相似度矩阵 
        Item.evaluate()
        pre_lst.append(Item.precision)
        rec_lst.append(Item.recall)
        F1_lst.append(Item.F1)

    print(pre_lst)
    print("平均值：precision = %.4f" % np.mean(pre_lst))
    print(rec_lst)
    print("平均值：recall = %.4f" % np.mean(rec_lst))
    print(F1_lst)
    print("平均值：F1 = %.4f" % np.mean(F1_lst))
    