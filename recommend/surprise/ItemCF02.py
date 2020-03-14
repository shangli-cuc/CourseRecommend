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

class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r,None) for (uid, iid, r) in
                            zip(df['XH'], df['KCH'], df['KCCJ'])]
        # self.raw_ratings = [(sid, cid, r, cname, cdescp) for (sid, cid, r, cname, cdescp) in
                    # zip(df['XH'], df['KCH'], df['KCCJ'], df['KCMC'], df['KCJJ'])]
        self.reader=reader

class ItemBasedCF:  
    def __init__(self,data_file):  
        self.data_file = data_file 
        self.train_file = []
        self.train_len=0
        self.test_file = []
        self.test_len=0
        self.train={}
        self.test={}
        df=pd.read_csv(self.data_file,header=None)
        # 学生数
        self.n_students = df[0].unique().shape[0]
        # 课程数
        self.n_courses = df[1].unique().shape[0]
        print('Number of students = ' + str(self.n_students) + ' | Number of courses = ' + str(self.n_courses) + ' | Number of data = ' + str(len(df)))

    def splitData(self,k,M=5,seed=1):
        # seed()方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
        # 本函数没有返回值
        # 当seed()没有参数时，每次生成的随机数是不一样的，
        # 而当seed()有参数时，每次生成的随机数是一样的，
        # 同时选择不同的参数生成的随机数也不一样
        random.seed(seed)
        n=0
        self.train_file=[]
        self.test_file=[]
        self.train_len=0
        self.test_len=0
        for line in open(self.data_file):
            # if(n==0): 
            #     n+=1 
            #     continue
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
            # if len(tmp)<4: continue
            # user,item,_,score = tmp[:4]
            user,item,score = tmp
            user=user.strip()
            item=item.strip()
            self.data_dict.setdefault(user,{})  
            self.data_dict[user][item] = float(score)
        return self.data_dict

    def ItemSimilarity(self,IUF=False,normalize=False): 
        self.N = dict()  #物品被多少个不同用户购买  
        self.C = dict()  #物品-物品的共现矩阵
        for user,items in self.train.items():  
            for i in items.keys():  
                self.N.setdefault(i,0)  
                self.N[i] += 1  #物品i出现一次就计数加一
                self.C.setdefault(i,{})  
                for j in items.keys():
                    if i == j : continue  
                    self.C[i].setdefault(j,0)  
                    if(IUF==False):
                        self.C[i][j] += 1  # 基础算法，物品i和j共现一次就计数加一
                    # 改进算法IUF（Inverse User Frequence）即用户活跃度对数的倒数的参数，来修正物品相似度的计算公式。认为活跃用户对物品相似度的贡献应该小于不活跃的用户
                    else:
                        self.C[i][j] += 1/math.log(1+len(items)*1.0) 

        # print ('N:',N) 
        # print ('C:',C)

        #计算相似度矩阵  
        self.W = dict()  
        
        # 一般的计算物品相似度
        for i,related_items in self.C.items():
            self.W.setdefault(i,{})  
            for j,cij in related_items.items():  
                self.W[i][j] = cij / (math.sqrt(self.N[i] * self.N[j]))  #按上述物品相似度公式计算相似度

        # for k,v in self.W.items():
        #     print (k+':'+str(v))
        # return self.W  
        
        if(normalize==True):
        # 归一化计算物品相似度
            self.W_max = dict() #记录每一列的最大值
            for i,related_items in self.C.items(): 
                self.W.setdefault(i,{})  
                for j,cij in related_items.items(): 
                    self.W_max.setdefault(j,0.0)#
                    self.W[i][j] = cij / (math.sqrt(self.N[i] * self.N[j]))  
                    if self.W[i][j]>self.W_max[j]:#
                        self.W_max[j]=self.W[i][j] #记录第j列的最大值，按列归一化
            # print('W:',self.W)
            for i,related_items in self.C.items():  #
                for j,cij in related_items.items(): #
                    self.W[i][j]=self.W[i][j] / self.W_max[j] # 按上述物品相似度公式计算相似度
        
        # print('W_max:',self.W_max)
        # 相似度矩阵
        # for k,v in self.W.items():
        #     print (k+':'+str(v))

        return self.W  

    # K：物品的相似邻居数
    # N：topN推荐
    # def Recommend(self,user,K=3,N=10):  
    def Recommend(self,user,K=3):  
        # 如果训练集中没有该用户，无法推荐
        if(user not in self.train):
            # print("训练集中不存在"+user)
            return
        N=len(self.test[user])
        # 如果存在，找出目标用户喜爱的物品，寻找与这些物品相似的物品集合
        rank=dict()
        train_items=self.train[user]
        for item,score in train_items.items():
            # if item not in self.W.keys():continue
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:
                if j in train_items.keys():continue
                rank.setdefault(j,0.0)
                rank[j]+=score*wj
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])

        
'''
        rank = dict() #记录user的推荐物品（没有历史行为的物品）和兴趣程度
        # action_item = self.train[user]     #用户user购买的物品和兴趣评分r_ui，现在还是统一为1，可以将成绩因素加入进去，将score即r_ui，根据成绩因素得到不同的值
        action_item = self.test[user]     #用户user购买的物品和兴趣评分r_ui，现在还是统一为1，可以将成绩因素加入进去，将score即r_ui，根据成绩因素得到不同的值
        train_item=self.train[user]
        for item,score in action_item.items():
            # 因为是直接对test内的user进行推荐，所以造成相似度矩阵 W 中可能没有该 user 的某个 item 的相似度值
            if item not in self.W.keys(): continue
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:  #使用与物品item最相似的K个物品进行计算
                if j in action_item.keys():  #如果物品j已经购买过，则不进行推荐
                    continue
                rank.setdefault(j,0)  
                rank[j] += score * wj  #如果物品j没有购买过，则累计物品j与item的相似度*兴趣评分，作为user对物品j的兴趣程度
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N]) 
'''
def ItemCF02(fileName,neighbors=range(1,41,1),IUF=False,normalize=False):
    #声明一个ItemBased推荐的对象
    Item = ItemBasedCF(fileName)#读取数据集 

    # 返回包含5折交叉验证后precision、recall、f1均值的result列表
    result=[]
    # 将用户行为数据集按照均匀分布随机分成 M 份，挑选一份作为测试集，将剩下的 M - 1 份作为训练集
    # 然后在训练集上建立用户兴趣模型，并在测试集上对用户行为进行预测，统计出相应的评测指标
    # 为了保证评测指标并不是过拟合的结果，需要进行 M 次实验，并且每次都使用不同的测试集
    # 然后将 M 次实验测出的评测指标的平均值作为最终的评测指标
    # 每次实验选取不同的 k（ 0 ≤ k ≤ M - 1 ）和相同的随机数种子 seed ，进行 M 次实验就可以得到 M 个不同的训练集和测试集
    # 如果数据集够大，模型够简单，为了快速通过离线实验初步地选择算法，也可以只进行一次实验
    # neighbor：相似物品近邻数
    for neighbor in neighbors:
        # M折交叉验证
        M=1
        pre_lst=[]
        rec_lst=[]
        f1_lst=[]
        coverage_lst=[]
        for k in range(M): #进行5次交叉验证
            Item.splitData(k,M,seed=1)
            Item.train=Item.readData(Item.train_file)
            Item.test=Item.readData(Item.test_file) 
            Item.ItemSimilarity(IUF=IUF,normalize=normalize) #计算物品相似度矩阵 
            recommendDic = dict()
            # print('训练集数量',Item.train_len)
            # print('测试集数量',Item.test_len)

            # 用户评分表示满意的阈值
            threshold=3.0
            precisions=[]
            recalls=[]
            f1s=[]
            # 记录推荐课程数用于计算覆盖率coverages
            recommend_courses=set()
            # for user in  Item.train.keys():
            for user,user_ratings in  Item.test.items():
                # 推荐列表长度，precision分母
                n_recommend=0.0

                # 测试集中正评分课程数量，即rating大于threshold，recall分母
                n_relevant=0.0

                # 推荐结果中正评分课程数量，即推荐结果的预测rating大于threshold，precision、recall分子
                n_recommend_relevant=0.0

                recommendDic[user] = Item.Recommend(user,K=neighbor) #对于训练user生成推荐列表
                if(recommendDic[user]==None):continue
                test_course=Item.test[user].keys()
                recommend_course = recommendDic[user].keys()
                
                # 对每个用户的推荐列表长度，等于该用户在测试集中的评分数，即根据该目标用户在测试集中评分的物品数量对他做出相应数量的推荐
                n_recommend = len(recommendDic[user])
                n_relevant = sum((rating >= threshold) for _,rating in user_ratings.items())

                # 推荐结果与测试集吻合的课程列表
                recommend_test_course=[x for x in recommend_course if x in test_course]
                if(len(recommend_test_course)==0):
                    n_recommend_relevant=0
                else:
                    # n_recommend_relevant = sum((rating >= threshold) for rating in recommedDic[user].values())
                    n_recommend_relevant = sum((recommendDic[user][course] >= threshold) for course in recommend_test_course) 
                if(n_recommend==0 or n_relevant==0 or n_recommend_relevant==0):
                    # print(user)
                    continue
                for course in recommend_course:
                    recommend_courses.add(course)

                precision = n_recommend_relevant*1.0/n_recommend*1.0
                recall = n_recommend_relevant*1.0/n_relevant*1.0
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(2.0*(precision*recall)/(precision+recall))
            
            # 这里的平均是对每一折实验中测试集内所有被测试用户平均，即每一折的最终数据，之后还要对每一折的结果取平均，得到这一次M折交叉验证试验的结果
            pre = np.mean(precisions)
            rec = np.mean(recalls)
            f1 = np.mean(f1s)
            # coverage不再对每一折实验中测试集内所有测试用户平均，直接得到一折实验的结果
            coverage = len(recommend_courses)/(Item.n_courses*1.0)

            pre_lst.append(pre)
            rec_lst.append(rec)
            f1_lst.append(f1)
            coverage_lst.append(coverage)
            # print(k,' precision:',pre,'recall:',rec,'f1:',f1,'coverage:',coverage) 

        # 这里的平均即是对每一折的结果取平均，得到这一次M折交叉验证试验的结果
        pre=np.mean(pre_lst)
        rec=np.mean(rec_lst)
        f1=np.mean(f1_lst)
        coverage=np.mean(coverage_lst)
        result.append([pre,rec,f1,coverage])
        print("neighbor：",neighbor)
        print(pre_lst,'precision平均：',pre)
        print(rec_lst,'recall平均：',rec)
        print(f1_lst,'F1平均：',f1)
        print(coverage_lst,'coverage平均：',coverage)
    return result

if __name__=='__main__':
    start=time.time()
    
    # fileName="uid_score_bid.csv"
    # fileName="191007.csv"
    # fileName="2013191024.csv"
    # fileName="toySet.csv"
    fileName="data/csv/ratingInfo2013.csv"
    ItemCF02(fileName,neighbors=range(1,41,1),IUF=False)
    end=time.time()
    print("finish....")
    print("total time:%.1f s"%(end-start))
