from pyecharts.charts import Bar, Pie, Line, Page
from pyecharts import options as opts
import Algorithms02
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# result保存不同参数的实验结果，用于作图
result_cf_exp=[]
result_iuf_exp=[]
result_norm_exp=[]
# 实验折数M、推荐列表长度N、邻域K
M, N, Ks = 8, 10, [5, 10, 20, 40, 80, 160]
for K in Ks:
    cf_exp = Algorithms02.Experiment(M, K, N, rt='ItemCF')
    metrics = cf_exp.run()
    result_cf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])
    
    iuf_exp = Algorithms02.Experiment(M, K, N, rt='ItemIUF')
    metrics = iuf_exp.run()
    result_iuf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

    norm_exp = Algorithms02.Experiment(M, K, N, rt='ItemCF-Norm')
    metrics = norm_exp.run()
    result_norm_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

result1=list(zip(*result_cf_exp))
result2=list(zip(*result_iuf_exp))
result3=list(zip(*result_norm_exp))
graph_name=['precision对比结果','recall对比结果','f1对比结果','coverage对比结果','popularity对比结果']
y_name=['precision %','recall %','f1 %','coverage %','popularity %']
for i in range(5):
    plt.figure()
    plt.title(graph_name[i]) 
    plt.xlabel("邻域K")
    plt.ylabel(y_name[i])
    plt.plot(Ks,result1[i],'-or',label='Item')
    plt.plot(Ks,result2[i],'-^b',label='ItemIUF')
    plt.plot(Ks,result3[i],'-sk',label='ItemCF-norm')
    plt.legend()
    plt.show()

'''
# result保存不同参数的实验结果，用于作图
result_cf_exp=[]
result_iuf_exp=[]
result_norm_exp=[]
# 实验折数M、推荐列表长度N、邻域K
M, Ns, K = 8, [5,10,15,20,25], 40
for N in Ns:
    cf_exp = Algorithms02.Experiment(M, K, N, rt='ItemCF')
    metrics = cf_exp.run()
    result_cf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])
    
    iuf_exp = Algorithms02.Experiment(M, K, N, rt='ItemIUF')
    metrics= iuf_exp.run()
    result_iuf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

    norm_exp = Algorithms02.Experiment(M, K, N, rt='ItemCF-Norm')
    metrics = norm_exp.run()
    result_norm_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

result1=list(zip(*result_cf_exp))
result2=list(zip(*result_iuf_exp))
result3=list(zip(*result_norm_exp))
graph_name=['precision对比结果','recall对比结果','f1对比结果','coverage对比结果','popularity对比结果']
y_name=['precision','recall','f1','coverage','popularity']
for i in range(5):
    plt.figure()
    plt.title(graph_name[i]) 
    plt.xlabel("推荐长度topN")
    plt.ylabel(y_name[i])
    plt.plot(Ns,result1[i],'-or',label='ItemCF')
    plt.plot(Ns,result2[i],'-^b',label='ItemIUF')
    plt.plot(Ns,result3[i],'-sk',label='ItemCF-Norm')
    plt.legend()
    plt.show()
'''
