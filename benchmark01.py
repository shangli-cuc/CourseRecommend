from pyecharts.charts import Bar, Pie, Line, Page
from pyecharts import options as opts
import Algorithms01
import matplotlib.pyplot as plt

'''
def benchmark(result):
    precisionTemp,recallTemp,f1Temp,coverageTemp,popularityTemp=zip(*result)

    linePrecision.add_xaxis(Ks)
    linePrecision.add_yaxis("precision",precisionTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    linePrecision.set_global_opts(
            title_opts=opts.TitleOpts(title="算法precision对比"),
            xaxis_opts=opts.AxisOpts(name="topK"),
            yaxis_opts=opts.AxisOpts(name="precision"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )

    lineRecall.add_xaxis(Ks)
    lineRecall.add_yaxis("recall",recallTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    lineRecall.set_global_opts(
            title_opts=opts.TitleOpts(title="算法recall对比"),
            xaxis_opts=opts.AxisOpts(name="topK"),
            yaxis_opts=opts.AxisOpts(name="recall"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )

    lineF1.add_xaxis(Ks)
    lineF1.add_yaxis("f1",f1Temp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    lineF1.set_global_opts(
            title_opts=opts.TitleOpts(title="算法F1对比"),
            xaxis_opts=opts.AxisOpts(name="topK"),
            yaxis_opts=opts.AxisOpts(name="F1"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )
        
    lineCoverage.add_xaxis(Ks)
    lineCoverage.add_yaxis("coverage",coverageTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    lineCoverage.set_global_opts(
            title_opts=opts.TitleOpts(title="算法coverage对比"),
            xaxis_opts=opts.AxisOpts(name="topK"),
            yaxis_opts=opts.AxisOpts(name="coverage"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )

    linePopularity.add_xaxis(Ks)
    linePopularity.add_yaxis("popularity",coverageTemp,is_smooth=True,label_opts=opts.LabelOpts(is_show=False))
    linePopularity.set_global_opts(
            title_opts=opts.TitleOpts(title="算法popularity对比"),
            xaxis_opts=opts.AxisOpts(name="topK"),
            yaxis_opts=opts.AxisOpts(name="popularity"),
            legend_opts=opts.LegendOpts(
                type_="scroll", pos_left="80%", orient="vertical"
            ),
            datazoom_opts=[opts.DataZoomOpts(orient="vertical",yaxis_index=0), opts.DataZoomOpts("horizontal",xaxis_index=0), opts.DataZoomOpts(type_="inside")]
        )
page = Page()
linePrecision=Line()
lineRecall=Line()
lineF1=Line()
lineCoverage=Line()
linePopularity=Line()
page.add(linePrecision,lineRecall,lineF1,lineCoverage,linePopularity)
page.render("对比.html")

'''

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# result保存不同参数的实验结果，用于作图
result_cf_exp=[]
result_iuf_exp=[]
result_norm_exp=[]
# 实验折数M、推荐列表长度N、邻域K
M, N, Ks = 8, 10, [5, 10, 20, 40, 80, 160]
for K in Ks:
    cf_exp = Algorithms01.Experiment(M, K, N, rt='ItemCF')
    metrics = cf_exp.run()
    result_cf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])
    
    iuf_exp = Algorithms01.Experiment(M, K, N, rt='ItemIUF')
    metrics = iuf_exp.run()
    result_iuf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

    norm_exp = Algorithms01.Experiment(M, K, N, rt='ItemCF-Norm')
    metrics = norm_exp.run()
    result_norm_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])
    
# labels=['ItemCF','ItemIUF','ItemCF-Norm']
# styles=['-ob','-^r','-sk']
# result=[result_cf_exp,result_iuf_exp,result_norm_exp]

result1=list(zip(*result_cf_exp))
result2=list(zip(*result_iuf_exp))
result3=list(zip(*result_norm_exp))
graph_name=['precision对比结果','recall对比结果','f1对比结果','coverage对比结果','popularity对比结果']
y_name=['precision','recall','f1','coverage','popularity']
for i in range(5):
    plt.figure()
    plt.title(graph_name[i]) 
    plt.xlabel("邻域K")
    plt.ylabel(y_name[i])
    plt.plot(Ks,result1[i],'-or',label='ItemCF')
    plt.plot(Ks,result2[i],'-^b',label='ItemIUF')
    plt.plot(Ks,result3[i],'-sk',label='ItemCF-Norm')
    plt.legend()
    plt.show()


# result保存不同参数的实验结果，用于作图
result_cf_exp=[]
result_iuf_exp=[]
result_norm_exp=[]
# 实验折数M、推荐列表长度N、邻域K
M, Ns, K = 8, [5,10,15,20,25], 40
for N in Ns:
    cf_exp = Algorithms01.Experiment(M, K, N, rt='ItemCF')
    metrics = cf_exp.run()
    result_cf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])
    
    iuf_exp = Algorithms01.Experiment(M, K, N, rt='ItemIUF')
    metrics= iuf_exp.run()
    result_iuf_exp.append([metrics['Precision'],metrics['Recall'],metrics['F1'],metrics['Coverage'],metrics['Popularity']])

    norm_exp = Algorithms01.Experiment(M, K, N, rt='ItemCF-Norm')
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

