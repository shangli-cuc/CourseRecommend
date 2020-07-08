import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# coverage1 = [0.252919279013,0.202976392702,0.121173952462]
# coverage2 = [0.242118273918,0.121700147539,0.082178436073]
F1 = [0.117176891800734,
0.137176891800734,
0.130938275611038,
0.123381058392801,
0.120113947286011
]

    
name = ['ItemCF(IPS)','Hybrid','5(1/60)', '10(1/30)', '15(1/20)']
x = np.arange(len(name))
width = 0.25

plt.bar(x, F1,  width=0.5, label='F1',color='royalblue',tick_label=name)
# plt.bar(x, coverage1,  width=width, label='label1',color='red')
# plt.bar(x + width, coverage2, width=width, label='label2', color='green', tick_label=name)


# 显示在图形上的值
# for a, b in zip(x,F1):
#     plt.text(a, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x,coverage2):
#     plt.text(a+width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')

plt.xticks()
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('F1')
plt.xlabel('新项目数量(所占比例)')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
plt.title("F1对比结果")
# plt.savefig('D:\\result.png')
plt.show()


