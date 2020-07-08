import matplotlib.pyplot as plt
import UserCF

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

F1 = [0.189159616388397,
    0.207273072584044,
    0.204987763144747,
    0.192934982570702]

precision = [0.117176891800734,
    0.130938275611038,
    0.129381058392801,
    0.120113947286011
    ]

recall = [0.490442401518593,
    0.497037540129413,
    0.493200419113740,
    0.490013859275103]

user_topNs = [5, 10, 15, 20, 25, 30]
item_topNs = [5, 10, 15, 20, 25, 30, 35, 40]

graph_name = ['precision对比结果', 'recall对比结果', 'f1对比结果']
y_name = ['precision', 'recall', 'f1']
# label = ['-or', '-^b', '-sk']
label = ['r', 'b', 'k', 'y']

user_name = ['ItemCF(IPS)', 'UserCF-cosine', 'UserCF-pearson']
item_name = ['ItemCF-IUF', 'ItemCF-pearson', 'ItemCF-cosine', 'ItemCF-jaccard']

plt.figure()
plt.title('F1对比结果')
plt.xlabel('新项目数量(所占比例)')
plt.ylabel('F1')
for i in range(len(usercf_f1)):
    plt.plot(user_topNs, usercf_f1[i], label[i], label=user_name[i])
plt.legend()
plt.show()

plt.figure()
plt.title('ItemCF f1对比结果')
plt.xlabel('推荐列表长度topN')
plt.ylabel('f1')
for i in range(len(itemcf_f1)):
    plt.plot(item_topNs, itemcf_f1[i], label[i], label=item_name[i])
plt.legend()
plt.show()

'''
for i in range(3):
    plt.figure()
    plt.title(graph_name[i])
    plt.xlabel('推荐列表长度topN')
    plt.ylabel(y_name[i])
    for i in range(len(precision)):
        plt.plot(topNs, precision[i], label[i], label=name[i])
    plt.legend()
    plt.show()
'''
