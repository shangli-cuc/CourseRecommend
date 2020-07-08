import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

F1 = [
    [0.048205287452578086,
    0.044278701694659015,
    0.041174733481284254,
    0.038459391276122835,
    0.07504236189705935,
    0.07607241650056841,
    0.07020572077942863,
    0.0648383406809743,
    0.05820542620217785,
    0.05289674804251784],

    [0.0818833617282542,
    0.07452444066576575,
    0.06910508712947203,
    0.06435521916116163,
    0.060370262811004216,
    0.16800837708575217,
    0.1413322690477718,
    0.118945281105919,
    0.10346277313641665,
    0.09121367490020327],

    [0.0726924161954436,
    0.06559285751437843,
    0.06026531831954436,
    0.05581880130541548,
    0.052231756190511805,
    0.19295653207225044,
    0.144410971259370122,
    0.1141108600918357963,
    0.09530407782225869,
    0.0820741539806875],  

    [0.07592150296120993,
    0.0687026400339503,
    0.06290766260259695,
    0.05866570504874076,
    0.05512193795090736,
    0.1950427673343734,
    0.14677359105476062,
    0.11692283603530632,
    0.09846278164894717,
    0.08531462226320409]
    ]
Ks = [5, 10, 15, 20, 25, 30, 35,40, 45, 50]


topNs = [5, 10]

graph_name = ['F1对比结果']
y_name = ['F1']
label = ['-or', '-^b', '-sk', '-*y']
label = ['r', 'b', 'k', 'y']

item_name = ['ItemCF-jaccard', 'ItemCF-cosine', 'ItemCF-pearson', 'ItemCF-ips']

plt.figure()
plt.title('ItemCF F1对比结果')
plt.xlabel('推荐列表长度topN')
plt.ylabel('F1')
for i in range(len(F1)):
    plt.plot(Ks, F1[i], label[i], label=item_name[i])
plt.legend()
plt.show()
