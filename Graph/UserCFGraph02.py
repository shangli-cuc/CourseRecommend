import matplotlib.pyplot as plt
import UserCF

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

F1_K = [
[0.053069384211873,
0.059640116226238,
0.069892261347455,
0.075729627982255,
0.078479692616493,
0.082610411734724,
0.082618297398741,
0.082621348719247,
0.082612739874379],


[0.058014339888191,
0.074066242897248,
0.084630249743442,
0.090192639861212,
0.093932874438000,
0.095154461327523,
0.095119827041011,
0.095143248712321,
0.095141293707001],


[0.077971950880089,
0.091689725540121,
0.093460061036280,
0.095762139881356,
0.096889813619806,
0.098148328257975,
0.098121347891271,
0.098112348941230,
0.098192347897521],


[0.087362324326604,
0.092383792565511,
0.094471749230385,
0.097641736998843,
0.098013011508512,
0.100849123289858,
0.100823407017248,
0.100823491287951,
0.100891284986732]

]
Ks = [5, 10, 15, 20, 25, 30, 35, 40, 45]


graph_name = ['F1对比结果']
y_name = ['F1']
label = ['-^b', '-sk', '-*y', '-or']
# label = ['b', 'k', 'y', 'r']

item_name = ['UserCF-jaccard', 'UserCF-cosine', 'UserCF-pearson', 'UserCF-ips']

plt.figure()
plt.title('UserCF F1对比结果')
plt.xlabel('邻域K')
plt.ylabel('F1')
for i in range(len(F1_K)):
    plt.plot(Ks, F1_K[i], label[i], label=item_name[i])
plt.legend()
plt.show()

F1_topN = [
[0.065272863483135,
0.082625955967687,
0.087555896728334,
0.093813154906467,
0.099000340207657,
0.104401526191030,
0.094111234803707,
0.089463836780010,
0.077005575233455,
0.069715045419862],

[0.081089405539401,
0.095157768211230,
0.100852965228185,
0.104259259097149,
0.106066454326210,
0.110195189868068,
0.107221402795434,
0.105402993568488,
0.103584576098352,
0.099579143555225],

[0.078000801466487,
0.098151414430785,
0.099387220157571,
0.100903448561817,
0.103780515681014,
0.1073261461367310,
0.106428564176575,
0.104601427475033,
0.100920770300912,
0.099071450529520],


[0.090769087819622,
0.100849123289858,
0.103342269175203,
0.104843056285108,
0.107592301329407,
0.113099417485230,
0.107273103815806,
0.102924852926617,
0.095194171173704,
0.087366718694508]

]
topNs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.figure()
plt.title('UserCF F1对比结果')
plt.xlabel('推荐列表长度topN')
plt.ylabel('F1')
for i in range(len(F1_topN)):
    plt.plot(topNs, F1_topN[i], label[i], label=item_name[i])
plt.legend()
plt.show()
