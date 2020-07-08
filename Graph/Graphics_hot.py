import matplotlib.pyplot as plt
import UserCF

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

usercf_f1 = [ 
    [0.053069384211872704,
    0.08261041173472372,
    0.09893065466575321,
    0.09515446132752267,
    0.08462753911848638,
    0.07406451659455178],
    
    [0.056379710209824706,
    0.08793389017354881,
    0.10075326925150627,
    0.09463942542081201,
    0.08394428318446426,
    0.07338605040879906],

    [0.0652422945516139,
    0.0890384947043254,
    0.0938086809120257,
    0.08755129498236972,
    0.07884905953142485,
    0.07101540625467515]
    ]

itemcf_f1 = [
    [0.0557767573001324,
    0.05847708123487993,
    0.06754978533324547,
    0.06859432383432873,
    0.06923848974983525,
    0.06523487345797424,
    0.06323484350944325,
    0.0578004214323423], 

    [0.05117767974585299,
    0.05631535730573003,
    0.06655333593385518,
    0.06737135077098383,
    0.06748423615519414,
    0.06468169926010904,
    0.06114267708137535,
    0.056682751967657], 

    [0.04501158681470495,
    0.05039679833625611,
    0.05972450713524764,
    0.06048313644056762,
    0.061523358027002,
    0.059354376462211623,
    0.05651654616818489,
    0.05249066485768441], 

    [0.03354908256225092,
    0.0396281090009931,
    0.047314864809332495,
    0.05013648351986903,
    0.05101506675690633,
    0.0489416771619651,
    0.045740935432426785,
    0.04198366024891151]
]

user_topNs = [5, 10, 15, 20, 25, 30]
item_topNs = [5, 10, 15, 20, 25, 30, 35, 40]

graph_name = ['precision对比结果', 'recall对比结果', 'f1对比结果', 'coverage对比结果', 'popularity对比结果']
y_name = ['precision', 'recall', 'f1', 'coverage', 'popularity']
# label = ['-or', '-^b', '-sk']
label = ['r', 'b', 'k', 'y']

user_name = ['UserCF-jaccard', 'UserCF-cosine', 'UserCF-pearson']
item_name = ['ItemCF-IUF', 'ItemCF-pearson', 'ItemCF-cosine', 'ItemCF-jaccard']

plt.figure()
plt.title('UserCF f1对比结果')
plt.xlabel('推荐列表长度topN')
plt.ylabel('f1')
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
