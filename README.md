# CourseRecommend

## 目录问题：

数据未上传，将准备的数据放在和recommend文件夹同级目录下运行，可能需要单独调整目录，最笨的办法当然是写成绝对路径

## 文件说明：

ItemCF.py是独立实现的基于item和user的CF算法，已经自有的course数据集上跑通，更换数据集需要自行调整，后缀为版本分为01,02,03，各有优劣

Algorithm.py试图将几种算法整合起来，同时让通过编写类，让代码易读，benchmarkForAlgorithm.py是对应的Algorithm.py的对比图像输出代码

Graph文件夹存放作图代码，ItemCFGraph01、02是两代版本，最好以第二代为主，第一代可做参考，UserCFGraph同理，Graphics_cold和hot分别是冷热启动的结果对比图
