# __author__ = 'png'
# -*- coding:utf-8 -*-

###########################################
 # Author: bravepam
 #
 # E-mail:1372120340@qq.com
###########################################
 
#############运用Logistic回归进行分类########################

from numpy import *
import matplotlib.pyplot as plt
import random

def loadData(filename,feat_size): #载入数据
    data = [];datacls = []
    with open(filename,'r') as fp:
        for line in fp:
            temp_data = [1.0] #第0维数据全部都是1
            linelst = line.strip().split('\t')
            for i in range(feat_size):
                temp_data.append(float(linelst[i]))
            datacls.append(int(linelst[feat_size])) #数据所属类别
            data.append(temp_data)
    return data,datacls

def sigmoid(inx): #Logistic函数
    return 1.0 / (1.0 + exp(-inx))

def gradientAscent(dataset,datacls,iter = 400): #梯度上升法求最优化模型参数，因为是最大化，所以称之为梯度上升
    data_mat = mat(dataset) #NumPy矩阵类型
    datacls_mat = mat(datacls).transpose()
    m,n = data_mat.shape
    alpha = 0.001 #学习率或者收敛步长
    weights = ones((n,1))
    for i in range(iter): #每一次迭代
        h = sigmoid(data_mat * weights) #计算所有数据在当前模型参数下的Logistic函数值
        error = datacls_mat - h #和真实值的误差
        weights = weights + alpha * data_mat.transpose() * error #更正模型参数
    return weights

def plotBestFitLine(dataset,dataset_cls,weights): #画出数据点
    data_mat = mat(dataset)
    m = shape(data_mat)[0]
    x0 = [];y0 = [] #第一类数据的横纵坐标
    x1 = [];y1 = []
    for i in range(m): #分出每个点
        if int(dataset_cls[i]) == 0:
            x0.append(float(data_mat[i,1]))
            y0.append(float(data_mat[i,2]))
        else:
            x1.append(float(data_mat[i,1]))
            y1.append(float(data_mat[i,2]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1,s = 40,c = 'red',marker = 's') #第二类数据，红色，正方形
    ax.scatter(x0,y0,s = 40,c = 'green') #第一类数据绿色，默认为圆形
    x = arange(-4.0,4.0,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2] #画出分隔直线
    ax.plot(x,y.transpose())
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stochasticGradientAscent(dataset,dataset_cls,iter = 20): #随机梯度上升法计算模型参数
    dataset = array(dataset) #NumPy数组，和矩阵是有区别的
    m,n = shape(dataset)
    weights = ones(n)
    for i in range(iter): #每一次迭代
        datalist = [x for x in range(m)] #从所有的样本中
        for j in range(m):
            alpha = 4.0 / (1.0 + i + j) + 0.01 #学习率，开始时较快，后来较慢
            index = int(random.uniform(0,len(datalist))) #随机选择一个
            h = sigmoid(sum(dataset[index] * weights)) #计算函数值
            error = dataset_cls[index] - h #误差
            weights += alpha * error * dataset[index] #根据该样本误差，更新模型参数，不再是根据所有样本了
            del (datalist[index])
    return weights

# if __name__ == '__main__':
#    dataset,dataset_cls = loadData("testSet.txt",2)
#    #weights = gradientAscent(dataset,dataset_cls)
#    weights = stochasticGradientAscent(dataset,dataset_cls)
#    print(weights)
#    plotBestFitLine(dataset,dataset_cls,weights)

############采用Logistic回归预测的疝气病的马是否会死亡#######################

def classify(weights,vec): #根据模型判别该样本属于哪一类
    h = sigmoid(sum(weights * vec))
    if h > 0.5: return 1.0 
    return 0.0

def horseColicTrain(trainfile,feat_size): #训练预测模型
    trainset = [];traincls = []
    with open(trainfile,'r') as fp: #解析数据
        for line in fp:
            data = []
            line = line.strip().split('\t')
            for i in range(feat_size):
                data.append(float(line[i]))
            trainset.append(data)
            traincls.append(float(line[feat_size]))
    return stochasticGradientAscent(trainset,traincls) #采用随机梯度上升法学习模型
    #return gradientAscent(trainset,traincls)

def horseColicTest(testfile,feat_size,weights): #测试该模型的正确率
    error,num = 0.0,0.0
    with open(testfile,'r') as fp: #解析测试数据
        for line in fp:
            num += 1.0
            data = []
            line = line.strip().split('\t')
            for i in range(feat_size):
                data.append(float(line[i]))
            if int(classify(weights,data)) != int(float(line[feat_size])): #若分类错误
                error += 1.0
    errorrate = error / num #错误率
    print("The error rate is:%f" % errorrate)
    return errorrate

def multiTest(trainfile,testfile,feat_size,testnum = 10): #多次测试取平均错误率
    sum_errorrate = 0.0
    for i in range(testnum):
          weights = horseColicTrain(trainfile,feat_size)
          sum_errorrate += horseColicTest(testfile,feat_size,weights)
    print("The average error rate is:%f" % (sum_errorrate / testnum))


if __name__ == '__main__':
    multiTest("horseColicTraining.txt","horseColicTraining.txt",21)