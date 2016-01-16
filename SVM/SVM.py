
###########################################
 # Author: bravepam
 #
 # E-mail:1372120340@qq.com
###########################################
 
 # -*-coding:utf-8 -*-

from numpy import *
import random
import matplotlib.pyplot as plt

#########支持向量机，实现算法是SMO(Sequential Minimal Optimization),包括简化版和完整版################
#########简化版SMO实现支持向量机，速度比较慢

def loadData(filename,feat_size): #加载数据集
    data = [];datacls = []
    with open(filename,'r') as fp:
        for line in fp:
            line = line.strip().split('\t')
            temp = []
            for i in range(feat_size):
                temp.append(float(line[i]))
            data.append(temp)
            if float(line[feat_size]) == 0.0:
                datacls.append(-1.0)
            else: datacls.append(float(line[feat_size])) #样本类别
    return data,datacls

def plotBestFitLine(dataset,dataset_cls,weights,svs): #画出数据点和支持向量
    data_mat = mat(dataset)
    m = shape(data_mat)[0]
    x0 = [];y0 = [] #第一类数据的横纵坐标
    x1 = [];y1 = []
    for i in range(m): #分出每个点
        if int(dataset_cls[i]) == 1:
            x0.append(float(data_mat[i,0]))
            y0.append(float(data_mat[i,1]))
        else:
            x1.append(float(data_mat[i,0]))
            y1.append(float(data_mat[i,1]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1,s = 40,c = 'red',marker = 's') #第二类数据，红色，正方形
    ax.scatter(x0,y0,s = 40,c = 'green') #第一类数据绿色，默认为圆形
    from matplotlib.patches import Circle
    svs_num = shape(svs)[0]
    for j in range(svs_num): #画出支持向量
    	cir = Circle(xy = (svs[j,0],svs[j,1]),radius = 0.05,alpha = 0.5)
    	ax.add_patch(cir)
    plt.axis("equal")
    plt.axis("scaled")
    # x = arange(-4.0,10.0,0.1)
    # y = (-weights[0] - weights[1] * x) / weights[2] #画出分隔直线
    # ax.plot(x,y.transpose())
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def randomSelectJ(i,m): #在选定alpha_i的情况下，随机选择另外一个alpha_j，总数为m个
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

def clipAlpha(alpha,L,H): #裁剪alpha，以满足KKT约束条件
    if alpha < L:
        alpha = L
    if alpha > H:
        alpha = H
    return alpha

def simpleSMO(data,datacls,toler,C,max_iter): #简化版的SMO算法
    data_mat = mat(data);datacls_mat = mat(datacls).transpose() #将数据集全部转化Numpy矩阵类型
    m,n = shape(data_mat)
    alpha = mat(zeros((m,1)));b = 0 #所求的模型参数
    iter = 0 #记录到目前为止，依然没有改变任何alpha对的迭代次数
    while (iter < max_iter): #当循环max_iter次后，还没有任何alpha被改变，则退出循环
        alpha_pair_changed = 0 #记录是否有alpha被改变
        for i in range(m):
            gxi = float(multiply(alpha,datacls_mat).T * (data_mat * data_mat[i,:].T)) + b
            Ei = gxi - datacls_mat[i] #计算在当前模型下，预测值和真实值之间的差异
            #若不满足KKT条件，这里的toler是容错率，在0附近则表示满足KKT，超过该范围，则可能不满足
            if(datacls_mat[i] * Ei < -toler and alpha[i] < C) or (datacls_mat[i] * Ei > toler and alpha[i] > 0):
                j = randomSelectJ(i,m) #随机选择另外一个alpha_j
                gxj = float(multiply(alpha,datacls_mat).T * (data_mat * data_mat[j,:].T)) + b
                Ej = gxj - datacls_mat[j] #同理，计算误差
                if datacls_mat[i] != datacls_mat[j]: #根据两者所属类别的异同决定alpha_j的范围，因为必须满足KKT条件
                    L = max(0,alpha[j] - alpha[i])
                    H = min(C,C + alpha[j] - alpha[i])
                else:
                    L = max(0,alpha[j] + alpha[i] - C)
                    H = min(C,alpha[j] + alpha[i])
                if L == H: #若范围上下限一样，则当前alpha不会有改变
                    print("L == H (%f)" % L)
                    continue #则跳过该次，直接开始下一次
                eta = data_mat[i,:] * data_mat[i,:].T + data_mat[j,:] * data_mat[j,:].T - 2.0 * data_mat[i,:] * data_mat[j,:].T
                if eta <= 0:
                    print("eta(%f) <= 0" % eta)
                    continue
                alphaIold = alpha[i].copy();alphaJold = alpha[j].copy() #保存以前的alpha
                alpha[j] = alphaJold + datacls_mat[j] * (Ei - Ej) / eta #计算alpha_j
                alpha[j] = clipAlpha(alpha[j],L,H) #裁剪alpha_j
                if abs(alphaJold - alpha[j]) < 0.00001: #若alpha_j没什么改变
                    print("j not move enough")
                    continue #则直接开始下一次
                alpha[i] = alphaIold + datacls_mat[i] * datacls_mat[j] * (alphaJold - alpha[j]) #计算alpha_i
                #根据alpha_i和alpha_j分别可以计算出一个b(1,2)
                b1 = -Ei - datacls_mat[i] * data_mat[i,:] * data_mat[i,:].T * (alpha[i] - alphaIold) - \
                datacls_mat[j] * data_mat[j,:] * data_mat[i,:].T * (alpha[j] - alphaJold) + b
                b2 = -Ej - datacls_mat[i] * data_mat[i,:] * data_mat[j,:].T * (alpha[i] - alphaIold) - \
                datacls_mat[j] * data_mat[j,:] * data_mat[j,:].T * (alpha[j] - alphaJold) + b
                if (alpha[i] > 0) and (alpha[i] < C): #若两个alpha均满足KKT条件
                    b = b1
                elif (alpha[j] > 0) and (alpha[j] < C): #则b1 == b2,
                    b = b2
                else: #如果有至少一个不满足，即为0或者C
                    b = (b1 + b2) / 2.0 #则区间[b1,b2]中的任何一个值都满足KKT条件，选择中点即可
                alpha_pair_changed += 1 #在该次循环遍历数据集下alpha对的改变数
                print("iter: %d i: %d, pairs changed %d" % (iter,i,alpha_pair_changed))
        if alpha_pair_changed == 0: iter += 1 #遍历了整个数据集没有改变任何一对alpha
        else: iter = 0 #否则，iter置0
        print("iteration number:%d" % iter)
    return alpha,b

def calcWeightsAndGetSVs(data,datacls,alpha): #根据SMO算法得到的alpha计算特征系数以及支持向量
    data_mat = mat(data);datacls_mat = mat(datacls).transpose()
    n = shape(data_mat)[1]
    SVindex = nonzero(alpha.A > 0.0)[0] #获得非零alpha值的编号列表
    #根据该列表依次得到非零alpha值，支持向量，支持向量类别
    alpha_not0 = alpha[SVindex];SVs = data_mat[SVindex];SVlabels = datacls_mat[SVindex]
    m = shape(SVs)[0]
    weights = zeros((n,1))
    for i in range(m):
        weights += multiply(alpha_not0[i] * SVlabels[i],SVs[i,:].T)
    return weights,SVs

# if __name__ == '__main__': # 简化版SMO main函数
#    data,datacls = loadData("testSet.txt",2)
#    alpha,b = simpleSMO(data,datacls,0.001,0.6,40) #调用简化SMO算法计算alpha和b
#    alpha = calcWeightsAndGetSVs(data,datacls,alpha) #计算出特征系数
#    weights = [b[0,0]]
#    m = shape(alpha)[0]
#    for i in range(m):
#        weights.append(alpha[i,0])
#    print(weights)
#    plotBestFitLine(data,datacls,weights) #画出整个数据集和分隔超平面

###############完整版的SMO算法，速度要快很多，因为对于alpha_j的选择做了很多优化

def kernel(data_mat,sample,ker): #计算核函数值
	m = shape(data_mat)[0]
	K = mat(zeros((m,1)))
	if ker[0] == 'line': #如果为线性，即没有用核函数，或者说用了将n维特征映射到n维空间的核函数
		K = data_mat * sample.T
	elif ker[0] == 'rbf': #高斯径向基函数
		for i in range(m): #计算所有样本点对sample样本的核函数值
			delta = data_mat[i,:] - sample
			K[i] = delta * delta.T
		K = exp(K / (-1.0 * ker[1] ** 2))
	else:
		raise NameError("The kernel is not recognized!")
	return K


class SMOData(object): #包装所有所需的数据，便于使用
    def __init__(self,data,datacls,C,toler,ker):
        self.data = mat(data)
        self.datacls = mat(datacls).transpose()
        self.C = C
        self.toler = toler
        self.m = shape(self.data)[0]
        self.alpha = mat(zeros((self.m,1)))
        self.b = 0
        self.E_cache = mat(zeros((self.m,2))) #记录所有样本的预测误差
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
        	self.K[:,i] = kernel(self.data,self.data[i,:],ker)

def calcEk(smo,k): #在当前模型下计算样本k的预测误差
    #gxk = float(multiply(smo.alpha,smo.datacls).T * (smo.data * smo.data[k,:].T)) + smo.b
    gxk = float(multiply(smo.alpha,smo.datacls).T * smo.K[:,k]) + smo.b
    Ek = gxk - smo.datacls[k]
    return Ek

def selectJ(smo,i,Ei): #在选择了alpha_i的情况下，选择alpha_j
    Ej = 0.0;maxDeltaE = 0.0;j = -1
    smo.E_cache[i] = [1,Ei] #首先更新Ei
    valid_ecache_list = nonzero(smo.E_cache[:,0].A)[0] #选出所有有效的E构成一个由其序号组成的列表
    if len(valid_ecache_list) > 1: #如果有多个有效的alpha
        for k in valid_ecache_list: #逐一迭代
            if k == i: continue
            Ek = calcEk(smo,k) 
            smo.E_cache[k] = [1,Ek] #更新Ek
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE: #寻找和Ei差别最大的alpha，因为这样的话alpha对更新最快
                maxDeltaE = deltaE;j = k;Ej = Ek
    else:#如果现在是迭代开始，此时只有一个有效的E，即Ei
        j = randomSelectJ(i,smo.m) #则随机选择一个alpha
        Ej = calcEk(smo,j)
        smo.E_cache[j] = [1,Ej]
    return j,Ej

def updateEk(smo,k): #更新Ek
    Ek = calcEk(smo,k)
    smo.E_cache[k] = [1,Ek]

def innerLoop(smo,i): #内层循环，即选择了alpha_i之后，选择alpha_j以及相关计算
    Ei = calcEk(smo,i)
   #若不满足KKT条件，toler是容错率，在0附近则表示满足KKT，超过该范围，则可能不满足
    if((smo.datacls[i] * Ei < -smo.toler and smo.alpha[i] < smo.C) or (smo.datacls[i] * Ei > smo.toler and smo.alpha[i] > 0)):
        j,Ej = selectJ(smo,i,Ei) #选择一个alpha_j，不再是随机选择了，以下同简化版
        if smo.datacls[i] != smo.datacls[j]: 
            L = max(0,smo.alpha[j] - smo.alpha[i])
            H = min(smo.C,smo.C + smo.alpha[j] - smo.alpha[i])
        else:
            L = max(0,smo.alpha[j] + smo.alpha[i] - smo.C)
            H = min(smo.C,smo.alpha[j] + smo.alpha[i])
        if L == H:
            print("L == H(%f)" % H)
            return 0 
        alphaIold = smo.alpha[i].copy();alphaJold = smo.alpha[j].copy()
        #eta = smo.data[i,:] * smo.data[i,:].T + smo.data[j,:] * smo.data[j,:].T - 2.0 * smo.data[i,:] * smo.data[j,:].T
        eta = smo.K[i,i] + smo.K[j,j] - 2.0 * smo.K[i,j]
        if eta <= 0:
            print("eta(%f) <= 0" % eta)
            return 0
        smo.alpha[j] = alphaJold + smo.datacls[j] * (Ei - Ej) / eta
        smo.alpha[j] = clipAlpha(smo.alpha[j],L,H)
        if abs(alphaJold - smo.alpha[j]) < 0.00001:
            print("j not move enough")
            return 0
        smo.alpha[i] = alphaIold + smo.datacls[i] * smo.datacls[j] * (alphaJold - smo.alpha[j])
        #没有采用核函数
        # b1 = smo.b - Ei - smo.datacls[i] * (smo.data[i,:] * smo.data[i,:].T) * (smo.alpha[i] - alphaIold) - \
        # smo.datacls[j] * (smo.data[j,:] * smo.data[i,:].T) * (smo.alpha[j] - alphaJold)
        # b2 = smo.b - Ej - smo.datacls[i] * (smo.data[i,:] * smo.data[j,:].T) * (smo.alpha[i] - alphaIold) - \
        # smo.datacls[j] * (smo.data[j,:] * smo.data[j,:].T) * (smo.alpha[j] - alphaJold)
        #采用核函数后
        b1 = smo.b - Ei - smo.datacls[i] * smo.K[i,i] * (smo.alpha[i] - alphaIold) - \
        smo.datacls[j] * smo.K[j,i] * (smo.alpha[j] - alphaJold)
        b2 = smo.b - Ej - smo.datacls[i] * smo.K[i,j] * (smo.alpha[i] - alphaIold) - \
        smo.datacls[j] * smo.K[j,j] * (smo.alpha[j] - alphaJold)
        if (0 < smo.alpha[i] < smo.C): smo.b = b1
        elif (0 < smo.alpha[j] < smo.C): smo.b = b2
        else: smo.b = (b1 + b2) / 2.0
        updateEk(smo,i) #更新Ei和Ej
        updateEk(smo,j)
        return 1 #如果成功进行优化，则返回1，这个用来计数，其实返回bool不计数也可以
    return 0 #否则

def completeSMO(data,datacls,C,toler,max_iter,ker = ("line",0.0)): #完整版的SMO
    smo = SMOData(data,datacls,C,toler,ker) #构造结构体
    iter = 0
    #前者表示是否搜寻整个数据集以寻找第一个alpha，后者计数每一次的遍历优化了几对alpha
    search_entire_set = True;alpha_pair_changed = 0
    #1、这里的max_iter和简化版不一样，和平常一样，表示最大可遍历次数，而不管每次是否都有alpha对被优化
    #2、退出条件是，达到最大遍历次数，或者遍历了整个数据集依然没有任何alpha对可优化，表示已经收敛
    while (iter < max_iter) and ((alpha_pair_changed > 0) or (search_entire_set)):
        alpha_pair_changed = 0
        if search_entire_set: #如果所有样本点都满足KKT条件，则表示要搜寻整个数据集
            for i in range(smo.m): #选择alpha_i
                alpha_pair_changed += innerLoop(smo,i) #累加优化的alpha对数
                print("full set,iter: %d ,i: %d, pairs changed: %d" % (iter,i,alpha_pair_changed))
        else: #否则如果存在不满足KKT条件的样本点
            break_KKT_alphas = nonzero((smo.alpha.A > 0) * (smo.alpha.A < smo.C))[0] #筛选出这样的样本点
            for i in break_KKT_alphas: #在这样的样本点里面选择一个
                alpha_pair_changed += innerLoop(smo,i)
                print("KKT,iter: %d,i: %d,pairs changed: %d" % (iter,i,alpha_pair_changed))
        iter += 1
        if search_entire_set: search_entire_set = False #如果之前搜寻的是整个数据集，那么下次不再如此
        elif (alpha_pair_changed == 0): #否则如果搜寻的是不满足KKT的样本，且没有任何alpha对被优化
            search_entire_set = True #则表示要扩大搜寻范围了
        print("iteration number: %d" % iter)
    return smo.alpha,smo.b

# if __name__ == '__main__': #完整版SMOmain函数
#     data,datacls = loadData("testSet.txt",2)
#     alpha,b = completeSMO(data,datacls,0.6,0.001,40)
#     weights,svs = calcWeightsAndGetSVs(data,datacls,alpha)
#     weightslst = [b[0,0]]
#     m = shape(weights)[0]
#     for i in range(m):
#         weightslst.append(weights[i,0])
#     print(weightslst)
#     print(svs)
#     plotBestFitLine(data,datacls,weightslst,svs)

def testSVM(ker = ('rbf',1.3)): #使用核函数对线性不可分数据进行分类的SVM测试函数
	data,datacls = loadData("testSetRBF.txt",2)
	alpha,b = completeSMO(data,datacls,200,0.0001,40,ker) #训练SVM
	data_mat = mat(data);datacls_mat = mat(datacls).transpose()
	svs_index = nonzero(alpha.A > 0.0)[0]
	svs = data_mat[svs_index];svs_cls = datacls_mat[svs_index]
	print("The number of support vectors is: %d" % shape(svs)[0])
	m,n = shape(data_mat)
	error = 0.0
	for i in range(m): #在训练集上测试
		K = kernel(svs,data_mat[i,:],ker)
		predict = multiply(alpha[svs_index],svs_cls).T * K + b
		if sign(predict) != sign(datacls_mat[i]):
			error += 1.0
	print("The error rate of training set is:%f" % (error / m))
	plotBestFitLine(data,datacls,None,svs) #画出数据点和支持向量，由于不是线性，故不存在分隔直线
	
	error = 0.0
	data,datacls = loadData("testSetRBF2.txt",2)
	data_mat = mat(data);datacls_mat = mat(datacls).transpose()
	m,n = shape(data_mat)
	for i in range(m): #在测试集上测试
		K = kernel(svs,data_mat[i,:],ker)
		predict = multiply(alpha[svs_index],svs_cls).T * K + b
		if sign(predict) != sign(datacls_mat[i]):
			error += 1.0
	print("The error rate of test set is:%f" % (error / m))

# if __name__ == '__main__':
# 	testSVM()

###############用SVM改进第二章的手写数字识别系统#######################

from os import listdir

def img2Vec(imagename): #将文本形式的图片内容转化向量
	vec = zeros((1,1024))
	with open(imagename,'r') as fp:
		i = 0
		for line in fp:
			for j in range(32):
				vec[0,32 * i + j] = int(line[j])
			i += 1
	return vec

def loadImages(dirname): #加载目录下面的图片文本
	imageslist = listdir(dirname)
	m = len(imageslist)
	data = zeros((m,1024));datacls = []
	for i in range(m):
		label = int(imageslist[i].split('_')[0])
		if label == 9: datacls.append(-1)
		else: datacls.append(1)
		data[i,:] = img2Vec("%s/%s" % (dirname,imageslist[i]))
	return data,datacls

def testHandwritingSVM(trainfile,testfile,ker = ('rbf',20)): #训练并测试SVM，和testSVM类似
	data,datacls = loadImages(trainfile)
	data_mat = mat(data);datacls_mat = mat(datacls).transpose()
	alpha,b = completeSMO(data,datacls,200,0.0001,10000,ker)
	svs_index = nonzero(alpha.A > 0.0)[0]
	svs = data_mat[svs_index];svs_cls = datacls_mat[svs_index]
	print("The number of support vectors is:%d" % shape(svs)[0])
	m,n = shape(data_mat)
	error = 0.0
	for i in range(m):
		K = kernel(svs,data_mat[i,:],ker)
		predict = multiply(alpha[svs_index],svs_cls).T * K + b
		if sign(predict) != sign(datacls_mat[i]):
			error += 1.0
	print("The error rate of training set is:%f" % (error / m))
	data,datacls = loadImages(testfile)
	data_mat = mat(data);datacls_mat = mat(datacls).transpose()
	m,n = shape(data_mat)
	error = 0.0
	for i in range(m):
		K = kernel(svs,data_mat[i,:],ker)
		predict = multiply(alpha[svs_index],svs_cls).T * K + b
		if sign(predict) != sign(datacls_mat[i]):
			error += 1.0
	print("The error rate of testing set is:%f" % (error / m))

# if __name__ == '__main__':
# 	print("start...")
# 	testHandwritingSVM("trainingDigits","testDigits",2) #在手写数字识别中使用SVM
    
#####################马患了疝气病之后，预测是否会死#################
#由于之前采用Logistic回归发现训练误差和预测误差都很大，说明很有可能
#欠拟合，即模型太简单（特征值还不够多，有21个），发生了高偏差低方差。
#数据无法得到改善，故采用SVM算法再次测试一下

def testHorseColicSVM(trainfile,testfile,feat_size,ker = ('rbf',20)): #训练并测试SVM，和testHandwritingSVM类似
    data,datacls = loadData(trainfile,feat_size)
    data_mat = mat(data);datacls_mat = mat(datacls).T
    alpha,b = completeSMO(data,datacls,230,0.0001,1000,ker)
    svs_index = nonzero(alpha.A > 0.0)[0]
    svs = data_mat[svs_index];svs_cls = datacls_mat[svs_index]
    print("The number of support vectors is:%d" % shape(svs)[0])
    m,n = shape(data_mat)
    error = 0.0
    for i in range(m):
        K = kernel(svs,data_mat[i,:],ker)
        predict = multiply(alpha[svs_index],svs_cls).T * K + b
        if sign(predict) != sign(datacls_mat[i]):
            error += 1.0
    print("The error rate of training set is:%f" % (error / m))

    data,datacls = loadData(testfile,feat_size)
    data_mat = mat(data);datacls_mat = mat(datacls).transpose()
    m,n = shape(data_mat)
    error = 0.0
    for i in range(m):
        K = kernel(svs,data_mat[i,:],ker)
        predict = multiply(alpha[svs_index],svs_cls).T * K + b
        if sign(predict) != sign(datacls_mat[i]):
            error += 1.0
    print("The error rate of testing set is:%f" % (error / m))

if __name__ == '__main__':
   testHorseColicSVM("horseColicTraining.txt","horseColicTest.txt",21,('rbf',90))
        