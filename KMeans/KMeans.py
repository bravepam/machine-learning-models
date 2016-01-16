# -*- coding:utf-8 -*-

###########################################
 # Author: bravepam
 #
 # E-mail:1372120340@qq.com
###########################################
 
from numpy import *
import matplotlib.pyplot as plt

#################K-Means算法####################

def loadData(filename): #加载数据
	data = []
	with open(filename,'r') as fp:
		for line in fp:
			line = line.strip().split('\t')
			temp = map(float,line) #map函数返回的是map对象
			data.append(list(temp))
	return mat(data)

def euclidDist(vec1,vec2): #向量的欧几里得距离
	return sqrt(sum(power(vec1 - vec2,2)))

def randCenter(datamat,k): #简单K-Means算法的初始中心初始化
	n = shape(datamat)[1]
	centers = mat(zeros((k,n)))
	for i in range(n):
		min_i = min(datamat[:,i]) #数据集中第i列的最小值
		range_i = float(max(datamat[:,i]) - min_i) #范围
		centers[:,i] = min_i + random.rand(k,1) * range_i #一次性初始化中心的第i列
	return centers

def KMeans(datamat,k,dist = euclidDist,createcent = randCenter): #简单K-means算法
	m = shape(datamat)[0]
	centers = createcent(datamat,k)
	clusters = mat(zeros((m,2))) #记载聚类结果，第0列记录样本属于哪个类别，第1列记录距离中心的距离的平方
	center_changed = True #记录是否有中心发生改变
	while center_changed: 
		center_changed = False
		#print(centers)
		for i in range(m):#对每个样本
			min_dist = inf
			for j in range(k): 
				d = dist(datamat[i,:],centers[j,:]) #计算其距每个中心的距离
				if d < min_dist: #如果该距离比当前的更小
					min_dist = d;min_index = j #则记录之
			if clusters[i,0] != min_index: center_changed = True #判断是否有聚类发生改变
			clusters[i,:] = min_index,min_dist ** 2 #将该样本分给距离最小的中心
		for c in range(k): #更新每个聚类的中心
			cluster_c = datamat[nonzero(clusters[:,0].A == c)[0]]
			centers[c,:] = mean(cluster_c,0)
	return centers,clusters

def plotClusters(centers,clusters,datamat): #绘制聚类结果
	k = shape(centers)[0]
	cluster_data = [];colors = []
	for i in range(k): #对每个聚类
		data = datamat[nonzero(clusters[:,0].A == i)[0]] #抽取它所包含的样本
		cluster_data.append(data)
		color = random.rand(3) #随机分配一个颜色给它
		colors.append(color)
	colors.append(random.rand(3)) #中心点的颜色
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(k): #对每个聚类
		ax.scatter(cluster_data[i][:,0],cluster_data[i][:,1],color = colors[i],linewidths = 4.0,alpha = 1.0) #绘制聚类中的点集
		ax.scatter(centers[i,0],centers[i,1],color = 'red',marker = '+',s = 300,linewidths = 4.0,alpha = 1.0) #中心点
	plt.show()

# if __name__ == '__main__': #简单K-Means算法测试函数
# 	datamat = loadData("testSet.txt")
# 	centers,clusters = KMeans(datamat,4)
# 	plotClusters(centers,clusters,datamat)
# 	print(centers)

def biKMeans(datamat,k,dist = euclidDist): #二分K-Means算法
	m = shape(datamat)[0]
	clusters = mat(zeros((m,2))) #记录聚类结果
	center0 = mean(datamat,0).tolist()[0] #最初，所有样本为一类时的聚类中心
	centerslist = [center0]
	for i in range(m):
		clusters[i,1] = dist(mat(center0),datamat[i,:]) ** 2 #最初的聚类结果
	while len(centerslist) < k: #只要类别数目没达到k个
		lowest_error = inf
		for j in range(len(centerslist)): #对每个聚类
			j_cluster = datamat[nonzero(clusters[:,0].A == j)[0],:] #抽取属于它的样本点
			j_2centers,j_2clusters = KMeans(j_cluster,2,dist) #对其进行2分聚类
			j_splited_error = sum(j_2clusters[:,1]) #计算该类别重新聚类后的误差和
			non_splited_error = sum(clusters[nonzero(clusters[:,0].A != j)[0],1]) #剩下的其他没有聚类的类别的误差和
			print("%d, split error:%f,non-split error:%f" % (j,j_splited_error,non_splited_error))
			if (j_splited_error + non_splited_error) < lowest_error: #如果两者的误差和比当前小
				lowest_error = j_splited_error + non_splited_error #则记录相关信息
				best_split_cluster = j;best_split_2centers = j_2centers.copy();best_split_2clusters = j_2clusters.copy()
		print("Best split cluster:%d" % best_split_cluster) #最佳重聚类类别标号
		#对重聚类的结果中标号1的类别分配新的类别标号
		best_split_2clusters[nonzero(best_split_2clusters[:,0].A == 1)[0],0] = len(centerslist)
		#对重聚类的结果中标号0的类别赋予重聚类之前的类别号
		best_split_2clusters[nonzero(best_split_2clusters[:,0].A == 0)[0],0] = best_split_cluster
		#筛选原始的聚类中属于最佳聚类类别的样本，并将它们更新为重聚类后的聚类结果
		clusters[nonzero(clusters[:,0].A == best_split_cluster)[0],:] = best_split_2clusters
		centerslist[best_split_cluster] = best_split_2centers[0,:].A[0] #更新原始被重聚类的类别中心
		centerslist.append(best_split_2centers[1,:].A[0]) #添加额外的一个聚类中心
	return mat(centerslist),clusters

if __name__ == '__main__': #2分K-Means算法测试函数
	datamat = loadData("testSet.txt")
	centers,clusters = biKMeans(datamat,4)
	print(centers)
	plotClusters(centers,clusters,datamat)

##################用K-Means算法对地理位置进行聚类，找出最佳聚会位置######################

def sphercicalDist(vec1,vec2): #地球球面距离计算，向量为经纬度
	a = sin(vec1[0,1] * pi / 180) * sin(vec2[0,1] * pi / 180)
	b = cos(vec1[0,1] * pi / 180) * cos(vec2[0,1] * pi / 180) * cos(pi * (vec2[0,0] - vec1[0,0]) / 180)
	return arccos(a + b) * 6371.0

def clusterClubs(iter = 5): #聚类函数
	places = []
	with open("places.txt",'r') as fp: #初始化数据
		for line in fp:
			line = line.strip().split('\t')
			places.append([float(line[4]),float(line[3])]) #经度和纬度
	placesmat = mat(places)
	centers,clusters = biKMeans(placesmat,iter,sphercicalDist) #进行二分K-Means聚类
	fig = plt.figure()
	rect = [0.1,0.1,0.8,0.8] #绘制图占整个窗口大小
	scattermarkers = ['s','o','^','8','p','d','v','h','>','<'] #聚类点形状
	axprops = dict(xticks = [],yticks = [])
	ax0 = fig.add_axes(rect,label = 'ax0',**axprops) #添加轴0，用于显示一幅图像
	img = plt.imread("Portland.png") 
	ax0.imshow(img)
	ax1 = fig.add_axes(rect,label = 'ax1',frameon = False) #添加轴1，和轴0的位置完全一样，用于显示点集
	for i in range(iter): #对每个聚类，绘制聚类结果
		i_cluster = placesmat[nonzero(clusters[:,0].A == i)[0],:]
		c = random.rand(4)
		ax1.scatter(i_cluster[:,0],i_cluster[:,1],color = c,s = 100,alpha = 1.0,marker = scattermarkers[i % len(scattermarkers)])
		ax1.scatter(centers[i,0],centers[i,1],marker = '+',s = 300,color = 'red',alpha = 1.0)
	plt.show()

# if __name__ == '__main__':
# 	clusterClubs()