# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadClusteringRes(filename): #加载聚类后的数据集
	data = [];
	with open(filename,'r') as fp:
		for n in fp:
			n = int(n)
			if(n == 0):
				data.append(mat([]))
				continue
			i = 0
			tempdata = [];
			for line in fp:
				line = line.strip().split('\t')
				temp = map(float,line) #map函数返回的是map对象
				tempdata.append(list(temp))
				i += 1
				if i == n:
					break;
			data.append(mat(tempdata))
	return data

def plotClusters(data): #绘制聚类结果
	k = len(data) - 2
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(k): #对每个聚类
		ax.scatter(data[i][:,0],data[i][:,1],color = random.rand(3),linewidths = 4.0,alpha = 1.0) #绘制聚类中的点集
	if shape(data[-2])[1] > 0: #画出边界点，形状为实心正三角形，事实上可以不画出边界点，因为它也属于某一簇
		ax.scatter(data[-2][:,0],data[-2][:,1],color = random.rand(3),marker = '^', linewidths = 6.0,alpha = 1.0)
	if shape(data[-1])[1] > 0: #画出噪声点，形状为实心钻石形，不属于任何簇
		ax.scatter(data[-1][:,0],data[-1][:,1],color = random.rand(3),marker = 'D', linewidths = 6.0,alpha = 1.0)
	plt.show()

if __name__ == '__main__':
	data = loadClusteringRes("clusters.txt") #绘制聚类后的结果，包括各簇以及边界点集合噪声点集
	plotClusters(data)