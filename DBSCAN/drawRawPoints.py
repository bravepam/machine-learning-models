# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadData(filename): #加载数据
	data = []
	with open(filename,'r') as fp:
		for line in fp:
			line = line.strip().split('\t')
			temp = map(float,line) #map函数返回的是map对象
			data.append(list(temp))
	return mat(data)

def plotClusters(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data[0][:,0],data[0][:,1],color = random.rand(3),linewidths = 4.0,alpha = 1.0) #绘制聚类中的点集
	plt.show()

if __name__ == '__main__':
 	data = loadData(".//cluster_data//Twomoons.txt") #绘制原始数据点，观察点分布
 	print(shape(data))
 	plotClusters([data])