# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def plotMdists(dists): #绘制k-距离的变化，观察图形找出最佳的半径和k值
	n = len(dists)
	x = [i for i in range(n)]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x,dists)
	plt.show()

if __name__ == '__main__':
	dists = [list(map(float,line.strip().split('\t'))) for line in open("k-dists.txt","r")][0]
	plotMdists(dists)