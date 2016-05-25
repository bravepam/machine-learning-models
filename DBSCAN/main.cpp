#include"dbscan.h"
#include<iostream>
#include<fstream>

using namespace std;

void findBestKAndRadius()
{
	system("python drawRawPoints.py");//画出原始点分布图
	for (int k = 1;; ++k)
	{
		cout << "k = " << k << endl;
		dbscan db(k, 2);
		db.readPoints(".//cluster_data//four_clusters.txt");
		db.computeKdists();
		db.drawKdists();
	}
}

void clustering()
{
	ifstream infile("k-and-radius.txt");
	double r;
	size_t k;
	while (infile >> r >> k)
	{
		dbscan db(r, k, 2);
		db.readPoints(".//cluster_data//four_clusters.txt");
		db.clustering();
		cout << "radius = " << r << " k = " << k << " clusters'number: " << db.getClustersNum() << endl;
		db.writePoints("points-info-after-clustering.txt");
		db.drawClusters();
	}
}

int main()
{
	//findBestKAndRadius();//先运行该函数找出一些最佳搭配的半径和k值
	clustering();//在运行该函数观察在不同的最佳搭配下的聚类效果
	return 0;
}