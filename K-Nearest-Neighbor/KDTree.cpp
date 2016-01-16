
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<vector>
#include<fstream>
#include"kdtree.h"

using namespace std;

int main()
{
	vector<vector<float>> dots;
	vector<int> ctg;
	ifstream infile("kNN-samples.txt");
	int cnt = 0;
	float v;
	vector<float> dot(2);
	while (infile >> v)
	{
		dot[cnt++] = v;
		if (cnt == 2)
		{
			infile >> v;
			dots.push_back(dot);
			ctg.push_back((int)v);
			cnt = 0;
		}
	}
	KDTree kdt(2);
	kdt.create(dots, ctg);
	vector<float> s = { 8.1f, 4.1f };
	vector<float> near;
	float mindist = kdt.nearest(s, near);
	printf("%f\n(", mindist);
	for (int i = 0; i != near.size(); ++i)
	{
		printf("%f", near[i]);
		if (i != near.size() - 1)
			printf(",");
	}
	printf(")\n");
	/*vector<vector<float>> knn;
	vector<float> dist;
	kdt.kNN(s,3,knn,dist);
	for (int i = 0; i != knn.size(); ++i)
	{
		printf("(");
		for (int j = 0; j != knn[i].size(); ++j)
		{
			printf("%f", knn[i][j]);
			if (j != knn[i].size() - 1)
				printf(",");
		}
		printf(") %f\n",dist[i]);
	}*/
	getchar();
	return 0;
}