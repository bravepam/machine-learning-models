#include<vector>
#include<fstream>
#include"kdtree.h"

using namespace std;

int main()
{
	vector<vector<double>> dots;
	vector<int> ctg;
	ifstream infile("kNN-samples.txt");
	int cnt = 0;
	const int dim = 2;
	double v;
	vector<double> dot;
	while (infile >> v)
	{
		++cnt;
		dot.emplace_back(v);
		if (cnt == dim)
		{
			infile >> v;
			dots.emplace_back(move(dot));
			ctg.emplace_back((int)v);
			cnt = 0;
		}
	}
	KDTree kdt(2);
	kdt.create(dots, ctg);
	kdt.print();
	vector<double> s = { 8.1f, 4.1f };
	//test nearest
	/*vector<double> near;
	double mindist = kdt.nearest(s, near);
	printf("%lf\n(", mindist);
	for (int i = 0; i != near.size(); ++i)
	{
		printf("%lf", near[i]);
		if (i != near.size() - 1)
			printf(",");
	}
	printf(")\n");*/
	//test knn
	vector<vector<double>> knn;
	vector<double> dist;
	kdt.kNN(s,3,knn,dist);
	for (int i = 0; i != knn.size(); ++i)
	{
		printf("(");
		for (int j = 0; j != knn[i].size(); ++j)
		{
			printf("%lf", knn[i][j]);
			if (j != knn[i].size() - 1)
				printf(",");
		}
		printf(") %lf\n",dist[i]);
	}
	getchar();
	return 0;
}