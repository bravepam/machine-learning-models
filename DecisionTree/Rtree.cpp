#include<fstream>
#include<iostream>
#include<vector>
#include"LeastSquareRegressionTree.h"

using namespace std;

const int dim = 1;

int main()
{
	ifstream infile("rdata.txt");
	vector<sample<double>> data;
	vector<double> temp;
	double v;
	int cnt = 0;
	while (infile >> v)
	{
		temp.push_back(v);
		++cnt;
		if (cnt == dim)
		{
			infile >> v;
			data.push_back(sample<double>(temp, v));
			cnt = 0;
		}
	}
	Rtree rt(data, 2);
	rt.create();
	vector<double> vec = { 6.5 };
	cout << rt.compute(vec) << endl;
	getchar();
	return 0;
}