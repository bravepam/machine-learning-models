
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<vector>
#include<fstream>
#include<iostream>
#include"NaiveBayes.h"

using namespace std;


int main()
{
	ifstream infile("NaiveBayes-samples.txt");
	vector<vector<int>> sample_points;
	vector<int> sample_values, temp(2);
	int v,cnt = 0;
	while (infile >> v)
	{
		if (v == -1) break;
		temp[cnt++] = v;
		if (cnt == 2)
		{
			sample_points.push_back(temp);
			infile >> v;
			sample_values.push_back(v);
			cnt = 0;
		}
	}
	NaiveBayes nb;
	vector<int> fv = { 3, 3 };
	nb.create(fv, 2, 1.0);
	nb.train(sample_points, sample_values);
	double p;
	vector<int> data = { 1, 0 };
	int k = nb.compute(data, p);
	cout << "Class: " << k << " Probability: " << p << endl;
	getchar();
	return 0;
}