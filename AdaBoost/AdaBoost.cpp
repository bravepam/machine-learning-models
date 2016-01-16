
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<vector>
#include<iostream>
#include<fstream>
#include"AdaBoost.h"

using namespace std;

//int main()
//{
//	vector<sample<int>> data;
//	int val, cnt = 0;
//	vector<int> x;
//	ifstream infile("dvc_data.txt");
//	while (infile >> val)
//	{
//		if (val == -2) break;
//		x.push_back(val);
//		++cnt;
//		if (cnt == 3)
//		{
//			infile >> val;
//			data.push_back(sample<int>(x, val));
//			cnt = 0;
//		}
//	}
//	vector<int> fte_values = { 2, 3, 3 };
//	DicreteValueClassifier ada(data, fte_values, 0.1);
//	ada.train();
//	vector<int> t = { 1, 1, 0 };
//	cout << ada.decide(t) << endl;
//	getchar();
//	return 0;
//}

//int main()
//{
//	vector<sample<double>> data;
//	vector<double> x;
//	double val;
//	int cnt = 0;
//	ifstream infile("cvc_data.txt");
//	while (infile >> val)
//	{
//		x.push_back(val);
//		++cnt;
//		if (cnt == 1)
//		{
//			infile >> val;
//			data.push_back(sample<double>(x, val));
//			cnt = 0;
//		}
//	}
//	ContinuousValueClassifier cvc(data, 0.05);
//	cvc.train();
//	getchar();
//	return 0;
//}

int main()
{
	vector<sample<double>> data;
	vector<double> x;
	double val;
	int cnt = 0;
	ifstream infile("bt_data.txt");
	while (infile >> val)
	{
		x.push_back(val);
		++cnt;
		if (cnt == 1)
		{
			infile >> val;
			data.push_back(sample<double>(x, val));
			cnt = 0;
		}
	}
	BoostingTree bt(data, 0.2);
	bt.train();
	getchar();
	return 0;
}