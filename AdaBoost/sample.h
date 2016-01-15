#ifndef SAMPLE_H
#define SAMPLE_H

#include<vector>

using namespace std;
const double e = 2.7182818284;

template <typename T>
struct sample
{//样本
	vector<T> x;//样本点
	T y;//样本值
	sample(vector<T> &_x, T _y) :x(move(_x)), y(_y){}
	void print()const
	{
		cout << '(';
		for (int i = 0; i != x.size(); ++i)
		{
			cout << x[i];
			if (i != x.size() - 1) cout << ',';
		}
		cout << ") " << y << endl;
	}
};


#endif