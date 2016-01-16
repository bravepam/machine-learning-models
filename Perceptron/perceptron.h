
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#ifndef PERCEPTRON_H
#define PERCEPTRON_H

//统计学习方法第二章————感知机（Perceptron）原始形式实现

#include<iostream>
#include<fstream>
#include<vector>
#include<string>

using namespace std;

class perceptron
{//感知机
private:
	float learningrate;
	vector<vector<float>> points;//输入点
	vector<float> tags;//每个点的取值
	int dim;//点维度
public:
	perceptron(float lr, int n) :learningrate(lr), dim(n){}
	void initData(const string &filename);
	void compute(vector<float>&, float&);
};

void perceptron::initData(const string &filename)
{
	ifstream infile(filename);
	if (!infile)
	{
		cout << "open " << filename << " error" << endl;
		return;
	}
	int i = 0;
	float v;
	vector<float> temp(dim);
	while (infile >> v)
	{
		temp[i++] = v;
		if (i == dim)
		{
			points.push_back(temp);
			infile >> v;
			tags.push_back(v);
			i = 0;
		}
	}
}

void perceptron::compute(vector<float> &w, float &b)
{//w是未知数系数向量，b是偏置量
	//目标，最小化：L(w,b) = -$[i = 1 to N] (yi(w*xi + b))
	//w <- w + lr * yi * xi
	//b <- b + lr * yi
	while (true)
	{
		int cnt = 0;
		for (int i = 0; i != points.size(); ++i)
		{
			float dot_pro = 0.0;
			for (int j = 0; j != dim; ++j)
				dot_pro += w[j] * points[i][j];
			if ((dot_pro + b) * tags[i] <= 0)
			{//存在误分类的点
				for (int k = 0; k != dim; ++k)
					w[k] += learningrate * tags[i] * points[i][k];
				b += learningrate * tags[i];
			}
			else ++cnt;//正确分类的点总数
		}
		if (cnt == points.size()) break;//若所有点都正确分类
	}
}

#endif