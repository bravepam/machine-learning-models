#include<vector>
#include<iostream>

using namespace std;

const double E = 2.7182818284;
const double PI = 3.1415926;

class GMM
{//高斯混合模型
private:
	struct GM
	{//高斯模型
		double weight;
		double expectation;
		double variance;
		GM() :weight(0.), expectation(0.), variance(1.0){}
		GM(double w, double e, double v) :weight(w), expectation(e), variance(v){}
		double compute(double y)const
		{//计算加权概率
			double temp1 = weight / (sqrt(2 * PI) * variance);
			double temp2 = pow(E, -(y - expectation) * (y - expectation) / (2 * variance));
			return temp1 * temp2;
		}
		double error(const vector<double> &v)const
		{//计算前后两次模型参数二范数
			double temp = pow(weight - v[0], 2) + pow(expectation - v[1], 2)
				+ pow(variance - v[2], 2);
			return sqrt(temp);
		}
		double setGM(const vector<double> &v)
		{//设置模型参数
			double err = error(v);
			weight = v[0], expectation = v[1], variance = v[2];
			return err;
		}
	};
private:
	vector<GM> gmm;
	double eps;//精度
private:
	void initGMM(const vector<double> &data)
	{//初始化GMM
		const int k = gmm.size(), n = data.size();
		vector<double> sum(k);
		double data_sum = 0.0;
		for (int j = 0; j != n; ++j)
		{
			sum[j % k] += data[j];
			data_sum += data[j];
		}
		for (int i = 0; i != k; ++i)
		{
			gmm[i].expectation = sum[i] / (n / k);
			gmm[i].variance = (gmm[i].expectation - data_sum / n) *
				(gmm[i].expectation - data_sum / n);
		}
	}
public:
	GMM(int k, double _eps) :gmm(k), eps(_eps)
	{
		double w = 1.0 / k;
		for (int i = 0; i != gmm.size(); ++i)
			gmm[i].weight = w;
	}
	void train(const vector<double>&);
	double compute(double y)const
	{
		double sum = 0.0;
		for (int i = 0; i != gmm.size(); ++i)
			sum += gmm[i].compute(y);
		return sum;
	}
	void print()const
	{
		printf("weight         expectation   variance\n");
		for (int i = 0; i != gmm.size(); ++i)
			printf("%-15lf%-15lf%-15lf\n", gmm[i].weight, gmm[i].expectation, gmm[i].variance);
	}
};

void GMM::train(const vector<double> &data)
{//训练模型
	initGMM(data);
	const int k = gmm.size(), n = data.size();
	vector<vector<double>> response(n, vector<double>(k + 1, 0.));
	while (true)
	{
		for (int i = 0; i != n; ++i)
		{//对每个样本
			response[i][k] = 0.0;
			for (int j = 0; j != k; ++j)
			{//计算各个高斯模型分量的响应
				response[i][j] = gmm[j].compute(data[i]);
				response[i][k] += response[i][j];
			}
			for (int jj = 0; jj != k; ++jj)
				response[i][jj] /= response[i][k]; //计算响应度
		}
		bool is_continue = false;
		for (int j = 0; j != k; ++j)
		{//对于每个高斯模型分量更新参数
			double sum_e = 0., sum_v = 0., sum_w = 0.;
			for (int i = 0; i != n; ++i)
			{
				sum_e += response[i][j] * data[i]; //均值
				sum_v += response[i][j] * (data[i] - gmm[j].expectation) *
					(data[i] - gmm[j].expectation); //方差
				sum_w += response[i][j];//权重
			}
			double err = gmm[j].setGM({ sum_w / n, sum_e / sum_w, sum_v / sum_w });
			if (err >= eps) is_continue = true;
		}
		if (!is_continue) break;
	}
}

int main()
{
	vector<double> data = { -67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75 };
	GMM gmm(3, 0.001);
	gmm.train(data);
	gmm.print();
	cout << gmm.compute(24) << endl;
	getchar();
	return 0;
}