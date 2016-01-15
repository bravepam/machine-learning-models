#ifndef REGRESSIONBOOSTINGTREE_H
#define REGRESSIONBOOSTINGTREE_H

#include<utility>
#include<vector>
#include"sample.h"

using namespace std;

//回归问题的Boost提升树基本分类器，即树桩
class RegressionBaseClassifier
{
private:
	int fte_id;
	double fte_value;
	double less;		//小于特征值，则取该值
	double great;		//不小于，则取该值
public:
	RegressionBaseClassifier(pair<pair<double, double>, pair<double, int>> &args) :
		fte_id(args.second.second), fte_value(args.second.first),
		less(args.first.first), great(args.first.second){}

    double predict(const vector<double> &x)const
	{//决策
		return x[fte_id] < fte_value ? less : great;
	}
};

//回归问题特征选择准则
class RegressionFeatureSelection
{
private:
	vector<sample<double>> data;	//样本
	//vector<double> residual;		//每次获得新的提升树之后的预测残差，直接在样本中更新
private:
	double squareError(double, const vector<int>&);
	pair<pair<double, double>, pair<double, double>> selectFeatureValue(int);

public:
	RegressionFeatureSelection(vector<sample<double>> &d) :data(move(d)){}

	pair<pair<double, double>, pair<double, int>> select(double&);
	vector<sample<double>>& getData()
	{
		return data;
	}
};

//选择能使预测误差最小的特征等信息，返回值依次为<less,great>，<特征值，特征>以及最小预测误差平方和
pair<pair<double, double>, pair<double, int>> RegressionFeatureSelection::select(double &min_error)
{
	int fte_values = data[0].x.size();
	pair<double, double> min_avg;
	pair<double, int> fte;
	min_error = INT_MAX;
	for (int i = 0; i != fte_values; ++i)
	{//对每个特征
		//选择能使预测误差平方和最小的特征值等信息
		pair<pair<double, double>, pair<double, double>> slt = selectFeatureValue(i);
		if (slt.second.second < min_error)
		{//若比当前预测误差平方和更小，则更新
			min_error = slt.second.second;
			min_avg = slt.first;
			fte = make_pair(slt.second.first, i);
		}
	}
	return make_pair(min_avg, fte);
}

//计算某一集合数据的方差，即预测误差平方和
double RegressionFeatureSelection::squareError(double avg, const vector<int> &data_id)
{
	double error = 0.0;
	for (int i = 0; i != data_id.size(); ++i)
		error += (data[data_id[i]].y - avg) * (data[data_id[i]].y - avg);
	return error;
}

//对于特征fte_id，选择最适合的特征值，返回值为<less，great>，<特征值，最小预测误差平方和>
pair<pair<double, double>, pair<double, double>> RegressionFeatureSelection::selectFeatureValue(int fte_id)
{
	pair<double, double> avg, value_error = make_pair(-1.0, INT_MAX);
	for (int i = 0; i != data.size(); ++i)
	{
		vector<int> less, not_less;
		double less_avg = 0.0, not_less_avg = 0.0;
		for (int j = 0; j != data.size(); ++j)
		{
			if (data[j].x[fte_id] < data[i].x[fte_id])
			{
				less.push_back(j);
				less_avg += data[j].y;
			}
			else
			{
				not_less.push_back(j);
				not_less_avg += data[j].y;
			}
		}
		less_avg /= (less.empty() ? 1 : less.size());
		not_less_avg /= (not_less.empty() ? 1 : not_less.size());
		double error = squareError(less_avg, less) + squareError(not_less_avg, not_less);
		if (error < value_error.second)
		{
			value_error = make_pair(data[i].x[fte_id], error);
			avg = make_pair(less_avg, not_less_avg);
		}
	}
	return make_pair(avg, value_error);
}

//回归问题提升树
class BoostingTree
{
private:
	vector<RegressionBaseClassifier> vrbc;	//树桩集合
	RegressionFeatureSelection rfs;			//特征选择器
	double eps;			//迭代精度
	const int iter;		//迭代次数，默认不开启
private:
	void updataWeight(const RegressionBaseClassifier&, vector<sample<double>>&);
public:
	BoostingTree(vector<sample<double>> &d, double _e, int _iter = -1) :
		vrbc(), rfs(d), eps(_e), iter(_iter){}

	void train();
	double predict(const vector<double>&);
};

//预测样本点的值
double BoostingTree::predict(const vector<double> &x)
{
	double res = 0.0;
	for (int i = 0; i != vrbc.size(); ++i)
		res += vrbc[i].predict(x);  //每个树桩的预测值之和
	return res;
}

//训练提升树
void BoostingTree::train()
{
	int i = 0; //迭代次数
	while (true)
	{
		++i;
		double min_error = 0.0;
		//选择能使预测误差平方和最小的特征等信息
		pair<pair<double, double>, pair<double, int>> slt = rfs.select(min_error);
		RegressionBaseClassifier rbc(slt); //构造树桩
		vrbc.push_back(rbc);
		updataWeight(rbc, rfs.getData()); //更新预测误差，用来进行记下一次预测
		if (min_error < eps || i == iter) return; //满足其一，则退出
	}
}

//更新预测误差
void BoostingTree::updataWeight(const RegressionBaseClassifier &rbc, vector<sample<double>> &data)
{
	for (int i = 0; i != data.size(); ++i)
		data[i].y -= rbc.predict(data[i].x); //当前值减去预测值，即预测误差
}

#endif