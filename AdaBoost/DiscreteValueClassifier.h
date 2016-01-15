//Boost分类器，特征值是离散的

#ifndef DISCRETEVALUECLASSIFIER_H
#define DISCRETEVALUECLASSIFIER_H

#include<vector>
#include"sample.h"

using namespace std;

//离散特征值基本分类器
class DiscreteValueBaseClassifier
{
private:
	int fte_id;		//所选特征
	int fte_value;		//对应特征的某特征值
	bool equal;		//表明该如何决策
	double a;		//分类器系数，即权重
public:
	DiscreteValueBaseClassifier(int fi, int fv, double _a, bool loe) :
		fte_id(fi), fte_value(fv), a(_a), equal(loe){}
	double decide(const vector<int> &x)const
	{//该基本分类器的决策函数
		int tag = 0;
		if (equal) //等于，则分类为1
			tag = x[fte_id] == fte_value ? 1 : -1;
		else  //不等于，则分类为1
			tag = x[fte_id] != fte_value ? 1 : -1;
		return a * tag;
	}
};

//离散值分类器的特征选择准则
class DiscreteClassifierFeatureSelection
{
private:
	vector<sample<int>> data;		//样本集合
	vector<double> weight;		//各个样本的权重
	vector<int> fte_values;		//每个特征的取值个数
private:
	pair<double, int> selectFeatureValue(int, bool&);
public:
	DiscreteClassifierFeatureSelection(vector<sample<int>> &d, vector<int> &fv)
		:data(move(d)), fte_values(move(fv)), weight()
	{
		double w = 1.0 / data.size();		//初始，每个样本的权重均为1 / N
		weight.resize(data.size());
		fill_n(weight.begin(), data.size(), w);
	}

	bool hasFeature()const
	{//判断是否还有特征可用
		for (int j = 0; j != fte_values.size(); ++j)
			if (fte_values[j] != 0) return true;
		return false;
	}

	void zeroFeature(int fte_id)
	{//如果特征已用，则置0
		fte_values[fte_id] = 0;
	}

	void updateWeight(const DiscreteValueBaseClassifier &bc)
	{//通过当前的基本分类器更新样本权重
		double zm = 0.0;
		for (int i = 0; i != data.size(); ++i)
			zm += weight[i] * pow(e, -1.0 * data[i].y * bc.decide(data[i].x));
		for (int j = 0; j != weight.size(); ++j)
			weight[j] *= pow(e, -1.0 * data[j].y * bc.decide(data[j].x)) / zm;
	}

	pair<int, int> select(double&, bool&);
};

//特征选择，返回值依次为所选特征、特征值、最小分类误差率和达到该误差率时的分类决策bool值
pair<int, int> DiscreteClassifierFeatureSelection::select(double &min_error, bool &equal)
{
	int fte_id = -1, fte_value = -1;
	min_error = INT_MAX;
	for (int i = 0; i != fte_values.size(); ++i)
	{//遍历每一个特征
		if (fte_values[i] == 0) continue;

		//从中选出能够使其分类误差率最小的特征值等信息
		pair<double, int> slt = selectFeatureValue(i, equal);
		if (slt.first < min_error)
		{//如果分类器误差率比当前更小，则更新
			min_error = slt.first;
			fte_id = i;
			fte_value = slt.second;
		}
	}
	return pair<int, int>(fte_id, fte_value);
}

//对于特征fte_id，选择能使其分类误差率最小的特征值，返回值依次为分类误差率、特征值以及决策bool值
pair<double, int> DiscreteClassifierFeatureSelection::selectFeatureValue(int fte_id, bool &equal)
{
	double min_error = INT_MAX;
	int temp_fte_value = -1;
	for (int j = 0; j != fte_values[fte_id]; ++j)
	{//对于该特征的每一个取值
		double error_equal = 0.0, error_not_equal = 0.0;
		for (int i = 0; i != data.size(); ++i)
		{//扫描每一个样本
			if (data[i].x[fte_id] == j) //假设样本对应值等于该特征值
				//则分类为1，累加分类误差率
				error_equal += static_cast<double>(data[i].y != 1) * weight[i];
			else
				error_equal += static_cast<double>(data[i].y != -1) * weight[i];

			if (data[i].x[fte_id] != j) //假设样本对应值不等于该特征值
				//则分类为1，累加分类误差率
				error_not_equal += static_cast<double>(data[i].y != 1) * weight[i];
			else
				error_not_equal += static_cast<double>(data[i].y != -1) * weight[i];
		}
		//比较两种情况下的分类误差率和当前最小分类误差率，以获得最小值，记下相关信息
		if (error_equal < min_error)
		{
			min_error = error_equal;
			temp_fte_value = j;
			equal = true;
		}
		if (error_not_equal < min_error)
		{
			min_error = error_not_equal;
			temp_fte_value = j;
			equal = false;
		}
	}
	return pair<double, int>(min_error, temp_fte_value);
}

//离散值Boost分类器
class DicreteValueClassifier
{
private:
	vector<DiscreteValueBaseClassifier> dvbc;		//基本分类器集合
	DiscreteClassifierFeatureSelection dcfs;		//特征选择器
	double eps;		//精度，控制何时训练停止
public:
	DicreteValueClassifier(vector<sample<int>> &data, vector<int> &fv, double _e)
		:dvbc(), dcfs(data, fv), eps(_e){}
	void train();
	int decide(const vector<int> &d)const
	{//决策函数
		double res = 0.0;
		for (int i = 0; i != dvbc.size(); ++i)
			res += dvbc[i].decide(d); //累加每个分类器的加权决策值
		if (res > 0.0) return 1;
		else return -1;
	}
};

//训练分类器
void DicreteValueClassifier::train()
{
	double prev_em = INT_MAX; //前一次的分类误差率
	while (true)
	{
		if (!dcfs.hasFeature()) return; //没有特征可用时，返回
		double em = 0.0;
		bool equal = false;
		pair<int, int> slt = dcfs.select(em, equal); //选择特征，使分类误差率最小
		if (em > prev_em) return; //如果该分类误差率没有下降，则退出
		else prev_em = em;
		double a = 0.5 * log((1 - em) / em); //计算分类器权重
		DiscreteValueBaseClassifier bc(slt.first, slt.second, a, equal); //构造基本分类器
		dvbc.push_back(bc);
		dcfs.updateWeight(bc);		//更新样本权值
		dcfs.zeroFeature(slt.first);		//置0当前特征，表示已用
		if (em < eps) return;
	}
}


#endif