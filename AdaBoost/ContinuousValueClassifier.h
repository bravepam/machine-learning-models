
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#ifndef CONTINUOUSVALUECLASSIFIER_H
#define CONTINUOUSVALUECLASSIFIER_H

#include<vector>
#include"sample.h"

using namespace std;

//连续值基本分类器，可以和离散值基本分类器组成继承关系
class ContinuousValueBaseCalssifier
{
private:
	int fte_id;
	double fte_value;	//特征值
	bool less;		//用于分类决策
	double a;
public:
	ContinuousValueBaseCalssifier(int fi, double fv, double _a, bool _less) :
		fte_id(fi), fte_value(fv), less(_less), a(_a){}

	double decide(const vector<double> &x)const
	{//该基本分类器的决策函数
		int tag = 0;
		if (less) //为真，则小于分类为1
			tag = x[fte_id] < fte_value ? 1 : -1;
		else //为假，则不小于分类为1
			tag = !(x[fte_id] < fte_value) ? 1 : -1;
		return a * tag;
	}
};

//连续值分类器特征选择准则
class ContinuousClassifierFeatureSelection
{
private:
	vector<sample<double>> data; 
	vector<double> weight;		//样本权重
private:
	pair<double, double> selectFeatureValue(int, bool&);
public:
	ContinuousClassifierFeatureSelection(vector<sample<double>> &d) :
		data(move(d)), weight()
	{
		double w = 1.0 / data.size();		//初始样本权值
		weight.resize(data.size());
		fill_n(weight.begin(), weight.size(), w);
	}

	pair<double, int> select(double&, bool&);
	void updateWeight(const ContinuousValueBaseCalssifier&);
};

//对于特征fte_id，选择能使分类误差率达到最小的特征值，返回值为特征值、分类误差率和分类决策bool值
pair<double, double> ContinuousClassifierFeatureSelection::selectFeatureValue(int fte_id, bool &less)
{
	double min_error = INT_MAX, fte_value = 0.0;
	for (int i = 0; i != data.size(); ++i)
	{//对于该特征的每一个值
		double error_less = 0.0, error_not_less = 0.0;
		for (int j = 0; j != data.size(); ++j)
		{//将每个样本进行筛选
			if (data[j].x[fte_id] < data[i].x[fte_id]) //若小于，则分类为1
				error_less += static_cast<double>(data[j].y != 1) * weight[j]; //累计分类误差率
			else error_less += static_cast<double>(data[j].y != -1) * weight[j];

			if (data[j].x[fte_id] >= data[i].x[fte_id]) //若不小于，则分类为1
				error_not_less += static_cast<double>(data[j].y != 1) * weight[j];
			else error_not_less += static_cast<double>(data[j].y != -1) * weight[j];
		}
		//比较两种情况下的分类误差率和当前最小分类误差率，以获得最小分类误差率等信息，并记下
		if (error_less < min_error)
		{
			min_error = error_less;
			fte_value = data[i].x[fte_id];
			less = true;
		}
		if (error_not_less < min_error)
		{
			min_error = error_not_less;
			fte_value = data[i].x[fte_id];
			less = false;
		}
	}
	return pair<double, double>(fte_value, min_error);
}

//选择能使分类误差率最小的特征，返回值依次为特征值、特征、最小分类误差率和分类决策bool值
pair<double, int> ContinuousClassifierFeatureSelection::select(double &min_error, bool &less)
{
	int fte_values = data[0].x.size(), fte_id = -1;
	double fte_value = 0.0;
	min_error = INT_MAX;
	for (int j = 0; j != fte_values; ++j)
	{//对每一个特征
		pair<double, double> slt = selectFeatureValue(j, less); //进行筛选
		if (slt.second < min_error)
		{//以期得到能使分类误差率最小的特征等信息
			min_error = slt.second;
			fte_value = slt.first;
			fte_id = j;
		}
	}
	return pair<double, int>(fte_value, fte_id);
}

//利用当前的基本分类器更新样本权重
void ContinuousClassifierFeatureSelection::updateWeight(const ContinuousValueBaseCalssifier &cvbc)
{
	double zm = 0.0;
	for (int i = 0; i != data.size(); ++i)
		zm += weight[i] * pow(e, -1.0 * data[i].y * cvbc.decide(data[i].x));
	for (int j = 0; j != weight.size(); ++j)
		weight[j] *= pow(e, -1.0 * data[j].y * cvbc.decide(data[j].x)) / zm;
}

//连续值分类器
class ContinuousValueClassifier
{
private:
	vector<ContinuousValueBaseCalssifier> cvbc; //基本分类器集合
	ContinuousClassifierFeatureSelection ccfs;	//特征选择器
	double eps;		//精度
public:
	ContinuousValueClassifier(vector<sample<double>> &data, double _e) :
		cvbc(), ccfs(data), eps(_e){}

	void train();
	int decide(const vector<double> &x)const
	{//决策函数
		double res = 0.0;
		for (int i = 0; i != cvbc.size(); ++i)
			res += cvbc[i].decide(x);	//累计决策值
		if (res > 0.0) return 1;
		else return -1;
	}
};

//训练分类器
void ContinuousValueClassifier::train()
{
	double prev_em = INT_MAX;		//前一次分类误差率
	while (true)
	{
		double em = 0.0;
		bool less = false;
		pair<double, int> slt = ccfs.select(em, less); //选择最适合的特征
		if (em > prev_em) return; //若和前一次相比，误差率没有下降，则退出
		else prev_em = em;
		double a = 0.5 * log((1.0 - em) / em); //计算分类器权重
		ContinuousValueBaseCalssifier bc(slt.second, slt.first, a, less); //构造基本分类器
		ccfs.updateWeight(bc); //更新样本权重
		cvbc.push_back(bc);
		if (em < eps) return; //达到精度，则退出
	}
}


#endif