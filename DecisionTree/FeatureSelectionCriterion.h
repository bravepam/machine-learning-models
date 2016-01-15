/**************************************************************************************
 *该头文件包含的主要都是决策树特征选择的准则，有四种，结构如下：
 *1、criterion是包含三个纯虚函数的抽象基类，作为其他准则类的父类使用；
 *2、InfoGain：基于信息增益来选择特征，继承于criterion，重定义了三个纯虚函数，
 *   并有自己的数据成员；
 *3、InfoGainRatio：基于信息增益比来选择特征，继承于InfoGain，有一个自己
 *   的私有函数，重定义了select虚函数；
 *4、Gini：基于基尼指数来选择特征，继承于InfoGain，重定义了三个纯虚函数；
 *5、LeastSquareError：基于最小平方和来选择特征，用于生成回归树的特征选择，无基类
 **************************************************************************************/


#ifndef FEATURESELECTIONCRITERION_H
#define FEATURESELECTIONCRITERION_H

#include<iostream>
#include<utility>
#include<vector>

using namespace std;

template<typename T>
struct sample
{//样本
	vector<T> x;//样本点，即特征
	T y;//特征取值，即所属类别
	sample(vector<T> &_x, T _y) :x(move(_x)), y(_y){}
	void print()const
	{
		cout << '(';
		for (int i = 0; i != x.size(); ++i)
		{
			cout << x[i];
			if (i != x.size() - 1)
				cout << ',';
		}
		cout << ") " << y << endl;
	}
};

//-------------------------------------------------------------------------------------------------------
class criterion
{//特征选择准则虚基类
public:
	virtual double entropy(vector<int>&, int) = 0;//计算熵
	virtual pair<double, int> conditionalEntropy(vector<int>&, int, vector<vector<int>>&) = 0;//计算条件熵
	//选择最优分割特征
	virtual pair<double, int> select(vector<int>&, vector<vector<int>>&) = 0;
};

//-------------------------------------------------------------------------------------------------------
class InfoGain :public criterion
{//基于信息增益的特征选择，继承于准则基类
	//将实现所有的虚函数
protected:
	vector<sample<int>> data;//样本集合
	vector<int> feature_values;//每个特征所能取的特征值的个数
	int cls_num;//类别个数
public:
	InfoGain(const vector<sample<int>> &d, const vector<int> &fv, int cn) :
		data(move(d)), feature_values(move(fv)), cls_num(cn){}

	//自定义的三个函数
	//检查当前数据集是否是同一类别，如果是，则返回true和类别值；
	//如果不是，则返回false和占多数的类别值
	pair<bool, int> checkData(vector<int>&);
	bool hasFeature();//是否还有可用特征
	int getSpecificFeatureValues(int fte_id)
	{
		return feature_values[fte_id];
	}
	void zeroSpecificFeatureValues(int fte_id)
	{
		feature_values[fte_id] = 0;
	}

	//计算数据集相对于类别的熵
	virtual double entropy(vector<int>&, int);
	//计算数据集相对于某一特征的条件熵
	virtual pair<double, int> conditionalEntropy(vector<int>&, int, vector<vector<int>>&);
	virtual pair<double, int> select(vector<int>&, vector<vector<int>>&);
	virtual ~InfoGain(){}
};

pair<bool, int> InfoGain::checkData(vector<int> &data_id)
{//检查数据集
	vector<int> cls_count(cls_num, 0);//计数每个类别的样本数
	for (int i = 0; i != data_id.size(); ++i)
	{
		for (int j = 0; j != cls_count.size(); ++j)
			if (data[data_id[i]].y == j)
			{
				++cls_count[j];
				break;
			}
	}
	int count = 0, cls_value = -1;
	for (int k = 0; k != cls_count.size(); ++k)
		if (cls_count[k] > count)
		{//记下占多数的类别
			count = cls_count[k];
			cls_value = k;
		}
	return pair<bool, int>(count == data_id.size(), cls_value);
}

bool InfoGain::hasFeature()
{//检查是否还有特征可用
	bool has = false;
	for (int i = 0; i != feature_values.size(); ++i)
	{
		has = feature_values[i] != 0;//如果特征取值个数不为0，则说明可用
		if (has) return true;
	}
	return false;
}

//fte_id即为特征ID
double InfoGain::entropy(vector<int> &data_id, int fte_id)
{
	int value_num = (fte_id == -1 ? cls_num : feature_values[fte_id]);
	vector<int> specific_fte_value_count(value_num, 0);//各特征值的所占样本数
	for (int i = 0; i != data_id.size(); ++i)
	{//统计各特征值的样本数
		for (int j = 0; j != value_num; ++j)
			if ((fte_id == -1 ? data[data_id[i]].y : data[data_id[i]].x[fte_id]) == j)
			{
				++specific_fte_value_count[j];
				break;
			}
	}
	double epy = 0.0, prob = 0.0;
	for (int k = 0; k != specific_fte_value_count.size(); ++k)
	{//计算熵
		if (specific_fte_value_count[k] > 0)
		{//数据子集必须大于0
			prob = specific_fte_value_count[k] * 1.0 / data_id.size();
			epy += -prob * log2(prob);
		}
	}
	return epy;
}

/*计算在某一特征下的条件熵
*fte_id是该特征ID，splited_data_id是分割后的子数据集集合
*/
pair<double, int> InfoGain::conditionalEntropy(vector<int> &data_id, int fte_id, 
	vector<vector<int>> &splited_data_id)
{
	for (int i = 0; i != data_id.size(); ++i)
	{//对每个样本
		for (int j = 0; j != feature_values[fte_id]; ++j)
			if (data[data_id[i]].x[fte_id] == j)
			{//统计其所属的子数据集
				splited_data_id[j].push_back(data_id[i]);
				break;
			}
	}
	double cdl_epy = 0.0, prob = 0.0;
	for (int k = 0; k != splited_data_id.size(); ++k)
	{//计算条件熵
		if (!splited_data_id[k].empty())
		{
			prob = splited_data_id[k].size() * 1.0 / data_id.size();//该数据子集所占总数据集的频率
			cdl_epy += prob * entropy(splited_data_id[k], -1);//条件熵
		}
	}
	return pair<double, int>(cdl_epy, int());//只需要条件熵
}

/*选择最优的分割特征
*splited_data_id是最优分割后的子数据集集合
*/
pair<double, int> InfoGain::select(vector<int> &data_id, vector<vector<int>> &splited_data_id)
{
	int split_fte_id = -1;//分割变量（特征）
	//相对于类别的熵
	double cls_epy = entropy(data_id, -1), max_info_gain = 0.0;
	for (int j = 0; j != feature_values.size(); ++j)
	{//对每一个特征
		if (feature_values[j] == 0) continue;//表示该特征已经被用过了
		vector<vector<int>> temp_splited_data(feature_values[j]);
		//都计算它的条件熵
		pair<double, int> ret = conditionalEntropy(data_id, j, temp_splited_data);
		if (cls_epy - ret.first > max_info_gain)
		{//以找出其中最大的信息增益
			max_info_gain = cls_epy - ret.first;
			split_fte_id = j;//记下当前最优分割特征
			splited_data_id.swap(temp_splited_data);//以及分割子数据集集合
		}
	}
	return pair<double, int>(max_info_gain, split_fte_id);//返回最大信息增益和分割特征
}

//-------------------------------------------------------------------------------------------------------
class InfoGainRatio :public InfoGain
{//基于信息增益比的特征选择，继承于信息增益准则
private:
	//自定义的私有成员函数
	double infoGainRatio(vector<int>&, int, vector<vector<int>>&);
public:
	InfoGainRatio(const vector<sample<int>> &d, const vector<int> &fv, int cn) :
		InfoGain(d, fv, cn){}
	//重载最优分割特征选择函数，其实基类稍作修改就可以为其所用
	virtual pair<double, int> select(vector<int>&, vector<vector<int>>&);
	virtual ~InfoGainRatio(){}
};

/*计算信息增益比
*fte_id特征ID
*splited_data_id为分割后的子数据集集合
*/
double InfoGainRatio::infoGainRatio(vector<int> &data_id, int fte_id,vector<vector<int>> &splited_data_id)
{
	double cls_epy = entropy(data_id, -1);//调用继承的基类成员计算相对于类别的熵
	//同样，使用基类成员计算在fte_id特征下的条件熵
	pair<double, int> ret = conditionalEntropy(data_id, fte_id, splited_data_id);
	double info_gain = cls_epy - ret.first;//得到信息增益
	double fte_id_epy = entropy(data_id, fte_id);//计算相对于fte_id特征的熵
	return info_gain / fte_id_epy;//得到信息增益比
}

/*选择最优的分割特征
*splited_data_id是最优分割后的子数据集集合
*/
pair<double, int> InfoGainRatio::select(vector<int> &data_id, vector<vector<int>> &splited_data_id)
{
	double max_info_gain_ratio = 0.0;
	int split_fte_id = -1;//分割特征
	for (int j = 0; j != feature_values.size(); ++j)
	{//对每一个特征
		if (feature_values[j] == 0) continue;//表示该特征已经被用过了
		vector<vector<int>> temp_splited_data(feature_values[j]);
		//计算出增益比
		double info_gain_ratio = infoGainRatio(data_id, j, temp_splited_data);
		if (info_gain_ratio > max_info_gain_ratio)
		{//以得到最大增益比
			max_info_gain_ratio = info_gain_ratio;
			split_fte_id = j;//记下当前最优分割特征
			splited_data_id.swap(temp_splited_data);//和分割数据集
		}
	}
	return pair<double, int>(max_info_gain_ratio, split_fte_id);//返回最大增益比和分割特征
}

//-------------------------------------------------------------------------------------------------------
class Gini :public InfoGain
{//基于基尼指数的特征选择准则，继承于InfoGain
public:
	Gini(const vector<sample<int>> &d, const vector<int> &fv, int cn) :
		InfoGain(d, fv, cn){}
	virtual double entropy(vector<int>&, int);
	virtual pair<double, int> conditionalEntropy(vector<int>&, int, vector<vector<int>>&);

	virtual pair<double, int> select(vector<int>&, vector<vector<int>>&);
	virtual ~Gini(){}
};

/*选择最优的分割特征
*splited_data_id是最优分割后的子数据集集合
*/
pair<double, int> Gini::select(vector<int> &data_id, vector<vector<int>> &splited_data_id)
{
	double min_gini_index = INT_MAX;
	//分割特征和分割特征的取值
	int split_fte_id = -1, split_fte_value = -1;
	for (int j = 0; j != feature_values.size(); ++j)
	{//对每一个特征
		if (feature_values[j] == 0) continue;//表示该特征已经被用过了
		vector<vector<int>> temp_splited_data(2);
		//计算它的最小基尼指数和相应的特征取值
		pair<double, int> ret = conditionalEntropy(data_id, j, temp_splited_data);
		if (ret.first < min_gini_index)
		{//已取得全局最小的基尼指数
			min_gini_index = ret.first;
			split_fte_id = j;//记下当前分割特征
			split_fte_value = ret.second;//和特征取值
			splited_data_id.swap(temp_splited_data);//以及分割的子数据集集合
		}
	}
	return pair<double, int>(split_fte_id, split_fte_value);//返回最优分割特征和特征取值
}

//计算基尼指数，不再是熵了，此时fte_id只可能是-1，即类别
double Gini::entropy(vector<int> &data_id, int fte_id)
{
	vector<int> specific_fte_value_count(cls_num, 0);
	for (int i = 0; i != data_id.size(); ++i)
	{//统计每个类别值得样本数
		for (int j = 0; j != cls_num; ++j)
			if (data[data_id[i]].y == j)
			{
				++specific_fte_value_count[j];
				break;
			}
	}
	double gini_index = 1.0;
	for (int k = 0; k != specific_fte_value_count.size(); ++k)
	{
		double temp = specific_fte_value_count[k] * 1.0 / data_id.size();
		gini_index -= temp * temp;
	}
	return gini_index;
}

//计算下特征fte_id分割下的基尼指数
pair<double, int> Gini::conditionalEntropy(vector<int> &data_id,int fte_id, 
	vector<vector<int>> &splited_data_id)
{
	vector<int> yes, no;
	double min_gini_index = INT_MAX, gini_index = 0.0;
	int split_value = -1;//分割特征的取值
	for (int j = 0; j != feature_values[fte_id]; ++j)
	{//对每一个特征取值
		for (int i = 0; i != data_id.size(); ++i)
		{//统计等于和不等于该值得样本数
			if (data[data_id[i]].x[fte_id] == j)
				yes.push_back(data_id[i]);
			else no.push_back(data_id[i]);
		}
		gini_index = (yes.size() * 1.0 / data_id.size()) * entropy(yes, -1)
			+ (no.size() * 1.0 / data_id.size()) * entropy(no, -1);//以得到基尼指数
		if (gini_index < min_gini_index)
		{//从而获得该特征下最小的基尼指数
			min_gini_index = gini_index;
			split_value = j;//记下
			splited_data_id[0].swap(yes);//分割的子数据集只有两类
			splited_data_id[1].swap(no);
		}
		yes.clear();
		no.clear();
	}
	return pair<double, int>(min_gini_index, split_value);//返回该特征下最优特征取值和相应的基尼指数
}

//-------------------------------------------------------------------------------------------------------
class LeastSquareError
{//最小二乘特征选择准则，用于回归树，因而数据点都是浮点数
private:
	vector<sample<double>> data;//数据样本
private:
	double squareError(vector<int>&, int, double);
	pair<double, double> specificFeatureMinSquareError(vector<int>&, int, vector<vector<int>>&);
public:
	LeastSquareError(const vector<sample<double>> &d) :data(move(d)){}
	pair<double, int> select(vector<int>&, vector<vector<int>>&);
	double average(vector<int> &data_id)
	{
		double sum = 0.0;
		for (int i = 0; i != data_id.size(); ++i)
			sum += data[data_id[i]].y;
		return sum / data_id.size();
	}
};

/*计算特征j下的平方差之和
*j是特征ID
*avg是该特征下的平均值
*/
double LeastSquareError::squareError(vector<int> &data_id, int j, double avg)
{
	double se = 0.0;
	for (int i = 0; i != data_id.size(); ++i)
		se += (data[data_id[i]].x[j] - avg) * (data[data_id[i]].x[j] - avg);
	return se;
}

/*寻找字特征j下最优的分割点
*splited_data_id是最优分割后的子数据集集合
*/
pair<double, double> LeastSquareError::specificFeatureMinSquareError(vector<int> &data_id, int j,
	vector<vector<int>> &splited_data_id)
{
	double min_se = INT_MAX, min_split_value = INT_MAX;
	vector<int> L, R;
	for (int i = 0; i != data_id.size(); ++i)
	{//对每一个特征j的取值
		//split_value记下该值
		double split_value = data[data_id[i]].x[j], left_avg = 0.0, right_avg = 0.0;
		for (int k = 0; k != data_id.size(); ++k)
		{
			if (data[data_id[k]].x[j] <= split_value)
			{//统计不大于该值的样本
				L.push_back(data_id[k]);
				left_avg += data[data_id[k]].x[j];
			}
			else
			{//以及大于该值的样本
				R.push_back(data_id[k]);
				right_avg += data[data_id[k]].x[j];
			}
		}
		if (L.empty() || R.empty()) continue;
		left_avg /= L.size();//计算均值
		right_avg /= R.size();
		double se = squareError(L, j, left_avg) + squareError(R, j, right_avg);//获得平方差之和
		if (se < min_se)
		{//如果是当前最小的平方差
			min_se = se;
			min_split_value = split_value;//则记下分割值
			splited_data_id[0].swap(L);//以及分割子集
			splited_data_id[1].swap(R);
		}
		L.clear(); R.clear();
	}
	return pair<double, double>(min_se, min_split_value);//返回该特征下的最小平方和以及分割点
}

//在数据集中找到最优分割特征和最优分割点
pair<double, int> LeastSquareError::select(vector<int> &data_id, vector<vector<int>> &splited_data_id)
{
	double min_se = INT_MAX, min_split_value = INT_MAX;
	int splited_fte_id = -1, fte_num = data[data_id[0]].x.size();//分割特征和特征总数
	for (int j = 0; j != fte_num; ++j)
	{//对每一个特征
		vector<vector<int>> temp_splited_data(2);
		//都计算它的最优分割点以及相应的最小平方和
		pair<double, double> se = specificFeatureMinSquareError(data_id, j, temp_splited_data);
		if (se.first < min_se)
		{//如果是当前最优
			min_se = se.first;
			splited_fte_id = j;//则记下该特征
			min_split_value = se.second;//和分割点
			splited_data_id.swap(temp_splited_data);//以及分割子集
		}
	}
	return pair<double, int>(min_split_value, splited_fte_id);//返回最优分割特征和最优分割点
}

#endif