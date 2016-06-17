#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H

#include<vector>
#include<utility>
#include<hash_map>
#include"util.h"

//信息增益特征选择准则
class InfoGain
{
private:
	//每个样本的特征值以及类别标签
	struct FeatCls
	{
		size_t sample_id; //该样本在森林训练集中的索引
		double feat_value; //某特征的值
		int cls; //该样本类别
		FeatCls(size_t si, double fv, int c) :sample_id(si), feat_value(fv), cls(c){}
		bool operator<(const FeatCls& rhs)const
		{
			return feat_value < rhs.feat_value;
		}
	};
	//特征分割点
	struct split
	{
		size_t index; //分割点索引，该索引是data中某个FeatCls对象集合中某元素的索引
		double split_value; //分割值
		split(size_t i, double sv) :index(i), split_value(sv){}
	};

private:
	const TreeDataSet* const pdata; //指向该树所用的训练数据集
	std::vector<size_t> features; //构建某树节点时随机选择的特征id集合

	//data的大小和features一样大，两者存在对应关系，即D = data[i]存的就是以F = features[i]
	//的值构建的FeatCls对象，data在getSplits函数中会被按照特征值大小排序
	std::vector<std::vector<FeatCls>> data;

	//构建某树节点时分割点集合，它也和features一样大且存在对应关系，即S = splits[i]存的就是以
	//F = features[i]的值构建的分割点，从排序后的data中由getSplits得到
	std::vector<std::vector<split>> splits; 
	std::vector<size_t> data_id; //构建某树节点时数据集id
private:
	//统计某一范围内数据集的类别数
	std::hash_map<int, size_t> countingClass(int, size_t, size_t)const;

	//计算该数据集的熵
	double entropy(int, size_t, size_t)const;

	//计算以某一特征分割数据集后的熵，即条件熵
	double conditionalEntropy(size_t, const split&, size_t)const;

	//初始化data，并给予它获得所有的分割点
	void getSplits();
public:
	InfoGain(const TreeDataSet* const p) :pdata(p){}

	//设置每次树节点分割时所需要的随机选择的特征和数据集id集合
	void setFeaturesAndData(std::vector<size_t>&, std::vector<size_t>&);
	void clear();
	double entropyAux(const std::hash_map<int, size_t>&, size_t)const;

	//检查数据集一些特性
	std::pair<bool, size_t> checkData(std::hash_map<int, size_t>&)const;

	//选择最佳分割特征以及分割点值
	std::pair<size_t, double> select(std::vector<std::vector<size_t>>&);
};

#endif