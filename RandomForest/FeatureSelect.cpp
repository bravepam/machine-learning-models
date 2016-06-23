#include"FeatureSelect.h"
#include<hash_map>
#include<algorithm>
#include<cassert>

void InfoGain::setFeaturesAndData(std::vector<size_t>& fs, std::vector<size_t>& di)
{
	features.swap(fs);
	data_id.swap(di);
	splits.resize(features.size());
	data.resize(features.size());
	getSplits(); 
}

void InfoGain::clear()
{
	features.clear();
	splits.clear();
	data.clear();
	data_id.clear();
}

std::pair<bool, size_t> InfoGain::checkData(std::hash_map<int, size_t>& cls_count)const
{
	cls_count = countingClass(-1, 0, data_id.size()); //统计该树节点所用数据集的类别数
	size_t cls_value = UINT_MAX, count = 0;
	for (auto iter = cls_count.begin(); iter != cls_count.end(); ++iter)
	{//找出哪个类别占数最大
		if (iter->second > count)
		{
			count = iter->second;
			cls_value = iter->first;
		}
	}
	return{ count == data_id.size(), cls_value }; //返回值为是否为同一类别，占数最大的类别
}

std::hash_map<int, size_t> InfoGain::countingClass(int which, size_t first, size_t last)const
{//which是特征ID的ID，即features[i]的i
	std::hash_map<int, size_t> cls_count;
	//which如果是-1，表明统计整个数据集的类别信息；否则是以某个特征下某一范围的类别信息
	const size_t that = (which == -1 ? 0 : which);
	for (size_t i = first; i != last; ++i)
		++cls_count[data[that][i].cls];
	return std::move(cls_count);
}

double InfoGain::entropyAux(const std::hash_map<int, size_t>&cls_count, size_t size)const
{
	double epy = 0.0, prob = 0.0;
	for (auto iter = cls_count.begin(); iter != cls_count.end(); ++iter)
	{
		prob = iter->second * 1.0 / size;
		epy += -prob * log2(prob);
	}
	return epy;
}

inline double InfoGain::entropy(int which, size_t first, size_t last)const
{
	//先统计类别
	std::hash_map<int, size_t> cls_count = countingClass(which, first, last);
	return entropyAux(cls_count, last - first); //再计算熵
}

inline double InfoGain::conditionalEntropy(size_t which, const split& s, size_t last)const
{
	//以which特征分割数据集后的熵
	const double less = entropy(which, 0, s.index);
	const double greater = entropy(which, s.index, last);
	double cdl_epy = less * s.index / last;
	cdl_epy += greater * (last - s.index) / last;
	return cdl_epy;
}

void InfoGain::getSplits()
{
	const std::vector<sample>& trainset = pdata->prf->train_set;

	for (size_t i = 0; i != features.size(); ++i)
	{//对每一个特征
		data[i].reserve(data_id.size());
		for (size_t j = 0; j != data_id.size(); ++j)
		{//在每一个数据样本中
			//提取出{样本id，特征值，该样本类别值}构成一个FeatCls对象
			data[i].emplace_back(data_id[j], trainset[data_id[j]].x[features[i]], trainset[data_id[j]].y);
		}
	}

	//通过data获得所有分割点
	for (size_t i = 0; i != data.size(); ++i)
	{//对每个特征下的data数据集进行扫描
		sort(data[i].begin(), data[i].end()); //先将FeatCls对象集合data[i]非降序排序
		int cls = data[i][0].cls;
		splits[i].reserve(100);
		for (size_t j = 1; j != data[i].size(); ++j)
		{//该集合中的扫描每一个对象
			if (data[i][j].cls != cls)
			{//如果类别发生变化，那么说明出现一个分割点
				//获得分割点值
				const double temp = (data[i][j - 1].feat_value + data[i][j].feat_value) / 2.0;
				splits[i].emplace_back(j, temp); //{分割点索引，分割点值}
				cls = data[i][j].cls;
			}
		}
	}
}

std::pair<size_t, double> InfoGain::select(std::vector<std::vector<size_t>>& splited_data_id)
{
	assert(!features.empty() && !splits.empty() && !data_id.empty());
	size_t split_feat_id_id = UINT_MAX;
	//const double cur_epy = entropy(-1, 0, data_id.size()); //当前数据集的熵
	double min_cdl_epy = INT_MAX;
	split best_split(0, 0.0);
	for (size_t i = 0; i != features.size(); ++i)
	{//对每个特征
		for (size_t j = 0; j != splits[i].size(); ++j)
		{//扫描它所有的分割点
			//计算以该分割点分割数据集后的条件熵
			const double ret = conditionalEntropy(i, splits[i][j], data[i].size());
			if (ret < min_cdl_epy)
			{//找出条件熵最小的分割
				split_feat_id_id = i; //特征ID的ID，即features[i]中的i
				best_split = splits[i][j];
				min_cdl_epy = ret;
			}
		}
	}
	printf("epy: %lf\t", min_cdl_epy);
	//根据最大分割点分割数据集
	assert(splited_data_id.size() >= 2);
	splited_data_id[0].reserve(best_split.index);
	splited_data_id[1].reserve(data_id.size() - best_split.index);
	//比分割点值小的数据集id集合
	for (size_t i = 0; i != best_split.index; ++i)
		splited_data_id[0].push_back(data[split_feat_id_id][i].sample_id);
	//比分割点值大的数据集id集合
	for (size_t i = best_split.index; i != data[split_feat_id_id].size(); ++i)
		splited_data_id[1].push_back(data[split_feat_id_id][i].sample_id);

	//返回值{最佳分割特征ID，最佳分割点值}
	return{ features[split_feat_id_id], best_split.split_value };
}
