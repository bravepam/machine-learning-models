#include"RandomForest.h"
#include"util.h"
#include<hash_map>
#include<algorithm>
#include<cassert>
#include<string>
#include<fstream>

void RandomForest::setParams(const RFParams* p)
{
	assert(p);
	prf.reset(p);
	rf.reserve(p->N);
	datasets.reserve(p->N);
	fis.reserve(p->D);
}

void RandomForest::train()
{
	assert(!prf->train_set.empty());
	for (size_t i = 0; i != prf->N; ++i)
	{
		//创建一棵随机树
		std::shared_ptr<RandomTree> prt = std::make_shared<RandomTree>(prf);
		//训练这棵树，训练完毕后将返回所用的数据集指针
		const TreeDataSet* ptds = prt->create();
		rf.emplace_back(prt); //将树添加到集合
		datasets.push_back(ptds); //数据集也是
	}
	if (prf->calc_fte_importance) //如果需要计算特征重要性
		FeatureImportance();
}

int RandomForest::predict(const sample& s)const
{
	assert(!rf.empty());
	std::hash_map<int, size_t> cls_count;
	for (size_t i = 0; i != rf.size(); ++i)
	{//针对每一棵树，都将给出一个预测类别
		++cls_count[rf[i]->predict(s)];
	}

	std::pair<int, size_t> max{ INT_MIN, 0 };
	for_each(cls_count.begin(), cls_count.end(),
		[&max](const std::hash_map<int,size_t>::value_type& item)
	{//然后从这些类别统计信息中找出占比最大的类别作为结果
		if (item.second > max.second)
			max = item;
	});
	return max.first;
}

double RandomForest::testError()
{
	assert(!prf->test_set.empty());
	if (test_err > 0.0) //已计算，则直接返回，下同
		return test_err;
	size_t error = 0;
	for_each(prf->test_set.begin(), prf->test_set.end(), [&error, this](const sample& s)
	{
		const int pred = predict(s);
		error += static_cast<size_t>(pred != s.y);
	});
	test_err = error * 1.0 / prf->test_set.size();
	return test_err;
}

//计算泛化误差，利用袋外误差估计
double RandomForest::generalizationError()
{
	if (gen_err > 0.0)
		return gen_err;
	std::vector<double> oob_errors; //记录每个样本的误差
	double sum_error = 0.0;
	const std::vector<sample>& train = prf->train_set;
	oob_errors.reserve(train.size());
	for (size_t i = 0; i != train.size(); ++i)
	{//对于训练集中的每个样本
		size_t error = 0, howmany_trees = 0;
		for (size_t j = 0; j != rf.size(); ++j)
		{//在所有的随机树中搜寻
			if (!datasets[j]->contains(i))
			{//如果该树所用的训练集中没有它，即该样本是该树的带外数据
				const int pred = rf[j]->predict(train[i]); //那么将进行预测
				error += static_cast<size_t>(pred != train[i].y);
				++howmany_trees;
			}
		}
		const double temp = error * 1.0 / howmany_trees; //该样本误差率
		sum_error += temp; //总误差率
		oob_errors.push_back(temp);
	}
	gen_err = sum_error / oob_errors.size(); //最后的泛化误差
	return gen_err;
}

double RandomForest::avgOobErrorOfTree()
{
	if (oob_err > 0.0)
		return oob_err;
	double sum_error = 0.0;
	for_each(rf.begin(), rf.end(), [&sum_error](const std::shared_ptr<RandomTree>& rt)
	{
		sum_error += rt->oobError();
	});
	oob_err = sum_error / rf.size();
	return oob_err;
}

//特征f的重要性度量按照如下方式计算：对于每棵使用了特征f的随机树，随机排列（也可以加噪声）
//扰乱该树所用训练数据集中特征f的值，然后再计算袋外误差，接着减去正常情况下的袋外误差，即
//该树中特征f的重要性度量，求得所有这样的树中特征f重要性的均值即为其最终重要性
const std::vector<std::pair<size_t, double>>& RandomForest::FeatureImportance()
{
	assert(prf->calc_fte_importance);
	if (!fis.empty())
		return fis;
	for (size_t i = 0; i != prf->D; ++i)
	{
		double sum_ooberror_of_trees_used_fte_i = 0.0,
			sum_pererror_of_trees_used_fte_i = 0.0;
		for (size_t j = 0; j != rf.size(); ++j)
		{
			if (rf[i]->usedFeature(i)) //如果该随机树建树过程中使用了该特征
			{
				sum_ooberror_of_trees_used_fte_i += rf[i]->oobError();
				sum_pererror_of_trees_used_fte_i += rf[i]->permutedOobError(i);
			}
		}
		fis.emplace_back(i, sum_pererror_of_trees_used_fte_i - sum_ooberror_of_trees_used_fte_i);
	}
	//按照特征重要性非升序排列
	std::sort(fis.begin(), fis.end(), [](const std::pair<size_t, double>& lhs,
		const std::pair<size_t, double>& rhs)->bool
	{
		return lhs.second > rhs.second;
	});
	return fis;
}

std::vector<sample> RandomForest::loadData(const std::string& filename, size_t D, size_t& size)
{
	std::ifstream infile(filename);
	assert(infile);
	std::vector<sample> data;
	data.reserve(size);
	double fte_val = 0.0;
	int cls_val = 0;
	std::vector<double> x;
	while (true)
	{
		x.reserve(D);
		while (infile >> fte_val)
		{
			x.push_back(fte_val);
			if (x.size() == D)
			{
				infile >> cls_val; //已读取完一个样本
				data.emplace_back(std::move(x), cls_val);
				break;
			}
		}
		if (!infile) break;
	}
	size = data.size(); //返回加载的样本数
	return std::move(data);
}