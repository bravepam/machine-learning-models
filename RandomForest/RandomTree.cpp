#include"RandomTree.h"
#include"util.h"
#include<vector>
#include<utility>
#include<cassert>
#include<algorithm>

struct RandomTree::node
{
	int fte_id_or_cls; //内节点，则是分割特征ID；叶子节点，则是类别值
	double split_value;
	bool leaf;
	node* less = nullptr;
	node* greater = nullptr;
	node(int fc, double fv, bool l = false) :fte_id_or_cls(fc), split_value(fv), leaf(l){}
};

std::vector<size_t> RandomTree::randomSelectFeatures(size_t slt_num, size_t feat_num)
{
	std::vector<size_t> features;
	samplingNoReplacement(slt_num, feat_num, features);
	return std::move(features);
}

void RandomTree::create(std::vector<size_t>& data_id, node*& r)
{
	ig.clear(); //清空前一次create残留的数据信息
	//获得随机特征集合
	std::vector<size_t> features = randomSelectFeatures(prf->F, prf->D);
	ig.setFeaturesAndData(features, data_id);
	std::hash_map<int, size_t> cls_count;
	//检查当前所用数据集，返回值{{是否同一类别，占比最大的类别}，类别统计信息}
	std::pair<bool, size_t> ret = ig.checkData(cls_count);

	++height;
	//四个条件终止继续生成子树：1、数据集为同一类别；2、数据集太小；3、熵值太小，即
	//基本上位同一类别；4、树已经足够深，即够茂盛
	if (ret.first || 
		(prf->tc.num > 0 && data_id.size() <= prf->tc.num) ||
		(prf->tc.eps > 0.0 && ig.entropyAux(cls_count, data_id.size()) <= prf->tc.eps) ||
		(prf->tc.depth > 0 && height >= prf->tc.depth))
	{
		r = new node(ret.second, 0.0, true); //以占比最大的类别作为该叶子的类别值
		return;
	}

	std::vector<std::vector<size_t>> splited_data_id(2);
	//否则，按照信息增益准则选取最佳分割。返回值{{最佳分割特征，最佳分割点值}，分割后的两个数据集}
	std::pair<size_t, double> slt = ig.select(splited_data_id);
	used_features[slt.first] = true;
	r = new node(slt.first, slt.second);
	//继续构建树
	create(splited_data_id[0], r->less);
	create(splited_data_id[1], r->greater);
}

void RandomTree::clear(node* r)
{
	if (r->less)
		clear(r->less);
	if (r->greater)
		clear(r->greater);
	delete r;
}

int RandomTree::predict(const sample& s)const
{
	assert(!empty());
	node* cur = root;
	while (!cur->leaf)
	{
		if (s.x[cur->fte_id_or_cls] < cur->split_value)
			cur = cur->less;
		else
			cur = cur->greater;
	}
	return cur->fte_id_or_cls;
}

double RandomTree::oobError()
{
	if (oob_err > 0.0)
		return oob_err;
	const auto& raw_data = prf->train_set;
	size_t err = 0;
	for (size_t i = 0; i != tree_data.oob.size(); ++i)
	{
		const int pred = predict(raw_data[tree_data.oob[i]]);
		err += static_cast<size_t>(pred != raw_data[tree_data.oob[i]].y);
	}
	oob_err = err * 1.0 / tree_data.oob.size();
	return oob_err;
}

double RandomTree::permutedOobError(size_t which)
{
	const auto& raw_data = prf->train_set;
	const size_t size = tree_data.oob.size();
	std::vector<double> which_value;
	which_value.reserve(size);
	for (size_t i = 0; i != size; ++i) //获得数据集中which特征对应的所有值
		which_value.push_back(raw_data[tree_data.oob[i]].x[which]);
	//随机排列
	std::random_shuffle(which_value.begin(), which_value.end());

	size_t err = 0;
	for (size_t i = 0; i != size; ++i)
	{
		node* cur = root;
		while (!cur->leaf)
		{
			if (cur->fte_id_or_cls == which)
			{//如果当前树节点的分割特征是刚刚随机排列扰动的特征
				//则用扰动后的值判断
				if (which_value[i] < cur->split_value)
					cur = cur->less;
				else cur = cur->greater;
			}
			else
			{//否则正常判断
				if (raw_data[tree_data.oob[i]].x[cur->fte_id_or_cls] < cur->split_value)
					cur = cur->less;
				else
					cur = cur->greater;
			}
		}
		err += static_cast<size_t>(raw_data[tree_data.oob[i]].y != cur->fte_id_or_cls);
	}
	const double oob_permuted_err = err * 1.0 / size;
	return oob_permuted_err;
}