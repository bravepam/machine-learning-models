#ifndef RANDOM_TREE_H
#define RANDOM_TREE_H

#include"FeatureSelect.h"
#include"util.h"
#include<vector>

//随机决策树
class RandomTree
{
private:
	struct node;
private:
	node* root;
	TreeDataSet tree_data; //所用数据集
	double oob_err = 0.0; //袋外误差
	double oob_permuted_err = 0.0; //随机排列某一特征所有值后的袋外误差
	size_t height = 0;
	InfoGain ig;
	const RFParams* const prf;
private:
	//随机选取特征
	static std::vector<size_t> randomSelectFeatures(size_t, size_t);
	void create(std::vector<size_t>&, node*&);
	void clear(node*);
public:
	RandomTree(const RFParams* const p) :root(nullptr), tree_data(p), ig(&tree_data),
		prf(p){}
	void create()
	{
		tree_data.bagging(prf->train_set.size());
		tree_data.getOobData();
		create(tree_data.train_data, root);
	}
	void clear()
	{
		if (root)
		{
			clear(root);
			root = nullptr;
		}
	}
	bool empty()const
	{
		return root == nullptr;
	}
	int predict(const sample&)const;
	void computeOobErr();
	//随机排列扰动后的袋外误差
	void permuteFeature(size_t);
};

#endif