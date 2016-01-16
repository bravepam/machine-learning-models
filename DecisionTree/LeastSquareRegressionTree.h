
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


//CART回归树，采用最小二乘作为特征选择准则，生成的树成为最小二乘回归树

#ifndef LEASTSQUAREREGRESSIONTREE_H
#define LEASTSQUAREREGRESSIONTREE_H

#include<vector>
#include"FeatureSelectionCriterion.h"

using namespace std;

struct node
{//节点
	int split_fte;//如果是内节点，则是分割特征，如果是叶子节点，则是-1
	double value;//在特征fte_id下，得到最优分割时所取的值
	node *small = nullptr;//小于或等于该值的子树
	node *large = nullptr;//大于该值的子树
	node(int sf, double sv) :split_fte(sf), value(sv){}
};

class Rtree
{//回归树
private:
	node *root = nullptr;
	LeastSquareError lse;//采用最小二乘作为特征选择准则
	int e;//阈值，样本数的最小值
	vector<int> data_id;//样本集ID集合
private:
	void create(vector<int>&, node*&);
	void clear(node*);
public:
	Rtree(const vector<sample<double>> &d, int _e) :e(_e), lse(d)
	{
		for (int i = 0; i != d.size(); ++i)
			data_id.push_back(i);
	}
	void create()
	{
		create(data_id, root);
	}
	void clear()
	{
		clear(root);
		root = nullptr;
	}
	double compute(vector<double>&);//计算该样本点的值
	~Rtree()
	{
		clear();
	}
};

void Rtree::create(vector<int> &data_id, node *&r)
{//创建回归树
	if ((int)data_id.size() <= e)
	{//如果样本集太小
		r = new node(-1, lse.average(data_id));//则不再分割，以样本集值的均值创建叶子节点
		return;
	}

	//否则分割节点
	vector<vector<int>> splited_data_id;
	pair<double, int> ret = lse.select(data_id, splited_data_id);//找到最优分割特征和分割点
	r = new node(ret.second, ret.first);//以其创建节点
	create(splited_data_id[0], r->small);//递归创建“不大于”子树
	create(splited_data_id[1], r->large);//递归创建“大于”子树
}

double Rtree::compute(vector<double> &x)
{//计算样本点x的回归值
	node *curr = root;
	while (curr->split_fte != -1)
	{
		if (x[curr->split_fte] <= curr->value)
			curr = curr->small;
		else curr = curr->large;
	}
	return curr->value;
}

void Rtree::clear(node *r)
{//清空树
	if (r == nullptr) return;
	clear(r->small);
	clear(r->large);
	delete r;
}

#endif