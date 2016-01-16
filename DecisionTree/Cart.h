
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


//CART分类树，采用基尼指数作为特征选择准则

#ifndef _CART_H
#define _CART_H

#include<vector>
#include"FeatureSelectionCriterion.h"

using namespace std;

struct node
{//节点
	int fte_id;//如果是内节点，那么是该节点的分割特征，是叶子则为-1
	int fte_cls_value;//如果是内节点，则是特征fte_id的取值，否则是类别取值
	node *yes = nullptr;//按特征fte_id的fte_cls_value值分割，等于该值所构成的子树
	node *no = nullptr;//不等于该值所构成的子树
	node(int fi, int fcv) :fte_id(fi), fte_cls_value(fcv){}
};

class Cart
{//决策树
private:
	node *root;
	Gini gini;//采用基尼指数作为特征选择准则
	vector<int> data_id;//样本集id号集合
	int e;//阈值，样本最小数目
	double g;//阈值，基尼指数最小值，小于这两个阈值任意一个，则表示不再分割了
private:
	void create(vector<int>&, node*&);
	void clear(node*);
public:
	Cart(const vector<sample<int>> &d, const vector<int> &fv, int cn, double _g, int _e) :
		gini(d, fv, cn), g(_g), e(_e)
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
	~Cart()
	{
		clear();
	}
	int compute(vector<int>&);//计算样本点的类别
};

void Cart::create(vector<int> &data_id, node *&r)
{//创建Cart决策树
	pair<bool, int> ret = gini.checkData(data_id);//检查数据集
	if (ret.first || !gini.hasFeature() || gini.entropy(data_id, -1) <= g)
	{//如果数据是同一类，或者不再有可用特征，或者该数据集的基尼指数小于阈值
		r = new node(-1, ret.second);//则创建叶子节点，以占多数的类别为该节点类别
		return;//结束
	}
	//否则，将进行分割
	vector<vector<int>> splited_data_id(2);//记录分割后的数据子集集合
	pair<double, int> slt = gini.select(data_id, splited_data_id);//选择最优分割特征及其取值
	r = new node((int)slt.first, slt.second);//特征号只会是整数
	gini.zeroSpecificFeatureValues((int)slt.first);//将该特征取值个数置0，表示不再可用
	create(splited_data_id[0], r->yes);//递归创建“是”子树
	create(splited_data_id[1], r->no);//递归创建“否”子树
}

void Cart::clear(node *r)
{//清空树
	if (r == nullptr) return;
	clear(r->yes);
	clear(r->no);
	delete r;
}

int Cart::compute(vector<int> &x)
{//计算x的类别
	node *curr = root;
	while (curr->fte_id != -1)
	{
		if (x[curr->fte_id] == curr->fte_cls_value)
			curr = curr->yes;
		else curr = curr->no;
	}
	return curr->fte_cls_value;
}

#endif