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
	Gini *pgini;//采用基尼指数作为特征选择准则
	vector<int> data_id;//样本集id号集合
	int e;//阈值，样本最小数目
	double g;//阈值，基尼指数最小值，小于这两个阈值任意一个，则表示不再分割了
private:
	void create(vector<int>&, node*&);
	void clear(node*);
public:
	Cart(const vector<sample<int>> &d, const vector<int> &fv, int cn, double _g, int _e) :
		data_id(d.size()), pgini(new Gini(d, fv, cn)), g(_g), e(_e)
	{
		for (int i = 0; i != data_id.size(); ++i)
			data_id[i] = i;
	}
	void create()
	{
		create(data_id, root);
	}
	void clear()
	{
		clear(root);
		delete pgini;
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
	vector<int> cls_count;
	pair<bool, int> ret = pgini->checkData(data_id, cls_count);//检查数据集
	if (ret.first || !pgini->hasFeature() || pgini->entropyAux(data_id.size(), cls_count) <= g)
	{//如果数据是同一类，或者不再有可用特征，或者该数据集的基尼指数小于阈值
		r = new node(-1, ret.second);//则创建叶子节点，以占多数的类别为该节点类别
		return;//结束
	}
	//否则，将进行分割
	vector<vector<int>> splited_data_id;//记录分割后的数据子集集合
	//选择最优分割特征及其取值
	pair<double, int> slt = pgini->select(data_id, splited_data_id);
	r = new node((int)slt.first, slt.second);//特征号只会是整数
	pgini->zeroSpecificFeatureValues((int)slt.first);//将该特征取值个数置0，表示不再可用
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

/*关于Cart剪枝算法，根据《统计学方法》的介绍，总结出一种详细可行的方法如下，没实现：
 *1、递归向下，在建树的时候，每到达一个新节点，计算出假设以该点为叶子节点的预测误差C_t，
 *   树创建完毕，那么各个节点作为叶子节点的预测误差也计算出来了，这个误差就是将该节点的
 *   子树剪掉所得到的预测误差，即我们在建树的过程中就可以将剪枝后的预测误差给计算出来；
 *2、自下而上的访问每一个内节点，该节点剪枝后的预测误差在之前已经计算出来；而不剪枝，即
 *   现在的预测误差就是以其为根的子树的孩子节点的预测误差之和，在访问的最初，也就是两个
 *   孩子叶节点的预测误差之和，计算出两者的预测误差的差的绝对值与叶子节点数目差（最初时
 *   为1，不剪有两个叶子，剪了有一个）的绝对值的比，这就是g_t，记下，并将其存入一个集合。然后向
 *   一层返回，返回的数据包括，叶子节点个数和不剪枝的预测误差（所有叶子节点的预测误差之和），
 *  以供上层节点使用，并计算出g_t，直至根节点计算完毕；
 *3、开始剪枝，
 *   (1) 初始，a0 = 0，此时整棵树即为最优，记为T0；
 *   (2) 从集合g_t中选出最小元素，记为a1，自上而下遍历T0，如果遇到某一节点的g_t值等于a1,则
 *       将其剪枝，以多数类作为该节点类别，得到的树记为T1，返回;
 *   (3) 继续从g_t集合中选取第二小的元素，记为a2，自上而下遍历T1，......，记为T2，返回；
 *   (4) 如此反复，直到得到的树Tn是一颗单节点树，只含有树根，至此剪枝完毕。
 *  此时[ai,ai+1)对应数Ti，i = 0,1,2...n，其中a0 = 0.最后采用交叉验证得到最优子树Ta和a值。
 */