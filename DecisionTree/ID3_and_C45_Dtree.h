//ID3算法和C4.5算法生成的决策树

#ifndef _ID3_C45_GINI_DTREE_H
#define _ID3_C45_GINI_DTREE_H

#include<vector>
#include"FeatureSelectionCriterion.h"

using namespace std;

struct basenode
{//ID3树节点类型
	//特征ID，如果是内节点，那么是按照该值所代表的特征进行的分割；
	//如果是叶子节点，则没有分割了，此时该值为-1.
	int fte_id;

	//该节点所属的类别，如果是内节点，则没有类别所属，该值为-1；
	//如果是叶子节点，该值则表示所属类别
	int cls_value;

	//如果该节点是内节点，则会按照特征fte_id进行分割，根据特征值的不同，将会产生
	//若干个孩子，孩子数由特征值个数决定；如果是叶子节点，则为空。
	vector<basenode*> child;
	basenode(int _fte_id, int _cls_value) :fte_id(_fte_id), cls_value(_cls_value), child(){}
	virtual ~basenode(){}
};

struct C45_node :public basenode
{//C45树节点类型，继承自ID3树节点，增加了一个信息损失域
	double loss;//如果剪掉该节点的所有孩子后的信息损失
	C45_node(int _fte_id, int _cls_value) :basenode(_fte_id, _cls_value), loss(0.0){}
	virtual ~C45_node(){}
};

class ID3tree
{//ID3决策树
protected:
	vector<int> data_id;
	basenode *root = nullptr;
	static InfoGain *pig;//特征选择准则
	double eps;//阈值，如果准则计算出的信息增益（比）低于该阈值，则不再分割节点，将其作为叶子节点
private:
	basenode* create(const vector<int>&);
	void clear(basenode*);
public:
	ID3tree(int samples_num, double _e) :data_id(samples_num), eps(_e)
	{
		for (int i = 0; i != data_id.size(); ++i)
			data_id[i] = i;
	}
	static void initCriterion(const vector<sample<int>> &d, const vector<int> &fv, int cn)
	{
		pig = new InfoGain(d, fv, cn);
	}
	void create()
	{
		root = create(data_id);
	}
	int compute(const vector<int>&);//查找该样本点所属的类别
	void clear()
	{//清空树
		clear(root);
		delete pig;
		root = nullptr;
	}
	virtual ~ID3tree()
	{
		clear();
	}
};

InfoGain *ID3tree::pig;//定义该准则

basenode* ID3tree::create(const vector<int> &data_id)
{
	vector<int> cls_count;
	pair<bool, int> ret = pig->checkData(data_id, cls_count);//检查数据集
	if (ret.first || !pig->hasFeature())
	{//如果是同一类别或者不再有特征可用
		basenode *p = new basenode(-1, ret.second);//则以占多数的类别值创建叶子节点
		return p;
	}
	vector<vector<int>> splited_data;
	//否则根据当前数据集、可选特征集等选择最优分割，其中splited_data将存储分割后的数据子集
	//返回值为最优信息增益和最优分割特征ID
	pair<double, int> slt = pig->select(data_id, splited_data);
	if (slt.first < eps)
	{//如果信息增益较低
		basenode *p = new basenode(-1, ret.second);//则不再分割，以占多数的类别值创建叶子节点
		return p;
	}
	//否则创建内节点
	basenode *p = new basenode(slt.second, -1);
	p->child.resize(pig->getSpecificFeatureValues(slt.second));//根据特征值个数分配孩子
	pig->zeroSpecificFeatureValues(slt.second);//将特征值个数置0表示该特征不再可用
	for (int i = 0; i != splited_data.size(); ++i)
		p->child[i] = create(splited_data[i]);//有几个特征值，就会有几个数据子集，逐一创建子树
	return p;
}

int ID3tree::compute(const vector<int> &data)
{//计算样本点所属的类别
	basenode *curr = root;
	while (curr->fte_id != -1)//根据分割当前节点的特征ID获得样本特征值，判断该往哪个孩子节点搜寻
		curr = curr->child[data[curr->fte_id]];
	return curr->cls_value;//直到叶子，返回该叶子类别即可
}

void ID3tree::clear(basenode *r)
{
	for (int i = 0; i != r->child.size(); ++i)
		clear(r->child[i]);
	//cout << r << ' ' << r->fte_id << ' ' << r->cls_value << endl;
	delete r;
}

class C45tree:public ID3tree
{//C45决策树，继承自ID3树
private:
	double a;//a用于控制预测误差和树复杂度平衡的参数
	basenode* create(const vector<int>&);
public:
	C45tree(int samples_num, double _e, double _a) :ID3tree(samples_num, _e), a(_a){}

	//重定义以下两个函数
	static void initCriterion(const vector<sample<int>> &d, const vector<int> &fv, int cn)
	{
		pig = new InfoGainRatio(d, fv, cn);
	}
	void create()
	{
		root = create(data_id);
	}
	virtual ~C45tree()
	{
		
	}
};

basenode* C45tree::create(const vector<int> &data_id)
{
	vector<int> cls_count;
	pair<bool, int> ret = pig->checkData(data_id, cls_count);//检查数据集
	//将该节点作为叶子而带来的损失
	double temp_loss = data_id.size() * 1.0 * pig->entropyAux(data_id.size(), cls_count) + a;
	if (ret.first || !pig->hasFeature())
	{//如果是同一类别或者不再有特征可用
		C45_node *p = new C45_node(-1, ret.second);//则以占多数的类别值创建叶子节点
		p->loss = temp_loss;
		return p;
	}
	vector<vector<int>> splited_data;

	//否则根据当前数据集、可选特征集等选择最优分割，其中splited_data将存储分割后的数据子集
	//返回值为最优信息增益比和最优分割特征ID
	pair<double, int> slt = pig->select(data_id, splited_data);
	if (slt.first < eps)
	{//如果信息增益比较低
		C45_node *p = new C45_node(-1, ret.second);//则不再分割，以占多数的类别值创建叶子节点
		p->loss = temp_loss;
		return p;
	}

	//否则创建内节点
	C45_node *p = new C45_node(slt.second, -1);
	p->loss = temp_loss;
	p->child.resize(pig->getSpecificFeatureValues(slt.second));//根据特征值个数分配孩子
	pig->zeroSpecificFeatureValues(slt.second);//将特征值个数置0表示该特征不再可用
	for (int i = 0; i != splited_data.size(); ++i)
		p->child[i] = create(splited_data[i]);//有几个特征值，就会有几个数据子集，逐一创建子树

	double sum_loss = 0.0;
	for (int j = 0; j != p->child.size(); ++j)
		sum_loss += static_cast<C45_node*>(p->child[j])->loss;//累加孩子节点的损失
	if (sum_loss >= p->loss)
	{//如果不剪枝的损失超过了剪枝的，则剪枝
		for (int k = 0; k != p->child.size(); ++k)
			delete p->child[k];//那么要进行剪枝
		p->child.clear();
		p->fte_id = -1, p->cls_value = ret.second;//修改该节点为叶子节点
	}
	else p->loss = sum_loss;//否则不剪，并更新损失为不剪枝的损失
	return p;
}

#endif