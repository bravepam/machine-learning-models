//包含了一个决策树的模板，根据不同的特征选择准则而实例化不同的决策树
//用于实例化ID3算法和C4.5算法生成的决策树


#ifndef ID3ANDC45DTREE_H
#define ID3ANDC45DTREE_H

#include<vector>
#include"FeatureSelectionCriterion.h"

using namespace std;

struct node
{//树节点类型
	//特征ID，如果是内节点，那么该节点是按照该值所代表的特征进行的分割；
	//如果是叶子节点，则没有分割了，此时该值为-1.
	int fte_id;

	//该节点所属的类别，如果是内节点，则没有类别所属，该值为-1；
	//如果是叶子节点，该值则表示所属类别
	int cls_value;

	//如果该节点是内节点，则会按照特征fte_id进行分割，根据特征值的不同，将会产生
	//若干个孩子，孩子数由特征值个数决定；如果是叶子节点，则为空。
	vector<node*> child;
	double loss;//如果剪掉该节点的所有孩子后的信息损失，不剪枝时不用
	node(int _fte_id, int _cls_value) :fte_id(_fte_id), cls_value(_cls_value), 
		child(),loss(0.0){}
};

template<typename ctr>
class Dtree
{//决策树结构，模板参数表示特征选择准则类型
private:
	node *root = nullptr;
	ctr *pctr;
	vector<int> data_id;
	double e;//阈值，如果准则计算出的信息值低于该阈值，则不再分割节点，将其作为叶子节点
private:
	node* create(vector<int>&,bool,double);
	void clear(node*);
public:
	Dtree(const vector<sample<int>> &d, const vector<int> &fv, int cn, double _e) :
		pctr(new ctr(d, fv, cn)),e(_e)
	{
		for (int i = 0; i != d.size(); ++i)
			data_id.push_back(i);
	}
	void create(bool prune = false,double a = 0.0)
	{//创建决策树，默认是不剪枝的，a用于控制预测误差和树复杂度平衡的参数
		root = create(data_id,prune,a);
	}
	int compute(const vector<int>&);//查找该样本点所属的类别
	void clear()
	{//清空树
		clear(root);
		delete pctr;
		root = nullptr;
	}
	~Dtree()
	{
		clear();
	}
};

template <typename ctr>
node* Dtree<ctr>::create(vector<int> &data_id,bool prune,double a)
{
	pair<bool, int> ret = pctr->checkData(data_id);//检查数据集

	//将该节点作为叶子而带来的损失
	double temp_loss = data_id.size() * 1.0 * pctr->entropy(data_id, -1) + a;
	if (ret.first || !pctr->hasFeature())
	{//如果是同一类别或者不再有特征可用
		node *p = new node(-1, ret.second);//则以占多数的类别值创建叶子节点
		if (prune) p->loss = temp_loss;//如果剪枝，则设置损失值，下同
		return p;
	}
	vector<vector<int>> splited_data;

	//否则根据当前数据集、可选特征集等选择最优分割，其中splited_data将存储分割后的数据子集
	//返回值为最优分割特征ID和所得到的信息值
	pair<double, int> slt = pctr->select(data_id, splited_data);
	if (slt.first < e)
	{//如果信息值较低
		node *p = new node(-1, ret.second);//则不再分割，以占多数的类别值创建叶子节点
		if (prune) p->loss = temp_loss;
		return p;
	}

	//否则创建内节点
	node *p = new node(slt.second, -1);
	if (prune) p->loss = temp_loss;
	p->child.resize(pctr->getSpecificFeatureValues(slt.second));//根据特征值个数分配孩子
	pctr->zeroSpecificFeatureValues(slt.second);//将特征值个数置0表示该特征不再可用
	for (int i = 0; i != splited_data.size(); ++i)
		p->child[i] = create(splited_data[i],prune,a);//有几个特征值，就会有几个数据子集，逐一创建子树

	if (prune)
	{//如要剪枝
		double sum_loss = 0.0;
		for (int j = 0; j != p->child.size(); ++j)
			sum_loss += p->child[j]->loss;//累加孩子节点的损失
		if (sum_loss >= p->loss)
		{//如果不剪枝的损失超过了剪枝的，则剪枝
			for (int k = 0; k != p->child.size(); ++k)
				delete p->child[k];//那么要进行剪枝
			p->child.clear();
			p->fte_id = -1, p->cls_value = ret.second;//修改该节点为叶子节点
		}
		else p->loss = sum_loss;//否则不剪，并更新损失为不剪枝的损失
	}
	return p;
}

template <typename ctr>
int Dtree<ctr>::compute(const vector<int> &data)
{//计算样本点所属的类别
	node *curr = root;
	while (curr->fte_id != -1)//根据分割当前节点的特征ID获得样本特征值，判断该往哪个孩子节点搜寻
		curr = curr->child[data[curr->fte_id]];
	return curr->cls_value;//直到叶子，返回该叶子类别即可
}

template <typename ctr>
void Dtree<ctr>::clear(node *r)
{
	for (int i = 0; i != r->child.size(); ++i)
		clear(r->child[i]);
	//cout << r << ' ' << r->fte_id << ' ' << r->cls_value << endl;
	delete r;
}

//以信息增益作为选择准则的ID3算法生成的决策树
typedef Dtree<InfoGain>				ID3_Dtree;

//以信息增益比作为选择准则的C4.5算法生成的决策树
typedef Dtree<InfoGainRatio>		C45_Dtree;

#endif