
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<iostream>
#include<sstream>
#include<fstream>
#include<string>
#include<unordered_map>
#include<set>
#include<unordered_set>
#include<algorithm>
#include<vector>
#include<functional>

using namespace std;

//FP树，用于实现FP-growth算法，高效挖掘频繁集

template <typename Key>
struct t_node
{//基本节点
	Key key;
	int count;
	t_node(Key v, int cnt) :key(v), count(cnt){}
};

template <typename Comp>
struct nodeComparer
{//t_node节点比较器
	template <typename Key>
	bool operator()(const t_node<Key>&lhs, const t_node<Key> &rhs)const
	{
		if (lhs.count == rhs.count) //如果计数值相等
			return Comp()(lhs.key, rhs.key); //则比较键值
		else return Comp()(lhs.count, rhs.count); //否则
	}
};

template <typename Key>
struct t_treeNode:public t_node<Key>
{//树节点类型，继承于t_node
	t_treeNode *parent;
	t_treeNode *next = nullptr;
	unordered_map<char, t_treeNode*> children; //孩子列表，用字典方便访问
	t_treeNode(Key v, int cnt, t_treeNode *par = nullptr) :t_node<Key>(v, cnt), parent(par), children(){}
};

template <typename Key>
struct t_headNode
{//头指针列表节点
	int count;
	t_treeNode<Key> *next = nullptr;//链接树中具有相同键值的树节点
	t_headNode(int cnt) :count(cnt){}
};

template <typename Key>
class FPTree
{//FP树
	typedef t_node<Key>						node;
	typedef t_headNode<Key>					headNode;
	typedef t_treeNode<Key>					treeNode;
private:
	unordered_map<Key, headNode*> head; //头指针列表
	treeNode *root;
	int eps; //计数阈值
private:
	void updateTree(const vector<Key>&, int, treeNode*, int); 
	void updateHead(treeNode*, treeNode*);
	void mineTree(string&, vector<string>&);
	void findPrefixPath(Key,unordered_map<string,int>&);
	void ascendTree(treeNode*, string&)const;
	void print(treeNode*, int)const;
public:
	FPTree(int e) :root(new treeNode(Key(), 0)), eps(e){}
	unordered_map<string, int> loadData(istream&)const;
	void create(const unordered_map<string, int>&);
	vector<string> mineTree()
	{
		string prefix("(");
		vector<string> frequent;
		mineTree(prefix, frequent);
		return frequent;
	}
	bool empty()const
	{
		return root->children.empty();
	}
	void clear();
	void print(int depth = 1)const
	{
		print(root, depth);
	}
	~FPTree()
	{
		clear();
		delete root;
	}
};

//根据数据集构建FP树
template <typename Key> 
void FPTree<Key>::create(const unordered_map<string, int> &data)
{
	unordered_set<Key> keys; //记录所有的键，即每个记录项
	for (auto iter1 = data.begin(); iter1 != data.end(); ++iter1) //第一次扫描整个数据集
	{//每条记录都是字符串对象，采用空格分开
		istringstream stream(iter1->first); //初始化一个字符串输入流
		Key k;
		while (stream >> k)
		{//逐一获得记录项
			if (head.find(k) == head.end()) //如果不存在于头指针列表，则插入
				head.insert(make_pair(k, new headNode(iter1->second)));
			else //否则只需要累加计数值即可
				head[k]->count += iter1->second;
			keys.insert(k);
		}
	}

	for (auto iter2 = keys.begin(); iter2 != keys.end(); ++iter2)
	{//扫描每个记录项
		if (head[*iter2]->count < eps) //如果计数值较低，则删除之
			head.erase(*iter2);
	}

	if (head.empty()) return; //如果没有出现次数超过阈值的记录项，则退出
	for (auto iter3 = data.begin(); iter3 != data.end(); ++iter3)
	{//第二次扫描整个数据集
		//对于每条记录，根据记录项和出现的总次数构造节点，然后插入到有序集合
		multiset<node,nodeComparer<greater<int>>> key_count; 
		istringstream stream(iter3->first);
		Key k;
		while (stream >> k)
		{//对于该条记录中的每个记录项
			if (head.find(k) != head.end()) //只有出现次数较高
				key_count.insert(node(k, head[k]->count)); //才会被插入到有序集合中
		}
		if (!key_count.empty())
		{//若最终该条记录中有出现次数较多的项
			vector<Key> order;
			//则按照出现次数从高到低的顺序提取出来
			for_each(key_count.begin(), key_count.end(), [&](const node &arg){order.push_back(arg.key); });
			updateTree(order, 0, root, iter3->second);//用于更新树
		}
	}
}

//根据order中元素的顺序，从节点r开始向下更新树
template <typename Key>
void FPTree<Key>::updateTree(const vector<Key> &order, int index, treeNode *r, int count)
{
	if (r->children.find(order[index]) != r->children.end()) //如果当前项是r的孩子
		r->children[order[index]]->count += count;//则直接更新计数值
	else
	{
		treeNode *p = new treeNode(order[index], count, r); //否则构造节点
		r->children.insert(make_pair(order[index], p)); //插入到孩子列表
		if (head[order[index]]->next)//同时更新头指针列表
			updateHead(head[order[index]]->next, p);
		else
			head[order[index]]->next = p;
	}
	if ((static_cast<int>(order.size()) - index) > 1) //如果order总还有余项
		updateTree(order, index + 1, r->children[order[index]], count);//则继续递归向下更新树
}

//更新头指针列表，实际上就是在单链表尾端插入一个节点
template <typename Key>
void FPTree<Key>::updateHead(treeNode *h, treeNode *cur)
{
	while (h->next)
		h = h->next;
	h->next = cur;
}

//从构造好的树中挖掘频繁项集
template <typename Key>
void FPTree<Key>::mineTree(string &prefix, vector<string> &frequent)
{
	//头指针列表head中存储的实际是单频繁项。根据项值和计数值构造节点，插入到有序集合
	//不同于建树时，此时是按照计数值从低到高顺序排序
	set <node, nodeComparer<less<int>>> key_count;
	for (auto iter = head.begin(); iter != head.end(); ++iter)
		key_count.insert(node(iter->first, iter->second->count));
	for (auto iter = key_count.begin(); iter != key_count.end(); ++iter)
	{//对于每一个单频繁项
		if (string(typeid(iter->key).name()) == string("int"))
		{//如果项值类型是int
			char ch[12];
			prefix += _itoa(iter->key, ch, 10); //则先要转为string类型，然后插入到前缀中
		}
		else prefix.push_back(iter->key); //如果是其他（实际上只会是char）
		frequent.push_back(prefix + ')'); //发现一个频繁项
		unordered_map<string, int> subdata;

		//以当前单频繁项为尾，在树中上溯挖掘条件基，也就是在已出现prefix的情况下挖掘记录
		findPrefixPath(iter->key, subdata); 
		FPTree<Key> subtree(eps);
		subtree.create(subdata); //根据挖掘到的记录构造的子数据集创建子FP树，以用于挖掘更复杂的频繁项
		if (!subtree.empty())
		{//如果树不空，即存在更复杂的频繁项
			prefix += ' ';//用于分隔记录项
			subtree.mineTree(prefix, frequent); //继续递归挖掘
			prefix.pop_back(); //删除空格
		}
		int index = prefix.rfind(' ');
		if (index == string::npos) prefix.resize(1); //删除iter->key
		else prefix.resize(index + 1);
	}
}

//挖掘条件基，实际上是（条件）路径
template <typename Key>
void FPTree<Key>::findPrefixPath(Key tail, unordered_map<string, int> &paths)
{
	treeNode *first = head[tail]->next; //键为tail节点构成一条单链表
	while (first)
	{//对于当前键为tail的节点
		string prefix;
		ascendTree(first, prefix);//由此上溯到根获得条件基
		if (!prefix.empty())//如果条件基存在
			paths.insert(make_pair(prefix, first->count)); 
		first = first->next; //下一个节点
	}
}

//自当前节点开始上溯到根，并顺序记下节点键值，构成一个条件基
template <typename Key>
void FPTree<Key>::ascendTree(treeNode *curr, string &prefix)const
{
	treeNode *par = curr->parent; //条件基不包括当前节点
	while (par->parent)
	{
		if (string(typeid(par->key).name()) == string("int"))
		{//如果键类型为int
			char ch[12];
			prefix += _itoa(par->key, ch, 10);
		}
		else prefix.push_back(par->key); //如果为char
		prefix.push_back(' '); //分隔记录项
		par = par->parent;
	}
	if (!prefix.empty()) //删除末尾的空格
		prefix.resize(prefix.size() - 1);
}

//打印FP树
template <typename Key>
void FPTree<Key>::print(treeNode *r, int depth = 1)const
{
	string right_indent(depth, ' '); //用于缩进，表现出树的层次结构
	cout << right_indent << r->key << ":" << r->count << endl;
	for (auto iter = r->children.begin(); iter != r->children.end(); ++iter)
		print(iter->second, depth + 1);
}

//清空树
template <typename Key>
void FPTree<Key>::clear()
{
	for (auto iter = head.begin(); iter != head.end(); ++iter)
	{//由于头指针列表是一个单链表集合，每条链表都是键值相同的节点的集合，因而可以通过
		//释放所有链表来达到清空树的目的
		treeNode *first = iter->second->next, *r = nullptr;
		while (first)
		{
			r = first;
			first = first->next;
			//cout << "delete tree " << r->key << ' ' << r->count << endl;
			delete r;
		}
		//cout << "delete head " << iter->first << ' ' << iter->second->count << endl;
		delete iter->second; //释放链表头
	}
}

//加载数据集
template <typename Key>
unordered_map<string, int> FPTree<Key>::loadData(istream &infile)const
{
	unordered_map<string, int> data;
	string line;
	while (getline(infile, line))
		data.insert(make_pair(line, 1)); //将数据集转化为字典，这样做主要是为了方便挖掘频繁集时递归构造子FP树
	return data;
}

int main()
{
	//FPTree<int> tree(100000);
	//大数据集，百万
	//ifstream infile("C:\\Users\\png\\Desktop\\machinelearninginaction\\Ch12\\kosarak.dat");
	FPTree<char> tree(3);
	ifstream infile("data.txt");
	unordered_map<string, int> data = tree.loadData(infile);
	tree.create(data);
	tree.print();
	vector<string> frequent = tree.mineTree();
	for (int i = 0; i != frequent.size(); ++i)
		cout << frequent[i] << endl;
	getchar();
	return 0;
}