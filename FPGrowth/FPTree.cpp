#include<iostream>
#include<sstream>
#include<string>
#include<unordered_map>
#include<algorithm>
#include<vector>

using namespace std;

//FP树，用于实现FP-growth算法，高效挖掘频繁集

struct node
{//基本节点
	string key;
	int count;
	node(const string& v, int cnt) :key(move(v)), count(cnt){}

	struct comparer
	{//node节点比较器
		struct less
		{
			bool operator()(const node& lhs, const node& rhs)const
			{
				if (lhs.count == rhs.count) //如果计数值相等
					return lhs.key < rhs.key; //则比较键值
				return lhs.count < rhs.count; //否则
			}
		};

		struct greater
		{
			bool operator()(const node& lhs, const node& rhs)const
			{
				if (lhs.count == rhs.count)
					return lhs.key > rhs.key;
				return lhs.count > rhs.count;
			}	
		};
	};
};

struct treeNode :public node
{//树节点类型，继承于node
	treeNode *parent;
	treeNode *next = nullptr;
	unordered_map<string, treeNode*> children; //孩子列表，用字典方便访问
	treeNode(const string& v, int cnt, treeNode *par = nullptr) :
		node(v, cnt), parent(par), children(){}
};

struct headNode
{//头指针列表节点
	int count;
	treeNode *ptreenode = nullptr;//链接树中具有相同键值的树节点
	treeNode *tail = nullptr; //链表尾节点
	headNode(int cnt) :count(cnt){}
};

class FPTree
{//FP树
private:
	unordered_map<string, headNode*> head; //头指针列表
	treeNode *root;
	int eps; //计数阈值
private:
	void updateTree(const vector<string>&, int, treeNode*, int); 
	void mineTree(string&, unordered_map<string, int>&);
	void findPrefixPath(const string&,unordered_map<string,int>&);
	void ascendTree(treeNode*, string&)const;
	void print(treeNode*, int)const;
public:
	FPTree(int e) :root(new treeNode(string(), 0)), eps(e){}
	unordered_map<string, int> loadData(const string&)const;
	void create(const unordered_map<string, int>&);
	unordered_map<string, int> mineTree()
	{
		string prefix("(");
		unordered_map<string, int> frequent;
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
void FPTree::create(const unordered_map<string, int> &data)
{
	unordered_map<string, int> umkey_count;//记录所有的键，即每个记录项，以及计数
	string k;
	for (auto iter1 = data.begin(); iter1 != data.end(); ++iter1) //第一次扫描整个数据集
	{//每条记录都是字符串对象，采用空格分开
		istringstream stream(iter1->first); //初始化一个字符串输入流
		while (stream >> k)
		{//逐一获得记录项
			if (umkey_count.find(k) == umkey_count.end()) //如果不存在于头指针列表，则插入
				umkey_count.emplace(move(k), iter1->second);
			else //否则只需要累加计数值即可
				umkey_count[k] += iter1->second;
		}
	}

	for (auto iter2 = umkey_count.begin(); iter2 != umkey_count.end(); ++iter2)
	{//扫描每个记录项
		if (iter2->second >= eps) //把较高频数的记录项留下
		{
			head.emplace(iter2->first, new headNode(iter2->second));
		}
	}

	if (head.empty()) return; //如果没有出现次数超过阈值的记录项，则退出
	for (auto iter3 = data.begin(); iter3 != data.end(); ++iter3)
	{//第二次扫描整个数据集
		//对于每条记录，根据记录项和出现的总次数构造节点，然后插入到集合
		vector<node> key_count; 
		istringstream stream(iter3->first);
		while (stream >> k)
		{//对于该条记录中的每个记录项
			if (head.find(k) != head.end()) //只有出现次数较高
				key_count.emplace_back(node(k, head[k]->count)); //才会被插入到集合中
		}
		if (!key_count.empty())
		{//若最终该条记录中有出现次数较多的项
			sort(key_count.begin(), key_count.end(), node::comparer::greater());
			vector<string> order;
			//则按照出现次数从高到低的顺序提取出来
			for_each(key_count.begin(), key_count.end(), [&order](const node &arg){order.emplace_back(arg.key); });
			updateTree(order, 0, root, iter3->second);//用于更新树
		}
	}
}

//根据order中元素的顺序，从节点r开始向下更新树
void FPTree::updateTree(const vector<string> &order, int index, treeNode *r, int count)
{
	if (r->children.find(order[index]) != r->children.end()) //如果当前项是r的孩子
		r->children[order[index]]->count += count;//则直接更新计数值
	else
	{
		treeNode *p = new treeNode(order[index], count, r); //否则构造节点
		r->children.emplace(order[index], p); //插入到孩子列表
		if (!(head[order[index]]->ptreenode))//同时更新头指针列表
			head[order[index]]->ptreenode = p;
		else 
			head[order[index]]->tail->next = p;
		head[order[index]]->tail = p;
	}
	if ((static_cast<int>(order.size()) - index) > 1) //如果order总还有余项
		updateTree(order, index + 1, r->children[order[index]], count);//则继续递归向下更新树
}

//从构造好的树中挖掘频繁项集
void FPTree::mineTree(string &prefix, unordered_map<string,int> &frequent)
{
	//头指针列表head中存储的实际是单频繁项。根据项值和计数值构造节点，插入到集合
	//不同于建树时，此时是按照计数值从低到高顺序排序
	vector<node> key_count;
	for (auto iter = head.begin(); iter != head.end(); ++iter)
		key_count.emplace_back(node(iter->first, iter->second->count));
	sort(key_count.begin(), key_count.end(), node::comparer::less());
	for (auto iter = key_count.begin(); iter != key_count.end(); ++iter)
	{//对于每一个单频繁项
		prefix.append(iter->key);
		frequent.emplace(prefix + ')',iter->count); //发现一个频繁项
		unordered_map<string, int> subdata;

		//以当前单频繁项为尾，在树中上溯挖掘条件基，也就是在已出现prefix的情况下挖掘记录
		findPrefixPath(iter->key, subdata); 
		FPTree subtree(eps);
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
void FPTree::findPrefixPath(const string& tail, unordered_map<string, int> &paths)
{
	treeNode *first = head[tail]->ptreenode;
	while (first)
	{//对于当前键为tail的节点
		string prefix;
		ascendTree(first, prefix);//由此上溯到根获得条件基
		if (!prefix.empty())//如果条件基存在
			paths.emplace(move(prefix), first->count);
		first = first->next;
	}
}

//自当前节点开始上溯到根，并顺序记下节点键值，构成一个条件基
void FPTree::ascendTree(treeNode *curr, string &prefix)const
{
	treeNode *par = curr->parent; //条件基不包括当前节点
	bool is_int = is_same<string, int>::value;
	while (par->parent)
	{
		prefix.append(par->key);
		prefix.push_back(' ');
		par = par->parent;
	}
	if (!prefix.empty()) //删除末尾的空格
		prefix.resize(prefix.size() - 1);
}

//打印FP树
void FPTree::print(treeNode *r, int depth = 1)const
{
	string right_indent(depth, ' '); //用于缩进，表现出树的层次结构
	cout << right_indent << r->key << ":" << r->count << endl;
	for (auto iter = r->children.begin(); iter != r->children.end(); ++iter)
		print(iter->second, depth + 1);
}

//清空树
void FPTree::clear()
{
	for (auto iter = head.begin(); iter != head.end(); ++iter)
	{//由于头指针列表是一个单链表集合，每条链表都是键值相同的节点的集合，因而可以通过
		//释放所有链表来达到清空树的目的
		treeNode *first = iter->second->ptreenode, *r = nullptr;
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
unordered_map<string, int> FPTree::loadData(const string& filename)const
{
	unordered_map<string, int> data;
	const size_t SIZE = 32 * 1024 * 1024;
	char* buf = new char[SIZE];
	FILE *fp;
	fopen_s(&fp, filename.c_str(), "rb");
	size_t len = fread_s(buf, SIZE, 1, SIZE, fp);
	buf[len] = '\0';
	string line;
	for (char* ptr = buf, *first = buf; static_cast<size_t>(ptr - buf) <= len; ++ptr)
	{
		if (*ptr == '\n' || (buf + len) == ptr)
		{
			if (*(ptr - 1) == '\r')
				line.append(first, ptr - 1);
			else line.append(first, ptr);
			//将数据集转化为字典，这样做主要是为了方便挖掘频繁集时递归构造子FP树
			data.emplace(move(line), 1);
			first = ptr + 1;
		}
	}
	delete buf;
	return data;
}

const string filename = "data.txt";
//大数据集，百万
//const string filename = "C:\\Users\\png\\Desktop\\machinelearninginaction\\Ch12\\kosarak.dat";

int main()
{
	//FPTree tree(100000);
	FPTree tree(3);
	unordered_map<string, int> data = tree.loadData(filename);
	tree.create(data);
	tree.print();
	unordered_map<string, int> frequent = tree.mineTree();
	for (auto i = frequent.begin(); i != frequent.end(); ++i)
		cout << i->first << "----------------" << i->second << endl;
	getchar();
	return 0;
}