/*kd树，k近邻（k-nearest neighbor）算法最主要的数据结构
 *用于在高维空间搜索特定的特征点，类似于BST，不过每个节点
 *的分割依据的是特征点某一维的数据集合的中值，从根到叶子，依次
 *循环依据各维。
 *由于建立后主要用于查询，因而主要算法只实现了最近邻点查询（nearest）
 *和k近邻点查询(k-nearest neighbor)，其他的诸如删除，修改，增加没实现。
 */

#ifndef KDTREE_H
#define KDTREE_H

#include<vector>
#include<stack>
#include<map>
#include<algorithm>

using namespace std;

struct feature
{//数据点结构
	vector<float> x;//点
	int cls;//类别
	feature(const vector<float> &_x, int _cls) :x(_x), cls(_cls){}
	void print()const
	{
		printf("(");
		for (int i = 0; i != x.size(); ++i)
		{
			printf("%f", x[i]);
			if (i != x.size() - 1)
				printf(",");
		}
		printf(") cls: %d", cls);
	}
	float distance(const vector<float> &dot)const
	{//两点之间的L2距离，即欧氏距离
		float sum = 0.0f;
		for (int i = 0; i != x.size(); ++i)
			sum += (x[i] - dot[i]) * (x[i] - dot[i]);
		return sqrt(sum);
	}
};

struct node
{//kd树节点
	feature *pf;//指向特征点
	int split_dim;//从该特征点分割时所采用的维度
	node *left = nullptr;
	node *right = nullptr;
	node(feature *const _pf, int sd) :pf(_pf), split_dim(sd){}
	void print()const
	{
		pf->print();
		printf(" split dim: %d\n", split_dim);
	}
};

class KDTree
{//kd树
private:
	node *root;
	int k;//k值，即数据点的维数
private:
	void create(vector<vector<float>>&, vector<int>&, int, int, node *&, int);//创建树
	int splitDot(vector<vector<float>>&, vector<int>&, int, int, int);
	void print(node*)const;
	void clear(node*);
	void enterSubSpace(stack<node*> &path, node *r, float dist, bool pass_root, vector<float> &dot)
	{//判断该进入r节点的哪个子空间
		if (abs(r->pf->x[r->split_dim] - dot[r->split_dim]) < dist)
		{//k为2，即2维情况下，（易推广到n维），以查询点dot为圆心，当前最小距离dist为半径画圆，
		 //判断节点r的特征点的split_dim维超平面是否与该圆相交。若相交，则说明r的另一子空间可能存
		 //在更近的特征点，否则跳出
			if (!pass_root)
			{//没有跨过根节点的时候，只需进入某一个子空间
				if (dot[r->split_dim] < r->pf->x[r->split_dim] && r->right != nullptr)
					path.push(r->right);
				if (dot[r->split_dim] > r->pf->x[r->split_dim] && r->left != nullptr)
					path.push(r->left);
			}
			else
			{//否则两个都要进
				if (r->right != nullptr) path.push(r->right);
				if (r->left != nullptr) path.push(r->left);
			}
		}
	}
public:
	KDTree(int _k) :k(_k), root(nullptr){}
	void create(vector<vector<float>> &dots, vector<int> &category)
	{
		create(dots, category, 0, dots.size() - 1, root, 0);
	}
	void print()const
	{
		if (root != nullptr)
			print(root);
	}
	float nearest(vector<float>&, vector<float>&);
	void kNN(vector<float>&, int, vector<vector<float>>&, vector<float>&);
	void clear(){ clear(root); }
	~KDTree()
	{
		clear(root);
	}
};

void KDTree::create(vector<vector<float>> &dots, vector<int> &category, int start, int end, node *&r, int h)
{//递归创建kd树，也只能递归创建。
	/*
	 *dots: 特征点集
	 *category: 对应的类别
	 *start,end: 当前可用于创建子树的点集范围
	 *r: 子树根
	 *h: 当前节点的深度
	 */
	int dim = h % k;//给节点应依据哪个维的数据进行分割
	int index = splitDot(dots, category, dim, start, end);//获得分割点的索引
	feature *pf = new feature(dots[index], category[index]);//创建特征
	r = new node(pf, dim);//以该特征创建节点
	if (start < index)//若左边还有数据
		create(dots, category, start, index - 1, r->left, h + 1);//则递归创建左子树
	if (index < end)//同上
		create(dots, category, index + 1, end, r->right, h + 1);
}

void KDTree::clear(node *r)
{
	if (r == nullptr) return;
	clear(r->left);
	clear(r->right);
	delete r->pf;
	delete r;
}

int KDTree::splitDot(vector<vector<float>> &dots, vector<int> &category, int dim, int start, int end)
{//获得以第dim维数据为依据的分割点索引
	vector<float> pivots;
	for (int i = 0; i <= end; ++i)//提取dim维数据
		pivots.push_back(dots[i][dim]);
	//sort(pivots.begin(), pivots.end());
	nth_element(pivots.begin(), pivots.begin() + pivots.size() / 2, pivots.end());
	float pivot = *(pivots.begin() + pivots.size() / 2);//获得中值，以其为枢轴
	int mid = (end - start + 1) / 2 + start;
	bool meet_pivot = false;
	while (start < end)
	{//分割点集
		while (start < end && dots[start][dim] < pivot) ++start;
		if (dots[start][dim] == pivot && !meet_pivot)
		{
			dots[start].swap(dots[mid]);
			std::swap(category[start], category[mid]);
			meet_pivot = true;
			continue;
		}
		while (start < end && dots[end][dim] > pivot) --end;
		dots[start].swap(dots[end]);
		std::swap(category[start], category[end]);
	}
	return mid;//获得分割点索引
}

void KDTree::print(node *r) const
{
	r->print();
	if (r->left != nullptr)
		print(r->left);
	if (r->right != nullptr)
		print(r->right);
}

float KDTree::nearest(vector<float> &dot, vector<float> &nrt)
{//最近邻算法，获得与dot最近的点，存入nrt（NeaResT）
	stack<node*> path;//存储搜索路径
	node *r = root, *pnrt = nullptr;
	while (r != nullptr)
	{//从根开始，一直搜寻到叶子，找到dot所在的最小子空间
		path.push(r);
		if (dot[r->split_dim] < r->pf->x[r->split_dim])
			r = r->left;
		else r = r->right;
	}
	float mindist = (float)INT_MAX;
	bool pass_root = false;
	while (!path.empty())
	{//回溯
		r = path.top();
		path.pop();
		float dist = r->pf->distance(dot);//节点r的特征点与dot的距离
		if (dist < mindist)
		{//如果更近
			pnrt = r;//则更新最近点
			mindist = dist;
		}
		enterSubSpace(path, r, mindist, pass_root, dot);//判断是否需要进入另外的子空间
		if (root == r) pass_root = true;
	}
	nrt = pnrt->pf->x;//获得最近点
	return mindist;//返回距离
}

void KDTree::kNN(vector<float> &dot, int k, vector<vector<float>> &knn, vector<float> &dist)
{//k近邻算法，找到k个与dot最近的点，并返回相对应的距离
	stack<node*> path;
	node *r = root, *pnrt = nullptr;
	while (r != nullptr)
	{
		path.push(r);
		if (dot[r->split_dim] < r->pf->x[r->split_dim])
			r = r->left;
		else r = r->right;
	}
	multimap<float, node*> mknn;//存储最近的k个点
	float maxdist = 0.0f;
	bool pass_root = false;
	while (!path.empty())
	{
		r = path.top();
		path.pop();
		float dist = r->pf->distance(dot);
		if (k > 0)
		{//当获得的点不足k个时，直接插入，无需判断
			mknn.insert(map<float, node*>::value_type(dist, r));
			maxdist = maxdist > dist ? maxdist : dist;//同时记下这些点中的最大距离
			--k;
		}
		else if (dist < maxdist)
		{//否则，若当前点的距离比最大距离小
			mknn.erase(--mknn.end());//则删掉k个点中的距离最大者
			mknn.insert(map<float, node*>::value_type(dist, r));//插入当前点
			maxdist = (--mknn.end())->first;//更新最大距离
		}
		enterSubSpace(path, r, maxdist, pass_root, dot);//判断是否需要进入其他子空间
		if (root == r) pass_root = true;
	}
	for (auto it = mknn.begin(); it != mknn.end(); ++it)
	{
		knn.push_back(it->second->pf->x);
		dist.push_back(it->first);
	}
}

#endif