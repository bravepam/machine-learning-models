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
#include<string>
#include<limits>

using namespace std;

struct feature
{//数据点结构
	const vector<double> x;//点
	int cls;//类别
	feature(const vector<double> &_x, int _cls) :x(_x), cls(_cls){}
	void print()const
	{
		printf("(");
		for (int i = 0; i != x.size(); ++i)
		{
			printf("%lf", x[i]);
			if (i != x.size() - 1)
				printf(",");
		}
		printf(") cls: %d", cls);
	}
	double distance(const vector<double> &dot)const
	{//两点之间的L2距离，即欧氏距离
		double sum = 0.0f;
		for (int i = 0; i != x.size(); ++i)
			sum += (x[i] - dot[i]) * (x[i] - dot[i]);
		return sqrt(sum);
	}
};

struct node
{//kd树节点
	const feature *const pf;//指向特征点
	int split_dim;//从该特征点分割时所采用的维度
	node *left = nullptr;
	node *right = nullptr;
	node(const feature *const _pf, int sd) :pf(_pf), split_dim(sd){}
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
	const int D;//D值，即数据点的维数
private:
	//递归创建KD树
	void create(vector<vector<double>>&, vector<int>&, int, int, node *&, int);
	//分割某一范围的数据点，返回分割点索引
	int splitDot(vector<vector<double>>&, vector<int>&, int, int, int);
	void print(node*, int)const;
	//递归查询最近邻
	node* nearest(node*, double&, const vector<double>&)const;
	void clear(node*);
	void searchToLeaf(node*, const vector<double>&, stack<node*>&)const;
	//递归查询knn
	void kNN(node*, multimap<double, node*>&, double&, const vector<double>&, int)const;
	//递归查询以查询点为圆心，一定半径内的所有点
	void kNNInRadius(node*, vector<vector<double>>&, vector<double>&, const vector<double>&, double)const;
public:
	KDTree(int _d) :D(_d), root(nullptr){}
	void create(vector<vector<double>> &dots, vector<int> &category)
	{
		create(dots, category, 0, dots.size() - 1, root, 0);
	}
	void print()const
	{
		if (root != nullptr)
			print(root, 0);
	}
	double nearest(const vector<double>& dot, vector<double>& nrt)const
	{
		double mindist = numeric_limits<double>::max();
		const node* const pnrt = nearest(root, mindist, dot);
		nrt = pnrt->pf->x;
		return mindist;
	}
	void kNN(const vector<double>& dot, int k, vector<vector<double>>& knns, vector<double>& dists)const
	{
		multimap<double, node*> mknn;
		double maxdist = 0.0;
		kNN(root, mknn, maxdist, dot, k);
		for (auto it = mknn.begin(); it != mknn.end(); ++it)
		{
			knns.emplace_back(it->second->pf->x);
			dists.emplace_back(it->first);
		}
	}
	void kNNInRadius(const vector<double>& dot, double radius, vector<vector<double>>& knns, vector<double>& dists)const
	{
		kNNInRadius(root, knns, dists, dot, radius);
	}
	void clear(){ clear(root); }
	~KDTree()
	{
		clear(root);
	}
};

void KDTree::create(vector<vector<double>> &dots, vector<int> &category, int start, int end, node *&r, int h)
{//递归创建kd树
	/*
	 *dots: 特征点集
	 *category: 对应的类别
	 *start,end: 当前可用于创建子树的点集范围
	 *r: 子树根
	 *h: 当前节点的深度
	 */
	const int dim = h % D;//给节点应依据哪个维的数据进行分割
	int index = splitDot(dots, category, dim, start, end);//获得分割点的索引
	feature *pf = new feature(dots[index], category[index]);//创建特征
	r = new node(pf, dim);//以该特征创建节点
	if (start < index)//若左边还有数据
		create(dots, category, start, index - 1, r->left, h + 1);//则递归创建左子树
	if (index < end)
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

int KDTree::splitDot(vector<vector<double>> &dots, vector<int> &category, int dim, int start, int end)
{//获得以第dim维数据为依据的分割点索引
	vector<double> pivots;
	pivots.reserve(end - start + 1);
	for (int i = start; i <= end; ++i)//提取dim维数据
		pivots.emplace_back(dots[i][dim]);
	nth_element(pivots.begin(), pivots.begin() + pivots.size() / 2, pivots.end());
	double pivot = *(pivots.begin() + pivots.size() / 2);//获得中值，以其为枢轴
	int mid = (end - start + 1) / 2 + start;
	bool meet_pivot = false;
	while (start < mid && end > mid)
	{//分割点集
		while (start < mid && dots[start][dim] < pivot) ++start;
		if (abs(dots[start][dim] - pivot) <= numeric_limits<double>::epsilon() && !meet_pivot)
		{
			dots[start].swap(dots[mid]);
			std::swap(category[start], category[mid]);
			meet_pivot = true;
			continue;
		}
		if (start == mid) break;
		while (mid < end && dots[end][dim] > pivot) --end;
		dots[start].swap(dots[end]);
		std::swap(category[start], category[end]);
	}
	return mid;//获得分割点索引
}

void KDTree::print(node *r, int indent) const
{
	for (int i = 0; i != indent; ++i)
		printf("\t");
	r->print();
	if (r->left != nullptr)
		print(r->left, indent + 1);
	if (r->right != nullptr)
		print(r->right, indent + 1);
}

node* KDTree::nearest(node* r,double& mindist,const vector<double>& dot)const
{//最近邻算法，获得与dot最近的点
	stack<node*> path;//存储搜索路径
	node *pnrt = nullptr, *curr = nullptr;
	searchToLeaf(r, dot, path);
	while (!path.empty())
	{//回溯
		curr = path.top();
		path.pop();
		const double dist = curr->pf->distance(dot);//节点r的特征点与dot的距离
		if (dist < mindist)
		{//如果更近
			pnrt = curr;//则更新最近点
			mindist = dist;
		}
		node* sub_pnrt = nullptr;
		if (abs(curr->pf->x[curr->split_dim] - dot[curr->split_dim]) < mindist)
		{//D为2，即2维情况下（易推广到n维），以查询点dot为圆心，当前最小距离dist为半径画圆，
			//判断节点curr的特征点的split_dim维超平面是否与该圆相交。若相交，则说明curr的另一子空间可能存
			//在更近的特征点
			if (dot[curr->split_dim] < curr->pf->x[curr->split_dim] && curr->right)
				sub_pnrt = nearest(curr->right, mindist, dot);
			if (dot[curr->split_dim] >= curr->pf->x[curr->split_dim] && curr->left)
				sub_pnrt = nearest(curr->left, mindist, dot);
		}
		if (sub_pnrt) //有效则说明在另一子空间确实存在更近的点
			pnrt = sub_pnrt;
	}
	return pnrt;
}

void KDTree::searchToLeaf(node* r, const vector<double>& dot, stack<node*>& path)const
{
	while (r != nullptr)
	{//从根开始，一直搜寻到叶子，找到dot所在的最小子空间
		path.emplace(r);
		if (dot[r->split_dim] < r->pf->x[r->split_dim])
			r = r->left;
		else r = r->right;
	}
}

void KDTree::kNN(node* r, multimap<double, node*>& mknn, double& maxdist, const vector<double> &dot, int k)const
{//k近邻算法，找到k个与dot最近的点，并返回相对应的距离
	stack<node*> path;
	node *pnrt = nullptr, *curr = nullptr;
	searchToLeaf(r, dot, path);
	while (!path.empty())
	{
		curr = path.top();
		path.pop();
		const double dist = curr->pf->distance(dot);
		if (k > 0)
		{//当获得的点不足k个时，直接插入，无需判断
			mknn.emplace(dist, curr);
			maxdist = maxdist > dist ? maxdist : dist;//同时记下这些点中的最大距离
			--k;
		}
		else if (dist < maxdist)
		{//否则，若当前点的距离比最大距离小
			mknn.erase(--mknn.end());//则删掉k个点中的距离最大者
			mknn.emplace(dist, curr);//插入当前点
			maxdist = (--mknn.end())->first;//更新最大距离
		}
		if (k > 0 || abs(curr->pf->x[curr->split_dim] - dot[curr->split_dim]) < maxdist)
		{//D为2，即2维情况下，（易推广到n维），以查询点dot为圆心，当前最小距离dist为半径画圆，
			//判断节点curr的特征点的split_dim维超平面是否与该圆相交。若相交，则说明curr的另一子空间可能存
			//在更近的特征点，否则跳出
			if (dot[curr->split_dim] < curr->pf->x[curr->split_dim] && curr->right)
				kNN(curr->right, mknn, maxdist, dot, k);
			if (dot[curr->split_dim] >= curr->pf->x[curr->split_dim] && curr->left)
				kNN(curr->left, mknn, maxdist, dot, k);
		}
	}
}

void KDTree::kNNInRadius(node* r, vector<vector<double>>& knns, vector<double>& dists, const vector<double>& dot, double radius)const
{
	stack<node*> path;
	searchToLeaf(r, dot, path);
	node* curr = nullptr;
	while (!path.empty())
	{
		curr = path.top();
		path.pop();
		const double dist = curr->pf->distance(dot);
		if (dist < radius)
		{
			knns.emplace_back(curr->pf->x);
			dists.emplace_back(dist);
		}
		if (abs(curr->pf->x[curr->split_dim] - dot[curr->split_dim]) < radius)
		{
			if (dot[curr->split_dim] < curr->pf->x[curr->split_dim] && curr->right)
				kNNInRadius(curr->right, knns, dists, dot, radius);
			if (dot[curr->split_dim] >= curr->pf->x[curr->split_dim] && curr->left)
				kNNInRadius(curr->left, knns, dists, dot, radius);
		}
	}
}

#endif