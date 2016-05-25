#ifndef _KDTREE_IN_DBSCAN_H
#define _KDTREE_IN_DBSCAN_H

#include<vector>
#include<stack>
#include<queue>
#include<fstream>
#include"dbscan.h"

class dbscan;

//适配dbscan算法的kdtree，用于快读查找k-近邻
class kdtree
{
private:
	struct node
	{
		const size_t m_dot_index; //节点所存储的数据点索引
		const size_t m_split_dim; //按哪个维度分割
		node* right = nullptr;
		node* left = nullptr;
		node(size_t di, size_t sd) :m_dot_index(di), m_split_dim(sd){}
	};

private:
	node* root;
	const size_t m_D;
	const dbscan* const pdb; //专门用来获取数据点信息

	void knn(node*, size_t, size_t, std::priority_queue<double>&, double&)const;
	void knnInRadius(node*, size_t, double, std::vector<size_t>&)const;
	void create(std::vector<size_t>&, size_t, size_t, node*&, size_t);
	size_t splitData(std::vector<size_t>&, size_t, size_t, size_t);
	void searchToLeaf(node*, size_t, std::stack<node*>&)const;
	void clear(node*);
public:
	kdtree(const dbscan* const _pdb, size_t d) :pdb(_pdb), root(nullptr), m_D(d){}
	double knn(size_t dot_index, size_t k)const
	{
		double maxdist = 0.0;
		std::priority_queue<double> kdists; //k个近邻的距离
		knn(root, dot_index, k, kdists, maxdist);
		return maxdist;
	}
	void knnInRadius(size_t dot_index, double radius, std::vector<size_t>& neighbors)const
	{
		knnInRadius(root, dot_index, radius, neighbors);
	}
	void create(std::vector<size_t>& data_indexes)
	{
		create(data_indexes, 0, data_indexes.size() - 1, root, 0);
	}
	void clear()
	{
		clear(root);
	}
	~kdtree()
	{

		clear(root);
	}
};

#endif