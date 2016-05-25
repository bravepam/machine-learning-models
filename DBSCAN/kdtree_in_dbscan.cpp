#include<algorithm>
#include<vector>
#include<queue>
#include<stack>
#include"kdtree_in_dbscan.h"

void kdtree::create(std::vector<size_t>& data_indexes, size_t start, size_t end, node*& r, size_t h)
{
	const size_t dim = h % m_D;//给节点应依据哪个维的数据进行分割
	size_t index = splitData(data_indexes, dim, start, end);//获得分割点的索引
	r = new node(data_indexes[index], dim);//创建特征
	if (start < index)//若左边还有数据
		create(data_indexes, start, index - 1, r->left, h + 1);//则递归创建左子树
	if (index < end)
		create(data_indexes, index + 1, end, r->right, h + 1);
}

size_t kdtree::splitData(std::vector<size_t>& data_indexes, size_t dim, size_t start, size_t end)
{
	std::vector<double> pivots;
	pivots.reserve(end - start + 1);
	for (size_t i = start; i <= end; ++i)//提取dim维数据
		pivots.emplace_back(pdb->m_points[data_indexes[i]].m_value[dim]);
	nth_element(pivots.begin(), pivots.begin() + pivots.size() / 2, pivots.end());
	double pivot = *(pivots.begin() + pivots.size() / 2);//获得中值，以其为枢轴
	size_t mid = (end - start + 1) / 2 + start;
	bool meet_pivot = false;
	while (start < mid && end > mid)
	{//分割点集
		while (start < mid && pdb->m_points[data_indexes[start]].m_value[dim] < pivot) ++start;
		if (abs(pdb->m_points[data_indexes[start]].m_value[dim] - pivot)
			<= std::numeric_limits<double>::epsilon() && !meet_pivot)
		{
			std::swap(data_indexes[start], data_indexes[mid]);
			meet_pivot = true;
			continue;
		}
		if (start == mid) break;
		while (mid < end && pdb->m_points[data_indexes[end]].m_value[dim] > pivot) --end;
		std::swap(data_indexes[start], data_indexes[end]);
	}
	return mid;//获得分割点索引
}

void kdtree::knn(node* r, size_t dot_index, size_t k,
	std::priority_queue<double>& kdists, double& maxdist)const
{
	std::stack<node*> path;
	node *pnrt = nullptr, *curr = nullptr;
	searchToLeaf(r, dot_index, path);
	while (!path.empty())
	{
		curr = path.top();
		path.pop();
		const double dist = pdb->distance(pdb->m_points[dot_index], pdb->m_points[curr->m_dot_index]);
		if (k > 0)
		{//当获得的点不足k个时，直接插入，无需判断
			kdists.emplace(dist);
			maxdist = maxdist > dist ? maxdist : dist;//同时记下这些点中的最大距离
			--k;
		}
		else if (dist < maxdist)
		{//否则，若当前点的距离比最大距离小
			kdists.pop();
			kdists.emplace(dist);//插入当前点
			maxdist = kdists.top();//更新最大距离
		}
		if (k > 0 || abs(pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] -
			pdb->m_points[dot_index].m_value[curr->m_split_dim]) < maxdist)
		{//D为2，即2维情况下，（易推广到n维），以查询点dot为圆心，当前最小距离dist为半径画圆，
			//判断节点curr的特征点的split_dim维超平面是否与该圆相交。若相交，则说明curr的另一子空间可能存
			//在更近的特征点，否则跳出
			if (pdb->m_points[dot_index].m_value[curr->m_split_dim] <
				pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] && curr->right)
				knn(curr->right, dot_index, k, kdists, maxdist);
			if (pdb->m_points[dot_index].m_value[curr->m_split_dim] >=
				pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] && curr->left)
				knn(curr->left, dot_index, k, kdists, maxdist);
		}
	}
}

void kdtree::knnInRadius(node* r, size_t dot_index, double radius, std::vector<size_t>& neighbors)const
{
	std::stack<node*> path;
	searchToLeaf(r, dot_index, path);
	node* curr = nullptr;
	while (!path.empty())
	{
		curr = path.top();
		path.pop();
		const double dist = pdb->distance(pdb->m_points[dot_index], pdb->m_points[curr->m_dot_index]);
		if (dist <= radius && curr->m_dot_index != dot_index) //不能把自己算入
			neighbors.emplace_back(curr->m_dot_index);
		if (abs(pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] -
			pdb->m_points[dot_index].m_value[curr->m_split_dim]) < radius)
		{//同上
			if (pdb->m_points[dot_index].m_value[curr->m_split_dim] <
				pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] && curr->right)
				knnInRadius(curr->right, dot_index, radius, neighbors);
			if (pdb->m_points[dot_index].m_value[curr->m_split_dim] >=
				pdb->m_points[curr->m_dot_index].m_value[curr->m_split_dim] && curr->left)
				knnInRadius(curr->left, dot_index, radius, neighbors);
		}
	}
}

void kdtree::searchToLeaf(node* r, size_t dot_index, std::stack<node*>& path)const
{
	while (r != nullptr)
	{//从根开始，一直搜寻到叶子，找到dot所在的最小子空间
		path.emplace(r);
		if (pdb->m_points[dot_index].m_value[r->m_split_dim] <
			pdb->m_points[r->m_dot_index].m_value[r->m_split_dim])
			r = r->left;
		else r = r->right;
	}
}

void kdtree::clear(node* r)
{
	if (!r)return;
	if (r->left)
		clear(r->left);
	if (r->right)
		clear(r->right);
	delete r;
}