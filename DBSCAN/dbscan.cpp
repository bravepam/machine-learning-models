#include<cmath>
#include<vector>
#include<string>
#include<cassert>
#include<numeric>
#include<fstream>
#include<sstream>
#include<algorithm>
#include"dbscan.h"

#define KDTREE //定义则采用kd树实现的knn算法加速，否则线性扫描

void dbscan::readPoints(const std::string& filename)
{
	std::ifstream infile(filename);
	assert(infile);
	int count = 0;
	double temp;
	std::vector<double> vtemp;
	while (infile >> temp)
	{
		++count;
		vtemp.emplace_back(temp);
		if (count == m_dim)
		{
			m_points.emplace_back(point(std::move(vtemp)));
			count = 0;
		}
	}
	infile.close();
#ifdef KDTREE
	pkdt = new kdtree(this, m_dim);
	//构造kdtree
	std::vector<size_t> data_indexes(m_points.size());
	std::iota(data_indexes.begin(), data_indexes.end(), 0);
	pkdt->create(data_indexes);//构造数据集只是数据点的索引
#endif
}

void dbscan::computeKdists()
{
	m_res.m_kdists.reserve(getPointsNum());
	double dist = 0.0;
	for (size_t i = 0; i != m_points.size(); ++i)
	{
#ifdef KDTREE
		const double dist = pkdt->knn(i, m_minpts + 1);//加1是因为会把自己也加入，返回k-距离
		m_res.m_kdists.emplace_back(dist);
#else
		int m = m_minpts;
		std::priority_queue<double> priq; //最大优先级队列
		for (int j = 0; j != m_points.size(); ++j)
		{
			if (i == j) continue;
			dist = distance(m_points[i], m_points[j]);
			--m;
			if (m >= 0)
				priq.emplace(dist);
			else if (dist < priq.top())
			{
				priq.pop();
				priq.emplace(dist);
			}
		}
		m_res.m_kdists.emplace_back(priq.top());//k-距离
#endif
	}
}

void dbscan::drawKdists()
{
	const std::string filename = "k-dists.txt";//k-距离数据输出文件
	assert(!m_res.m_kdists.empty());
	m_res.writeKDists(filename);
	const std::string command = "python drawKDists.py";
	system(command.c_str());
}

void dbscan::drawClusters()
{
	const std::string filename = "clusters.txt";//聚类结果输出文件
	assert(!m_res.m_clusters.empty());
	m_res.writeClusters(filename, *this);
	const std::string command = "python drawClusters.py";
	system(command.c_str());
}

void dbscan::findCorePoints()
{
	double dist = 0.0;
	for (size_t i = 0; i != m_points.size(); ++i)
	{
#ifdef KDTREE
		pkdt->knnInRadius(i, m_radius, m_points[i].m_neighbors);
#else
		for (size_t j = 0; j != m_points.size(); ++j)
		{
			if (i == j)
				continue;
			dist = distance(m_points[i], m_points[j]);
			if (dist <= m_radius)//如果在半径内，则说明在点i的邻域内
				m_points[i].m_neighbors.emplace_back(j);//记下索引
		}
#endif
		if (m_points[i].m_neighbors.size() >= m_minpts)//超过点数阈值
			m_points[i].m_type = CORE;//则为核心点
	}
}

void dbscan::clustering()
{
	findCorePoints();
	size_t cluster_id = 0;
	for (size_t i = 0; i != m_points.size(); ++i)
	{
		if (m_points[i].m_type == CORE && !m_points[i].m_visited)
		{//该点为核心点，且没有访问过（没有被纳入任何一个簇）
			coreClustering(i, cluster_id);//则对该簇进行拓展
			++cluster_id;
		}
	}
	m_cluster_num = cluster_id;//得到簇数
}

void dbscan::coreClustering(size_t pid, size_t cluster_id)
{
	if (m_points[pid].m_visited)
		return;
	else if (m_points[pid].m_type == NOISE)
	{//coreClustering最开始的调用是由核心点调用的，因此如果出现某点为NOISE，则说明该点
		//是边界点，因为它出现在某核心点邻域内，而自己又不是核心点
		m_points[pid].m_type = BOUNDARY;
		//return;
	}
	m_points[pid].m_cluster_id = cluster_id;
	m_points[pid].m_visited = true;
	if (m_points[pid].m_type != CORE)
		return;
	for (size_t i = 0; i != m_points[pid].m_neighbors.size(); ++i)
	{
		coreClustering(m_points[pid].m_neighbors[i], cluster_id);//深度优先搜索
	}
}

void dbscan::writePoints(const std::string& filename)
{
	std::ofstream outfile(filename);
	assert(outfile);
	m_res.m_clusters.resize(m_cluster_num);//设置聚类结果中的簇大小
	outfile << "cluster numbers: " << m_cluster_num << std::endl;
	for (size_t i = 0; i != m_points.size(); ++i)
	{
		std::ostringstream ostream;
		ostream << "ID: " << i << " cluster ID: " << m_points[i].m_cluster_id
			<< " Type: ";
		switch (m_points[i].m_type)
		{
		case CORE:
			ostream << "core";
			m_res.m_clusters[m_points[i].m_cluster_id].emplace_back(i);
			break;
		case BOUNDARY:
			ostream << "boundary";
			m_res.m_clusters[m_points[i].m_cluster_id].emplace_back(i);//边界点也属于某簇
			m_res.m_boundary.emplace_back(i);
			break;
		case NOISE:
			ostream << "noise";
			m_res.m_noise.emplace_back(i);
			break;
		default:
			ostream << "unknown";
			break;
		}
		ostream << " [";
		for (size_t j = 0; j != m_points[i].m_value.size(); ++j)
			ostream << m_points[i].m_value[j] << ',';
		ostream << ']' << std::endl << "neighbors: ";
		for (size_t k = 0; k != m_points[i].m_neighbors.size(); ++k)
			ostream << m_points[i].m_neighbors[k] << '\t';
		ostream << "\n\n";
		outfile << ostream.str();
	}
	outfile.close();
}

void dbscan::clusters::writeKDists(const std::string& filename)
{
	std::ofstream outfile(filename);
	assert(outfile);
	sort(m_kdists.begin(), m_kdists.end()); //先排序
	for (size_t i = 0; i != m_kdists.size(); ++i)
		outfile << m_kdists[i] << '\t';//再写
	outfile.close();
}

void dbscan::clusters::writeClusters(const std::string& filename, const dbscan& db)const
{
	std::ofstream outfile(filename);
	assert(outfile);
	//先写各个簇
	for (size_t i = 0; i != m_clusters.size(); ++i)
	{
		outfile << m_clusters[i].size() << std::endl;//先写簇大小
		for (size_t j = 0; j != m_clusters[i].size(); ++j)//再写该簇所含数据点
		{
			auto& p = db.m_points[m_clusters[i][j]].m_value;
			for (size_t k = 0; k != p.size(); ++k)
				outfile << p[k] << '\t';
			outfile << std::endl;
		}
	}

	//写边界点集
	outfile << m_boundary.size() << std::endl;
	for (size_t i = 0; i != m_boundary.size(); ++i)
	{
		auto& p = db.m_points[m_boundary[i]].m_value;
		for (size_t k = 0; k != p.size(); ++k)
			outfile << p[k] << '\t';
		outfile << std::endl;
	}

	//写噪声点集
	outfile << m_noise.size() << std::endl;
	for (size_t i = 0; i != m_noise.size(); ++i)
	{
		auto& p = db.m_points[m_noise[i]].m_value;
		for (size_t k = 0; k != p.size(); ++k)
			outfile << p[k] << '\t';
		outfile << std::endl;
	}
	outfile.close();
}

dbscan::~dbscan()
{
	delete pkdt;
}