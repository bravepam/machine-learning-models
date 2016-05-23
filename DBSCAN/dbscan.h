#ifndef _DBSCAN_H
#define _DBSCAN_H

#include<vector>
#include<cmath>
#include<cassert>
#include<string>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<queue>
#include<functional>
//#include<iostream>

//DBSCAN聚类算法
class dbscan
{
private:
	enum datatype{ CORE, BOUNDARY, NOISE }; //点类型，分别为核心点、边界点、噪声点
	//数据点
	struct point
	{
		const std::vector<double> m_value; //点值
		std::vector<size_t> m_neighbors; //邻域内的其他点，只存索引
		datatype m_type = NOISE;
		int m_cluster_id = -1;
		bool m_visited = false;
		point(std::vector<double>&& v) :m_value(std::move(v)){}
	};

	//保存聚类结果
	struct clusters
	{
		std::vector<std::vector<int>> m_clusters; //簇
		std::vector<size_t> m_boundary; //边界点集
		std::vector<size_t> m_noise; //噪声点集
		std::vector<double> m_kdists; //每个数据点的k-距离，即第k个最小距离

		//以下两个函数是将聚类结果写到文件，然后调用python程序进行绘图
		//以可视化聚类结果。通过改变writeKDists的K值（dbscan中的m_minpts）可以找到最佳的阈值和半径

		//将聚类结果写入文件，先写簇大小，在写包含的点
		void writeClusters(const std::string&, const dbscan&)const;
		//将k-距离排序后写入文件
		void writeKDists(const std::string&);
	};
private:
	friend struct clusters;
	std::vector<point> m_points; //数据点集
	const double m_radius; //邻域半径
	const size_t m_minpts; //邻域内点数阈值
	const size_t m_dim; //数据点维度
	size_t m_cluster_num; //最终簇数
	clusters m_res; //聚类结果

private:
	//计算数据点距离，也可以用其他度量方法
	double distance(const point& lhs, const point& rhs)const
	{
		assert(lhs.m_value.size() == rhs.m_value.size());
		double temp = 0.0;
		for (int i = 0; i != lhs.m_value.size(); ++i)
			temp += (lhs.m_value[i] - rhs.m_value[i]) * (lhs.m_value[i] - rhs.m_value[i]);
		return sqrt(temp);
	}

	//基于当前核心点递归聚类
	void coreClustering(int, int);
	//筛选核心点
	void findCorePoints();

	dbscan(const dbscan&);
	dbscan& operator=(const dbscan&);
public:
	//该构造函数只用于寻找最佳半径和k值时使用
	dbscan(size_t k, size_t d) :m_radius(0.), m_minpts(k), m_dim(d){}
	dbscan(double r, size_t m, size_t d) :m_radius(r), m_minpts(m), m_dim(d){}
	size_t getClustersNum()const
	{
		return m_cluster_num;
	}
	size_t getPointsNum()const
	{
		return m_points.size();
	}
	void readPoints(const std::string&);
	//以下两个函数用于计算k-距离，然后调用python程序绘制，以找出最佳的k和半径
	void computeKdists();
	void drawKdists();

	void clustering();
	//将最终的聚类结果写入文件，按照点ID，所属簇ID，数据点值，邻域内数据点索引的顺序
	void writePoints(const std::string&);
	//绘制聚类的结果
	void drawClusters();
};

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
}

void dbscan::computeKdists()
{
	m_res.m_kdists.reserve(getPointsNum());
	double dist = 0.0;
	for (int i = 0; i != m_points.size(); ++i)
	{
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
	for (int i = 0; i != m_points.size(); ++i)
	{
		for (int j = 0; j != m_points.size(); ++j)
		{
			if (i == j)
				continue;
			dist = distance(m_points[i], m_points[j]);
			if (dist <= m_radius)//如果在半径内，则说明在点i的邻域内
				m_points[i].m_neighbors.emplace_back(j);//记下索引
		}
		if (m_points[i].m_neighbors.size() >= m_minpts)//超过点数阈值
			m_points[i].m_type = CORE;//则为核心点
	}
}

void dbscan::clustering()
{
	findCorePoints();
	int cluster_id = 0;
	for (int i = 0; i != m_points.size(); ++i)
	{
		if (m_points[i].m_type == CORE && !m_points[i].m_visited)
		{//该点为核心点，且没有访问过（没有被纳入任何一个簇）
			coreClustering(i, cluster_id);//则对该簇进行拓展
			++cluster_id;
		}
	}
	m_cluster_num = cluster_id;//得到簇数
}

void dbscan::coreClustering(int pid, int cluster_id)
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
	for (int i = 0; i != m_points[pid].m_neighbors.size(); ++i)
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
	for (int i = 0; i != m_points.size(); ++i)
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
		for (int j = 0; j != m_points[i].m_value.size(); ++j)
			ostream << m_points[i].m_value[j] << ',';
		ostream << ']' << std::endl << "neighbors: ";
		for (int k = 0; k != m_points[i].m_neighbors.size(); ++k)
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
	//double sum = 0.0;
	//for_each(m_kdists.begin(), m_kdists.end(), [&sum](const double& v){sum += v; });
	//std::cout << sum << std::endl;
	for (int i = 0; i != m_kdists.size(); ++i)
		outfile << m_kdists[i] << '\t';//再写
	outfile.close();
}

void dbscan::clusters::writeClusters(const std::string& filename, const dbscan& db)const
{
	std::ofstream outfile(filename);
	assert(outfile);
	//先写各个簇
	for (int i = 0; i != m_clusters.size(); ++i)
	{
		outfile << m_clusters[i].size() << std::endl;//先写簇大小
		for (int j = 0; j != m_clusters[i].size(); ++j)//再写该簇所含数据点
		{
			auto& p = db.m_points[m_clusters[i][j]].m_value;
			for (int k = 0; k != p.size(); ++k)
				outfile << p[k] << '\t';
			outfile << std::endl;
		}
	}

	//写边界点集
	outfile << m_boundary.size() << std::endl;
	for (int i = 0; i != m_boundary.size(); ++i)
	{
		auto& p = db.m_points[m_boundary[i]].m_value;
		for (int k = 0; k != p.size(); ++k)
			outfile << p[k] << '\t';
		outfile << std::endl;
	}

	//写噪声点集
	outfile << m_noise.size() << std::endl;
	for (int i = 0; i != m_noise.size(); ++i)
	{
		auto& p = db.m_points[m_noise[i]].m_value;
		for (int k = 0; k != p.size(); ++k)
			outfile << p[k] << '\t';
		outfile << std::endl;
	}
	outfile.close();
}

#endif