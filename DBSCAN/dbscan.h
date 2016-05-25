#ifndef _DBSCAN_H
#define _DBSCAN_H

#include<vector>
#include<cassert>
#include"kdtree_in_dbscan.h"

enum datatype{ CORE, BOUNDARY, NOISE }; //点类型，分别为核心点、边界点、噪声点

//DBSCAN聚类算法
class dbscan
{
private:
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
		std::vector<std::vector<size_t>> m_clusters; //簇
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
	friend class kdtree;
	std::vector<point> m_points; //数据点集
	const double m_radius; //邻域半径
	const size_t m_minpts; //邻域内点数阈值
	const size_t m_dim; //数据点维度
	size_t m_cluster_num; //最终簇数
	clusters m_res; //聚类结果
	kdtree* pkdt = nullptr;

private:
	//计算数据点距离，也可以用其他度量方法
	double dbscan::distance(const point& lhs, const point& rhs)const
	{
		assert(lhs.m_value.size() == rhs.m_value.size());
		double temp = 0.0;
		for (int i = 0; i != lhs.m_value.size(); ++i)
			temp += (lhs.m_value[i] - rhs.m_value[i]) * (lhs.m_value[i] - rhs.m_value[i]);
		return sqrt(temp);
	}

	//基于当前核心点递归聚类
	void coreClustering(size_t, size_t);
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
	~dbscan();
};

#endif