#ifndef UTIL_H
#define UTIL_H

#include<vector>
#include<random>
#include<memory>

//样本结构
struct sample
{
	std::vector<double> x;
	int y;

	template <typename U>
	sample(U&& u, int y_) :x(std::forward<U>(u)), y(y_){}
};

struct RFParams;

//每棵随机决策树所用的数据集
struct TreeDataSet
{
	std::vector<size_t> train_data; //训练集
	std::vector<size_t> oob; //带外数据集，均存储样本id
	const std::shared_ptr<const RFParams> prf;

	TreeDataSet(const std::shared_ptr<const RFParams>& p) :prf(p){}
	//利用bagging算法获得训练集
	void bagging(size_t);
	//随机树训练数据集的补集即为袋外数据，全集为随机森林的训练数据集
	const std::vector<size_t>& oobData();
	bool contains(size_t)const;
};

//终止条件结构，不用的项设为0
struct Termcriteria
{
	const double eps; //熵阈值
	const size_t num; //数据集大小阈值
	const size_t depth; //树最大深度
	const size_t iter; //迭代次数阈值
	Termcriteria(double e, size_t n, size_t d, size_t i) :
		eps(e), num(n), depth(d), iter(i){}
};

struct RFParams
{
	std::vector<sample> train_set;
	std::vector<sample> test_set;
	const size_t cls_num;
	const size_t D;
	const size_t F;
	const size_t N;
	Termcriteria tc;
	const bool calc_fte_importance;

	template <typename U>
	RFParams(U&& ts1, U&& ts2, size_t d, size_t cn, size_t f, size_t n, bool c,
		const Termcriteria& t) :train_set(std::forward<U>(ts1)), test_set(std::forward<U>(ts2)),
		cls_num(cn), D(d), F(f), N(n), tc(t), calc_fte_importance(c){}
};

extern unsigned int prev_seed;
//采用线性同余获得随机数，X(i+1) = {X(i) * A + C} mod B,此处A = 16807,C = 0,B = 2147483647(2^31 - 1)
using lce = std::linear_congruential_engine < unsigned long, 16807, 0, 2147483647 >;

extern std::mt19937 getMt19937(); //获得线性同余随机数发生器，用随机种子初始化

//有放回方式随机构造一个向量
extern void samplingWithReplacement(size_t, size_t, std::vector<size_t>&);

//无放回方式随机构造一个向量
extern void samplingNoReplacement(size_t, size_t, std::vector<size_t>&);

//参数依次为训练集，测试集，类别数目，数据维度，构造树节点时可选取的特征数，树的总数，
//训练终止条件，是否计算特征重要性
extern std::shared_ptr<RFParams> newRFParams
(	const std::vector<sample>&,
	const std::vector<sample>&,
	size_t, size_t, size_t, size_t,
	const Termcriteria&,
	bool
);

#endif